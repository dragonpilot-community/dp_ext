#!/usr/bin/env python3
from typing import List, Tuple, Dict
import time
import rtree
import math
import json
import threading
from collections import deque

import cereal.messaging as messaging
from cereal import log, custom
from openpilot.common.params import Params

from openpilot.dp_ext.selfdrive.tetood.lib.hmm_map_matcher import HMMMapMatcher
from openpilot.dp_ext.selfdrive.tetood.lib.overpass_api_helper import OverpassAPIHelper
from openpilot.dp_ext.selfdrive.tetood.lib.speed_camera_loader import SpeedCameraLoader
from openpilot.dp_ext.selfdrive.tetood.lib.taiwan_speed_camera_loader import TaiwanSpeedCameraLoader
from openpilot.dp_ext.selfdrive.tetood.lib.utils import calculate_bearing, haversine_distance, feature_is_ahead, angle_diff, calculate_predicted_position

RADIUS = 3500

FREQ = 2 # 250 ms

class TeToo:
    def __init__(self):
        self.current_position = None
        self.current_bearing = None
        self.road_network = {}
        self.traffic_signals = {}
        self.speed_cameras = {}
        self.index = rtree.index.Index()
        self.map_matcher = None
        self.overpass_helper = None

        self.current_way = None

        self.current_road_name = None
        self.current_max_speed = 0
        self.current_tags = {}

        speed_camera_source = None
        speed_camera_data_type = "osm_json"
        self.speed_camera_loader = SpeedCameraLoader(speed_camera_source, speed_camera_data_type)

        self.params = Params()

        self.fetching_thread = None
        self.data_lock = threading.Lock()
        self.current_road_network = None
        self.v_ego = 0.

        self.gps_history_max_size = 100
        self.gps_history = deque(maxlen=self.gps_history_max_size)

        self.bearing_history_max_size = 60
        self.bearing_history = deque(maxlen=self.bearing_history_max_size)

        self._frame = 0

        self.last_matched_bearing = None
        self.bearing_change_threshold = 45  # degrees

        self._taiwan_speed_camera_enabled = True#self.params.get_bool("dp_tetoo_data_taiwan_speed_camera")
        if self._taiwan_speed_camera_enabled:
            self.tw_speed_cameras = TaiwanSpeedCameraLoader().load_speed_cameras()
        else:
            self.tw_speed_cameras = None

        self.prefetched_data = None
        self.last_fetch_position = None
        last_data = self.params.get("dp_tetoo_data")
        if last_data is not None and last_data != "":
            self.prefetched_data = json.loads(last_data)
            self._build_road_network()
            self.map_matcher = HMMMapMatcher(self.current_road_network)

            last_pos = self.params.get("dp_tetoo_gps")
            if last_pos is not None and last_pos != "":
                self.last_fetch_position = json.loads(last_pos)


    def update_position(self, lat: float, lon: float, bearing: float):
        new_position = (lat, lon)
        # e.g. The system being turned off and on again in a different location
        if self.current_position:
            distance_moved = haversine_distance(self.current_position, new_position)
            if distance_moved > 50:  # Reset if we've moved more than 50 meters
                self.current_way = None
                self.bearing_history.clear()  # Reset bearing history on large movements
                self.gps_history.clear()  # Reset GPS history on large movements

        self.current_position = new_position

        # Update bearing history
        self.bearing_history.append(bearing)

        # Calculate moving average of bearing
        self.current_bearing = sum(self.bearing_history) / len(self.bearing_history)

        self.gps_history.append((lat, lon))

        self._check_and_fetch_data()

        self._map_match()
        return self._get_road_info()

    def _check_and_fetch_data(self):
        if not self.last_fetch_position or haversine_distance(self.current_position, self.last_fetch_position) > RADIUS - self._boundary_offset():
            self._fetch_data()

    def _boundary_offset(self):
        """reduce boundary at high speed"""
        return 300 if self.v_ego < 16.67 else 600

    def _fetch_data(self):
        def fetch():
            overpass_helper = OverpassAPIHelper()
            self.prefetched_data = overpass_helper.fetch_data(self.current_position[0], self.current_position[1], RADIUS, self.v_ego > 23)
            self._build_road_network()
            self.last_fetch_position = self.current_position
            self.params.put_nonblocking("dp_tetoo_gps", json.dumps(self.last_fetch_position))
            self.params.put_nonblocking("dp_tetoo_data", json.dumps(self.prefetched_data, ensure_ascii=False))

            with self.data_lock:
                self.map_matcher = HMMMapMatcher(self.current_road_network)

        if self.fetching_thread is not None and self.fetching_thread.is_alive():
            return
        self.fetching_thread = threading.Thread(target=fetch)
        self.fetching_thread.start()

    def _build_road_network(self):
        new_road_network = {}
        new_index = rtree.index.Index()
        self.traffic_signals = {}
        self.speed_cameras = {}

        for element in self.prefetched_data['elements']:
            if element['type'] == 'way':
                way_id = element['id']
                new_road_network[way_id] = {
                    'nodes': element['nodes'],
                    'tags': element.get('tags', {})
                }
            elif element['type'] == 'node':
                node_id = element['id']
                lat, lon = element['lat'], element['lon']
                new_index.insert(node_id, (lon, lat, lon, lat))
                new_road_network[node_id] = {'lat': lat, 'lon': lon}
                if element.get('tags', {}).get('highway') == 'traffic_signals':
                    self.traffic_signals[node_id] = {'lat': lat, 'lon': lon, 'tags': element.get('tags')}
                elif element.get('tags', {}).get('highway') == 'speed_camera':
                    self.speed_cameras[node_id] = {'lat': lat, 'lon': lon, 'tags': element.get('tags')}

        # Add traffic signals to the index
        for node_id, node_data in self.traffic_signals.items():
            new_index.insert(node_id, (node_data['lon'], node_data['lat'], node_data['lon'], node_data['lat']))

        # Load speed cameras
        self.speed_cameras = self.speed_camera_loader.load_speed_cameras(self.prefetched_data)

        # Add speed cameras to the index
        for camera_id, camera_data in self.speed_cameras.items():
            new_index.insert(camera_id, (camera_data['lon'], camera_data['lat'], camera_data['lon'], camera_data['lat']))

        # taiwan specific camera
        if self._taiwan_speed_camera_enabled:
            for camera_id, camera_data in self.tw_speed_cameras.items():
                new_index.insert(camera_id, (camera_data['lon'], camera_data['lat'], camera_data['lon'], camera_data['lat']))
                self.speed_cameras[camera_id] = camera_data

        with self.data_lock:
            self.road_network = new_road_network
            self.index = new_index
            self.current_road_network = self.road_network

    def _map_match(self):
        if len(self.gps_history) < 2:
            return

        force_match = False
        if self.last_matched_bearing is not None:
            if angle_diff(self.current_bearing, self.last_matched_bearing) > self.bearing_change_threshold:
                force_match = True

        if force_match or self._frame % 2 == 0:
            if self.map_matcher is None:
                return

            # Use the entire GPS history for matching
            matched_way = self.map_matcher.match(self.gps_history)

            if matched_way is not None:
                self.current_way = matched_way
                self.last_matched_bearing = self.current_bearing  # Update last matched bearing
            else:
                self.current_way = None

        self._frame += 1

    def _get_candidate_roads(self, gps_points: List[Tuple[float, float]]) -> List[int]:
        candidate_roads = []
        current_lat, current_lon = gps_points[-1]  # Use the most recent GPS point

        nearby_nodes = list(self.index.nearest((current_lon, current_lat, current_lon, current_lat), 5))

        # Calculate bearing change
        bearing_change = 0
        if len(self.bearing_history) >= 2:
            recent_bearing = self.bearing_history[-1]
            previous_bearing = self.bearing_history[-2]
            bearing_change = angle_diff(recent_bearing, previous_bearing)

        # Adjust bearing weight based on bearing change
        bearing_weight = 20 if bearing_change > self.bearing_change_threshold else 10

        for node in nearby_nodes:
            for way_id, way_data in self.road_network.items():
                if isinstance(way_id, int) and 'nodes' in way_data and node in way_data['nodes']:
                    # Calculate the bearing of the road segment
                    node_index = way_data['nodes'].index(node)
                    if node_index < len(way_data['nodes']) - 1:
                        next_node = way_data['nodes'][node_index + 1]
                    elif node_index > 0:
                        next_node = way_data['nodes'][node_index - 1]
                    else:
                        continue  # Skip if it's a single-node way

                    node1_lat, node1_lon = self.road_network[node]['lat'], self.road_network[node]['lon']
                    node2_lat, node2_lon = self.road_network[next_node]['lat'], self.road_network[next_node]['lon']

                    road_bearing = calculate_bearing((node1_lat, node1_lon), (node2_lat, node2_lon))

                    # Calculate the difference between road bearing and vehicle bearing
                    bearing_diff = angle_diff(road_bearing, self.current_bearing)

                    # Calculate distance to the road
                    distance = haversine_distance((current_lat, current_lon), (node1_lat, node1_lon))

                    # Check if the road name matches the current road name
                    road_name = way_data.get('tags', {}).get('name')
                    name_match_bonus = 0
                    if road_name == self.current_road_name:
                        name_match_bonus = -0  # Reduce the score for roads with matching names

                    candidate_roads.append((way_id, distance, bearing_diff, name_match_bonus))

        # Sort candidate roads by a combination of distance, bearing difference, and name match
        candidate_roads.sort(key=lambda x: x[1] + x[2] * bearing_weight + x[3])  # Adjusted bearing weight

        return [road[0] for road in candidate_roads[:5]]  # Return top 5 candidates

    def _get_road_info(self):
        if self.current_way is None:
            return {
                'id': None,
                'name': self.current_road_name,
                'maxspeed': self.current_max_speed,
                'tags': self.current_tags,
                'confidence': 0.0
            }

        way_data = self.road_network.get(self.current_way, {})
        new_tags = way_data.get('tags', {})
        new_road_name = new_tags.get('name', None)
        new_max_speed = new_tags.get('maxspeed', '0')

        # Only update if the new name is different
        if new_road_name != None and new_road_name != self.current_road_name:
            self.current_road_name = new_road_name
            self.current_max_speed = new_max_speed
            self.current_tags = new_tags

        return {
            'id': self.current_way,
            'name': self.current_road_name,
            'maxspeed': self.current_max_speed,
            'tags': self.current_tags,
            # 'confidence': self.current_way_confidence
        }

    def _check_feature_ahead(self, feature_type: str, max_distance: float, max_distance_in_parallel: float) -> Tuple[bool, float, Dict, str]:
        if self.current_position is None or self.current_bearing is None:
            return False, float('inf'), {}, ""

        current_lat, current_lon = self.current_position
        closest_feature_distance = float('inf')
        closest_feature_info = {}
        closest_feature_id = ""

        # Define a bounding box for the spatial query
        search_distance = max_distance / 111000  # Convert meters to degrees (approximate)
        bbox = (
            current_lon - search_distance,
            current_lat - search_distance,
            current_lon + search_distance,
            current_lat + search_distance
        )

        # Query the rtree index for nearby features
        nearby_features = list(self.index.intersection(bbox))

        for feature_id in nearby_features:
            # Check if this feature is of the correct type
            if feature_type == custom.TeToo.FeatureType.trafficSignal and feature_id not in self.traffic_signals:
                continue
            if feature_type == custom.TeToo.FeatureType.speedCamera and feature_id not in self.speed_cameras:
                continue

            feature_data = self.traffic_signals.get(feature_id) or self.speed_cameras.get(feature_id)
            if not feature_data:
                continue

            feature_lat, feature_lon = feature_data['lat'], feature_data['lon']

            feature_point = (feature_lat, feature_lon)

            distance = haversine_distance(self.current_position, feature_point)

            # Skip features beyond the maximum look-ahead distance
            if distance > max_distance:
                continue

            projected_point = calculate_predicted_position(self.current_position, self.current_bearing, distance)
            parallel_distance_to_feature = haversine_distance(projected_point, feature_point)

            if parallel_distance_to_feature > max_distance_in_parallel:
                continue

            bearing = calculate_bearing(self.current_position, (feature_lat, feature_lon))

            # Check if the feature is ahead based on bearing
            if feature_is_ahead(self.current_bearing, bearing) and distance < closest_feature_distance:
                closest_feature_distance = distance
                closest_feature_info = feature_data
                closest_feature_id = feature_id

        return bool(closest_feature_info), closest_feature_distance, closest_feature_info, closest_feature_id

    def check_traffic_signal_ahead(self) -> Tuple[bool, float, Dict, str]:
        MAX_TRAFFIC_SIGNAL_DISTANCE = 250  # meters
        MAX_DISTANCE_IN_PARALLEL = 10
        return self._check_feature_ahead(custom.TeToo.FeatureType.trafficSignal, MAX_TRAFFIC_SIGNAL_DISTANCE, MAX_DISTANCE_IN_PARALLEL)

    def check_speed_camera_ahead(self) -> Tuple[bool, float, Dict, str]:
        MAX_SPEED_CAMERA_DISTANCE = 1000  # meters
        MAX_DISTANCE_IN_PARALLEL = 30
        return self._check_feature_ahead(custom.TeToo.FeatureType.speedCamera, MAX_SPEED_CAMERA_DISTANCE, MAX_DISTANCE_IN_PARALLEL)

    def _get_feature(self, type, lat, lon, func, display_tags=False):
        ahead, distance, info, id = func()
        if not ahead:
            return {}
        feature = {
            'id': str(id),
            'type': type,
            'lat': float(info['lat']),
            'lon': float(info['lon']),
            # keep it for future
            # 'bearing': float(calculate_bearing((lat, lon), (info['lat'], info['lon'])))
            'distance': float(distance),
        }
        if display_tags:
            feature['tags'] = json.dumps(info['tags'], ensure_ascii=False)
        return feature

    def tetoo_thread(self):
        sm = messaging.SubMaster(['liveLocationKalman', 'carState'])
        pm = messaging.PubMaster(['teToo'])
        te_too_dat_prev = {}
        update_counter = 0
        while True:
            sm.update()
            self.v_ego = sm['carState'].vEgo
            location = sm['liveLocationKalman']
            localizer_valid = (location.status == log.LiveLocationKalman.Status.valid) and location.positionGeodetic.valid

            # Update GPS and bearing history at 20 Hz
            if localizer_valid:
                lat = location.positionGeodetic.value[0]
                lon = location.positionGeodetic.value[1]
                bearing = math.degrees(location.calibratedOrientationNED.value[2])

                # Update GPS history
                self.gps_history.append((lat, lon))

                # Update bearing history
                self.bearing_history.append(bearing)

                # Calculate moving average of bearing
                self.current_bearing = sum(self.bearing_history) / len(self.bearing_history)

            # Main logic at 2 Hz
            if update_counter % 10 == 0:
                use_prev = False
                if self.v_ego < 1.3:
                    use_prev = True
                if not localizer_valid:
                    use_prev = True

                dat = messaging.new_message('teToo', valid=True)
                if use_prev:
                    dat.teToo = te_too_dat_prev
                else:
                    dat.teToo.lat = float(lat)
                    dat.teToo.lon = float(lon)
                    dat.teToo.bearing = float(self.current_bearing) if self.current_bearing is not None else 0.

                    road_info = self.update_position(lat, lon, bearing)
                    dat.teToo.name = str("" if road_info['name'] is None else road_info['name'])
                    dat.teToo.maxspeed = float(road_info['maxspeed'])

                    features = []
                    traffic_signal_ahead = self._get_feature(custom.TeToo.FeatureType.trafficSignal, lat, lon, self.check_traffic_signal_ahead)
                    if traffic_signal_ahead:
                        features.append(traffic_signal_ahead)
                    speed_camera_ahead = self._get_feature(custom.TeToo.FeatureType.speedCamera, lat, lon, self.check_speed_camera_ahead, True)
                    if speed_camera_ahead:
                        features.append(speed_camera_ahead)
                    dat.teToo.nearestFeatures = features

                dat.teToo.updatingData = self.fetching_thread is not None and self.fetching_thread.is_alive()
                pm.send('teToo', dat)
                te_too_dat_prev = dat.teToo

            update_counter += 1
            time.sleep(1/20)  # Sleep for 50ms (20 Hz)

def main():
    tetoo = TeToo()
    tetoo.tetoo_thread()

if __name__ == "__main__":
    main()
