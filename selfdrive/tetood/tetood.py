#!/usr/bin/env python3
from typing import List, Tuple, Dict
import time
import rtree
import math
import json
import threading

import cereal.messaging as messaging
from cereal import log, custom
from openpilot.common.params import Params

from openpilot.dp_ext.selfdrive.tetood.lib.hmm_map_matcher import HMMMapMatcher
from openpilot.dp_ext.selfdrive.tetood.lib.overpass_api_helper import OverpassAPIHelper
from openpilot.dp_ext.selfdrive.tetood.lib.speed_camera_loader import SpeedCameraLoader
from openpilot.dp_ext.selfdrive.tetood.lib.taiwan_speed_camera_loader import TaiwanSpeedCameraLoader
from openpilot.dp_ext.selfdrive.tetood.lib.utils import calculate_bearing, haversine_distance, feature_is_ahead

RADIUS = 3500

class TeToo:
    def __init__(self):
        self.current_position = None
        self.current_bearing = None
        self.road_network = {}
        self.traffic_signals = {}
        self.speed_cameras = {}
        self.index = rtree.index.Index()
        self.gps_history = []
        self.map_matcher = None
        self.overpass_helper = None

        self.current_way = None
        self.current_way_confidence = 0.0
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

        self.tw_speed_cameras = TaiwanSpeedCameraLoader().load_speed_cameras()

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
        if self.current_position:
            distance_moved = haversine_distance(self.current_position, new_position)
            if distance_moved > 50:  # Reset if we've moved more than 50 meters
                self.current_way = None
                self.current_way_confidence = 0

        self.current_position = new_position
        self.current_bearing = bearing
        self.gps_history.append((lat, lon))
        if len(self.gps_history) > 5:  # Keep only last 5 points
            self.gps_history.pop(0)
        self._check_and_fetch_data()

        self._map_match()
        return self._get_road_info()

    def _check_and_fetch_data(self):
        if not self.last_fetch_position or haversine_distance(self.current_position, self.last_fetch_position) > RADIUS - self._boundary_offset():
            self._fetch_data()

    def _boundary_offset(self):
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
                    self.traffic_signals[int(f"103{node_id}")] = {'lat': lat, 'lon': lon, 'tags': element.get('tags')}
                elif element.get('tags', {}).get('highway') == 'speed_camera':
                    self.speed_cameras[int(f"102{node_id}")] = {'lat': lat, 'lon': lon, 'tags': element.get('tags')}

        # Load speed cameras
        self.speed_cameras = self.speed_camera_loader.load_speed_cameras(self.prefetched_data)

        # Add speed cameras to the index
        for camera_id, camera_data in self.speed_cameras.items():
            new_index.insert(camera_id, (camera_data['lon'], camera_data['lat'], camera_data['lon'], camera_data['lat']))

        # taiwan specific camera
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

        candidate_roads = self._get_candidate_roads(self.gps_history)
        if not candidate_roads:
            return

        if self.map_matcher is None:
            return

        matched_way = self.map_matcher.match(self.gps_history, candidate_roads)

        if matched_way is not None:
            self.current_way = matched_way
            self.current_way_confidence = self._calculate_confidence(matched_way)
        else:
            self.current_way = None
            self.current_way_confidence = 0.0

    def _calculate_confidence(self, way_id):
        if self.current_position is None:
            return 0.0

        way_nodes = self.road_network[way_id]['nodes']
        closest_distance = float('inf')

        for node in way_nodes:
            node_lat = self.road_network[node]['lat']
            node_lon = self.road_network[node]['lon']
            distance = haversine_distance(self.current_position, (node_lat, node_lon))
            closest_distance = min(closest_distance, distance)

        # Calculate confidence based on distance
        # High confidence (1.0) if within 5 meters, low confidence (0.1) if 50 meters or more away
        confidence = max(0.1, min(1.0, 1.0 - (closest_distance - 5) / 45))
        return confidence

    def _get_candidate_roads(self, gps_points: List[Tuple[float, float]]) -> List[int]:
        candidate_roads = set()
        for lat, lon in gps_points:
            nearby_nodes = list(self.index.nearest((lon, lat, lon, lat), 5))
            for node in nearby_nodes:
                for way_id, way_data in self.road_network.items():
                    if isinstance(way_id, int) and 'nodes' in way_data and node in way_data['nodes']:
                        candidate_roads.add(way_id)
        return list(candidate_roads)

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

        # Only update if the new name is different and confidence is high enough
        if new_road_name != None and new_road_name != self.current_road_name and self.current_way_confidence > 0.7:
            self.current_road_name = new_road_name
            self.current_max_speed = new_max_speed
            self.current_tags = new_tags

        return {
            'id': self.current_way,
            'name': self.current_road_name,
            'maxspeed': self.current_max_speed,
            'tags': self.current_tags,
            'confidence': self.current_way_confidence
        }

    def _check_feature_ahead(self, feature_dict: Dict, max_distance: float) -> Tuple[bool, float, Dict, str]:
        if self.current_position is None or self.current_bearing is None:
            return False, float('inf'), {}, ""

        current_lat, current_lon = self.current_position
        closest_feature_distance = float('inf')
        closest_feature_info = {}
        closest_feature_id = ""

        for feature_id, feature_data in feature_dict.items():
            feature_lat, feature_lon = feature_data['lat'], feature_data['lon']

            distance = haversine_distance((current_lat, current_lon), (feature_lat, feature_lon))

            # # Skip features beyond the maximum look-ahead distance
            if distance > max_distance:
                continue

            bearing = calculate_bearing((current_lat, current_lon), (feature_lat, feature_lon))

            if feature_is_ahead(self.current_bearing, bearing) and distance < closest_feature_distance:
                closest_feature_distance = distance
                closest_feature_info = feature_data
                closest_feature_id = feature_id

        return bool(closest_feature_info), closest_feature_distance, closest_feature_info, closest_feature_id

    def check_traffic_signal_ahead(self) -> Tuple[bool, float, Dict, str]:
        MAX_TRAFFIC_SIGNAL_DISTANCE = 250  # meters
        return self._check_feature_ahead(self.traffic_signals, MAX_TRAFFIC_SIGNAL_DISTANCE)

    def check_speed_camera_ahead(self) -> Tuple[bool, float, Dict, str]:
        MAX_SPEED_CAMERA_DISTANCE = 1000  # meters
        return self._check_feature_ahead(self.speed_cameras, MAX_SPEED_CAMERA_DISTANCE)

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
        while True:
            sm.update()
            self.v_ego = sm['carState'].vEgo
            location = sm['liveLocationKalman']
            localizer_valid = (location.status == log.LiveLocationKalman.Status.valid) and location.positionGeodetic.valid

            use_prev = False
            if self.v_ego < 1.3:
                use_prev = True
            if not localizer_valid:
                use_prev = True

            dat = messaging.new_message('teToo', valid=True)
            if use_prev:
                dat.teToo = te_too_dat_prev
            else:
                lat = location.positionGeodetic.value[0]
                lon = location.positionGeodetic.value[1]
                bearing = math.degrees(location.calibratedOrientationNED.value[2])

                dat.teToo.lat = float(lat)
                dat.teToo.lon = float(lon)
                dat.teToo.bearing = float(bearing)

                road_info = self.update_position(lat, lon, bearing)
                dat.teToo.name = str(road_info['name'])
                dat.teToo.maxspeed = float(road_info['maxspeed'])
                # keep it for future
                # dat.teToo.tags = json.dumps(road_info['tags'], ensure_ascii=False)

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
            time.sleep(0.2)

def main():
    tetoo = TeToo()
    tetoo.tetoo_thread()

if __name__ == "__main__":
    main()
