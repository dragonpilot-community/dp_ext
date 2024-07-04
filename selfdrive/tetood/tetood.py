#!/usr/bin/env python3

from typing import List, Tuple
import time
import rtree
import math
import threading

from openpilot.dp_ext.selfdrive.tetood.lib.hmm_map_matcher import HMMMapMatcher
from openpilot.dp_ext.selfdrive.tetood.lib.overpass_api_helper import OverpassAPIHelper
from openpilot.dp_ext.selfdrive.tetood.lib.utils import haversine_distance

import cereal.messaging as messaging
from cereal import log

RADIUS = 3500

class TeToo:
    def __init__(self):
        self.current_v_ego = 0.
        self.current_position = None
        self.current_bearing = None
        self.prefetched_data = None
        self.road_network = None
        self.index = rtree.index.Index()
        self.last_fetch_position = None
        self.gps_history = []
        self.map_matcher = None
        self.current_way = None

        self.fetching_thread = None

        self.data_lock = threading.Lock()
        self.current_road_network = None

    def update_position(self, lat: float, lon: float, bearing: float):
        self.current_position = (lat, lon)
        self.current_bearing = bearing
        self.gps_history.append((lat, lon))
        if len(self.gps_history) > 5:  # Keep only last 5 points
            self.gps_history.pop(0)
        self._check_and_fetch_data()
        self._map_match()
        return self._get_road_info()

    def _check_and_fetch_data(self):
        if not self.last_fetch_position or \
                haversine_distance(self.current_position, self.last_fetch_position) > RADIUS - self._get_boundary_offset():
            self._fetch_data()

    def _get_boundary_offset(self):
        # 16.67 m/s = 60 km/h
        return 300 if self.current_v_ego <= 16.67 else 600

    def _fetch_data(self):
        def fetch():
            overpass_helper = OverpassAPIHelper()
            print("fetching")
            self.prefetched_data = overpass_helper.fetch_data(self.current_position[0], self.current_position[1], RADIUS)
            print("building network")
            self._build_road_network()
            self.last_fetch_position = self.current_position

            with self.data_lock:
                self.map_matcher = HMMMapMatcher(self.current_road_network)

        if self.fetching_thread is not None and self.fetching_thread.is_alive():
            return
        self.fetching_thread = threading.Thread(target=fetch)
        self.fetching_thread.start()


    def _build_road_network(self):
        new_road_network = {}
        new_index = rtree.index.Index()
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

        self.current_way = self.map_matcher.match(self.gps_history, candidate_roads)

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
            return None, None

        with self.data_lock:
            try:
                way_data = self.current_road_network[self.current_way]
                road_name = way_data['tags'].get('name', 'Unknown')
                speed_limit = way_data['tags'].get('maxspeed', 'Unknown')
                return road_name, speed_limit
            except KeyError:
                return 'Unknown', 'Unknown'

    def tetoo_thread(self):
        sm = messaging.SubMaster(['liveLocationKalman', 'carState'])
        road_info = {"name": None, "speed_limit": None}
        while True:
            sm.update()
            self.current_v_ego = sm['carState'].vEgo

            location = sm['liveLocationKalman']
            localizer_valid = (location.status == log.LiveLocationKalman.Status.valid) and location.positionGeodetic.valid

            if self.current_v_ego < 1.3: #road_info is not None and sm['carState'].vEgo < 1.3:
                pass
            elif localizer_valid:
                lat = location.positionGeodetic.value[0]
                lon = location.positionGeodetic.value[1]
                bearing = math.degrees(location.positionGeodetic.value[2])

                road_name, speed_limit = self.update_position(lat, lon, bearing)
                road_info = {"name": road_name, "speed_limit": speed_limit}
            print(f"Current road: {road_info['name']}, Speed limit: {road_info['speed_limit']}")
            time.sleep(0.5)

def main():
    tetoo = TeToo()
    tetoo.tetoo_thread()

if __name__ == "__main__":
    main()
