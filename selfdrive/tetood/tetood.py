#!/usr/bin/env python3

from typing import List, Tuple
import time
import rtree
import math
from openpilot.dp_ext.selfdrive.tetood.hmm_map_matcher import HMMMapMatcher
from openpilot.dp_ext.selfdrive.tetood.overpass_api_helper import OverpassAPIHelper

import cereal.messaging as messaging
from cereal import log


class TeToo:
    def __init__(self):
        self.current_position = None
        self.current_bearing = None
        self.prefetched_data = None
        self.road_network = None
        self.index = rtree.index.Index()
        self.last_fetch_position = None
        self.gps_history = []
        self.map_matcher = None
        self.overpass_helper = None

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
        if not self.last_fetch_position or self._distance(self.current_position, self.last_fetch_position) > 1000:
            self._fetch_data()

    def _fetch_data(self):
        self.overpass_helper = OverpassAPIHelper()
        print("fetching")
        self.prefetched_data = self.overpass_helper.fetch_data(self.current_position[0], self.current_position[1])
        print("building network")
        self._build_road_network()
        self.last_fetch_position = self.current_position

        # Implement Overpass API query here
        # After fetching and processing data:
        self.map_matcher = HMMMapMatcher(self.road_network)

    def _build_road_network(self):
        self.road_network = {}
        self.index = rtree.index.Index()
        for element in self.prefetched_data['elements']:
            if element['type'] == 'way':
                way_id = element['id']
                self.road_network[way_id] = {
                    'nodes': element['nodes'],
                    'tags': element.get('tags', {})
                }
            elif element['type'] == 'node':
                node_id = element['id']
                lat, lon = element['lat'], element['lon']
                self.index.insert(node_id, (lon, lat, lon, lat))
                self.road_network[node_id] = {'lat': lat, 'lon': lon}

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
        if not hasattr(self, 'current_way'):
            return None, None

        way_data = self.road_network[self.current_way]
        road_name = way_data['tags'].get('name', 'Unknown')
        speed_limit = way_data['tags'].get('maxspeed', 'Unknown')
        return road_name, speed_limit

    def tetoo_thread(self):
        sm = messaging.SubMaster(['liveLocationKalman', 'carState'])
        road_info = {"name": None, "speed_limit": None}
        while True:
            sm.update()
            location = sm['liveLocationKalman']
            localizer_valid = (location.status == log.LiveLocationKalman.Status.valid) and location.positionGeodetic.valid

            if False: #road_info is not None and sm['carState'].vEgo < 1.3:
                pass
            elif localizer_valid:
                lat = location.positionGeodetic.value[0]
                lon = location.positionGeodetic.value[1]
                bearing = location.positionGeodetic.value[2]

                road_name, speed_limit = self.update_position(lat, lon, bearing)
                road_info = {"name": road_name, "speed_limit": speed_limit}
            print(f"Current road: {road_info['name']}, Speed limit: {road_info['speed_limit']}")
            time.sleep(0.5)

    @staticmethod
    def _distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        # Haversine formula for distance calculation
        R = 6371000  # Earth radius in meters
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

def main():
    tetoo = TeToo()
    tetoo.tetoo_thread()

if __name__ == "__main__":
    main()
