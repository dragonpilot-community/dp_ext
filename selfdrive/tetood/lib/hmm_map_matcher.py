import numpy as np
from typing import List, Tuple, Dict
from rtree import index

class HMMMapMatcher:
    def __init__(self, road_network: Dict):
        self.road_network = road_network
        self.spatial_index = self._build_spatial_index()

    def _build_spatial_index(self):
        """Creates an R-tree spatial index of the road network, this allows for fast querying of nearby roads."""
        idx = index.Index()
        for way_id, way_data in self.road_network.items():
            if isinstance(way_id, int) and 'nodes' in way_data:
                for node in way_data['nodes']:
                    node_data = self.road_network[node]
                    idx.insert(way_id, (node_data['lon'], node_data['lat'], node_data['lon'], node_data['lat']))
        return idx

    def match(self, gps_history: List[Tuple[float, float]]) -> int:
        """Takes a list of GPS points and returns the ID of the most likely road, implements the Viterbi algorithm to find the most probable sequence of roads."""
        candidate_roads = self._get_candidate_roads(gps_history)
        num_states = len(candidate_roads)
        num_observations = len(gps_history)

        emission_probs = self._calculate_emission_probabilities(gps_history, candidate_roads)
        transition_probs = self._calculate_transition_probabilities(candidate_roads)
        initial_probs = np.ones(num_states) / num_states

        viterbi = np.zeros((num_states, num_observations))
        backpointers = np.zeros((num_states, num_observations), dtype=int)

        viterbi[:, 0] = initial_probs * emission_probs[:, 0]

        for t in range(1, num_observations):
            for s in range(num_states):
                viterbi[s, t] = np.max(viterbi[:, t-1] * transition_probs[:, s]) * emission_probs[s, t]
                backpointers[s, t] = np.argmax(viterbi[:, t-1] * transition_probs[:, s])

        best_path_pointer = np.argmax(viterbi[:, -1])
        best_path = [best_path_pointer]
        for t in range(num_observations - 1, 0, -1):
            best_path_pointer = backpointers[best_path_pointer, t]
            best_path.append(best_path_pointer)
        best_path.reverse()

        return candidate_roads[best_path[-1]]

    def _get_candidate_roads(self, gps_points: List[Tuple[float, float]]) -> List[int]:
        """Finds potential road candidates for each GPS point using the spatial index, returns a list of unique road IDs that are near any of the GPS points."""
        candidate_roads = set()
        for lat, lon in gps_points:
            nearby_roads = list(self.spatial_index.nearest((lon, lat, lon, lat), 5))
            candidate_roads.update(nearby_roads)
        return list(candidate_roads)

    def _calculate_emission_probabilities(self, gps_points, candidate_roads):
        emission_probs = np.zeros((len(candidate_roads), len(gps_points)))
        for i, road in enumerate(candidate_roads):
            for j, (lat, lon) in enumerate(gps_points):
                distance = self._perpendicular_distance(lat, lon, self.road_network[road]['nodes'])
                emission_probs[i, j] = np.exp(-distance / 20)  # 20 meters as standard deviation
        return emission_probs / emission_probs.sum(axis=0)

    def _calculate_transition_probabilities(self, candidate_roads):
        """Computes the probability of transitioning from one road to another, currently uses a simple model with high probability of staying on the same road."""
        num_roads = len(candidate_roads)
        transition_probs = np.zeros((num_roads, num_roads))
        for i in range(num_roads):
            for j in range(num_roads):
                if i == j:
                    transition_probs[i, j] = 0.9  # High probability of staying on the same road
                else:
                    transition_probs[i, j] = 0.1 / (num_roads - 1)  # Equal probability for other roads
        return transition_probs

    def _perpendicular_distance(self, lat, lon, road_nodes):
        """Calculates the minimum perpendicular distance from a point to a road, iterates through road segments to find the closest one."""
        min_distance = float('inf')
        for i in range(len(road_nodes) - 1):
            node1 = self.road_network[road_nodes[i]]
            node2 = self.road_network[road_nodes[i+1]]
            distance = self._point_to_line_distance(lat, lon, node1['lat'], node1['lon'], node2['lat'], node2['lon'])
            min_distance = min(min_distance, distance)
        return min_distance

    @staticmethod
    def _point_to_line_distance(px, py, x1, y1, x2, y2):
        """Calculates the perpendicular distance from a point to a line segment,"""
        numerator = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1)
        denominator = ((y2-y1)**2 + (x2-x1)**2)**0.5
        return numerator / denominator if denominator != 0 else float('inf')
