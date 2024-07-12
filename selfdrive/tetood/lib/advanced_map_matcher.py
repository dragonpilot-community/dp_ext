import math
from typing import List, Dict, Tuple, Optional
from rtree import index
import numpy as np
from functools import lru_cache, cached_property

PADDING = 0.0001
LANE_WIDTH = 3.7  # meters
MIN_WAY_DIST = 500  # meters
TO_RADIANS = math.pi / 180.0

class Position:
    def __init__(self, latitude: float, longitude: float, bearing: float, speed: float):
        self.latitude = latitude
        self.longitude = longitude
        self.bearing = bearing
        self.speed = speed

class Coordinates:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

class Way:
    def __init__(self, id: int, nodes: List[Coordinates], name: str, ref: str, oneway: bool, lanes: int):
        self.id = id
        self.nodes = nodes
        self.name = name
        self.ref = ref
        self.oneway = oneway
        self.lanes = max(lanes, 2)

    @cached_property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        return (
            min(node.latitude for node in self.nodes),
            min(node.longitude for node in self.nodes),
            max(node.latitude for node in self.nodes),
            max(node.longitude for node in self.nodes)
        )

class OnWayResult:
    def __init__(self, on_way: bool, distance: float, is_forward: bool):
        self.on_way = on_way
        self.distance = distance
        self.is_forward = is_forward

class CurrentWay:
    def __init__(self, way: Way, distance: float, on_way: OnWayResult, start_position: Coordinates, end_position: Coordinates):
        self.way = way
        self.distance = distance
        self.on_way = on_way
        self.start_position = start_position
        self.end_position = end_position

class NextWayResult:
    def __init__(self, way: Way, is_forward: bool, start_position: Coordinates, end_position: Coordinates):
        self.way = way
        self.is_forward = is_forward
        self.start_position = start_position
        self.end_position = end_position

class AdvancedMapMatcher:
    def __init__(self, road_data: Dict):
        self.ways = self._load_ways(road_data)
        self.rtree = index.Index()
        self.gps_buffer: List[Position] = []
        self.current_way: Optional[CurrentWay] = None
        self._build_rtree()
        self.way_vectors = {}  # Add this line
        self._precompute_way_vectors()  # Add this line

        # HMM variables
        self.hmm_states: List[Tuple[int, int]] = []
        self.hmm_transitions: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.current_timestep = 0
        self.current_viterbi_path: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.current_viterbi_prob: Dict[Tuple[int, int], float] = {}

        self.initialize_hmm()

    def _load_ways(self, road_data: Dict) -> List[Way]:
        """
        Load ways from the provided road data.

        Args:
            road_data (Dict): Dictionary containing road network data.

        Returns:
            List[Way]: List of Way objects created from the road data.
        """
        ways = []
        for element in road_data['elements']:
            if element['type'] == 'way':
                nodes = [Coordinates(node['lat'], node['lon']) for node in element['nodes']]
                tags = element.get('tags', {})
                way = Way(
                    id=element['id'],
                    nodes=nodes,
                    name=tags.get('name', ''),
                    ref=tags.get('ref', ''),
                    oneway=tags.get('oneway', 'no').lower() == 'yes',
                    lanes=int(tags.get('lanes', '2'))
                )
                ways.append(way)
        return ways

    def _build_rtree(self):
        data = [(idx, way.bounding_box(), None) for idx, way in enumerate(self.ways)]
        self.rtree = index.Index(data)

    def initialize_hmm(self):
        """Initialize the Hidden Markov Model for map matching."""
        self.build_hmm()
        self.current_timestep = 0
        self.current_viterbi_path = {}
        self.current_viterbi_prob = {}

        initial_prob = 1.0 / len(self.hmm_states)
        for state in self.hmm_states:
            self.current_viterbi_path[state] = [state]
            self.current_viterbi_prob[state] = initial_prob

    def build_hmm(self):
        """Build the Hidden Markov Model states and transitions."""
        self.hmm_states = [(way.id, i) for way in self.ways for i in range(len(way.nodes) - 1)]
        self.hmm_transitions = {
            (way.id, i): [(way.id, i + 1)] for way in self.ways for i in range(len(way.nodes) - 2)
        }
        for way in self.ways:
            self.hmm_transitions[(way.id, len(way.nodes) - 2)] = []

    @lru_cache(maxsize=1000)
    def on_way(self, way: Way, pos: Position) -> OnWayResult:
        """
        Determine if a position is on a given way.

        Args:
            way (Way): The way to check.
            pos (Position): The position to check.

        Returns:
            OnWayResult: An object containing whether the position is on the way,
                         the distance to the way, and if it's in the forward direction.
        """
        if self._is_within_bounding_box(way, pos):
            distance = self.distance_to_way(pos, way)
            lanes = max(way.lanes, 2)
            road_width_estimate = float(lanes) * LANE_WIDTH
            max_dist = 5 + road_width_estimate

            if distance < max_dist:
                is_forward = self.is_forward(way.nodes[0], way.nodes[-1], pos.bearing)
                if not is_forward and way.oneway:
                    return OnWayResult(False, distance, is_forward)
                return OnWayResult(True, distance, is_forward)

        return OnWayResult(False, float('inf'), False)

    def _find_nearest_ways(self, pos: Position, k: int = 5) -> List[Way]:
        nearest = list(self.rtree.nearest((pos.latitude, pos.longitude, pos.latitude, pos.longitude), k))
        return [self.ways[i] for i in nearest]

    def next_ways(self, pos: Position, current_way: CurrentWay, is_forward: bool) -> List[NextWayResult]:
        next_ways = []
        dist = 0.0
        way = current_way.way
        forward = is_forward
        start_pos = pos

        while dist < MIN_WAY_DIST:
            d = self._distance_to_end_of_way(start_pos, way, forward)
            if d <= 0:
                break
            dist += d

            # Use _find_nearest_ways instead of checking all ways
            nearest_ways = self._find_nearest_ways(start_pos, k=5)
            nw = self._find_best_next_way(way, nearest_ways, forward)

            if nw is None:
                break

            next_ways.append(nw)
            way = nw.way
            start_pos = Position(nw.start_position.latitude, nw.start_position.longitude, 0, 0)
            forward = nw.is_forward

        if not next_ways:
            nearest_ways = self._find_nearest_ways(pos, k=5)
            nw = self._find_best_next_way(current_way.way, nearest_ways, is_forward)
            if nw:
                next_ways.append(nw)

        return next_ways

    def _find_best_next_way(self, current_way: Way, candidate_ways: List[Way], is_forward: bool) -> Optional[NextWayResult]:
        """
        Find the best next way from a list of candidate ways.

        Args:
            current_way (Way): The current way.
            candidate_ways (List[Way]): List of candidate ways to choose from.
            is_forward (bool): Whether to look for the next way in the forward direction.

        Returns:
            Optional[NextWayResult]: The best next way result, or None if no suitable way is found.
        """
        match_node = current_way.nodes[-1] if is_forward else current_way.nodes[0]
        match_bearing_node = current_way.nodes[-2] if is_forward else current_way.nodes[1]

        # Filter candidate ways to those that connect to the current way
        connecting_ways = [way for way in candidate_ways if match_node in (way.nodes[0], way.nodes[-1])]

        if not connecting_ways:
            return None

        # First, check for ways with matching names
        name_matches = [way for way in connecting_ways if way.name == current_way.name]
        if name_matches:
            return self._select_best_way(name_matches, match_node, match_bearing_node)

        # Then, check for ways with matching references
        ref_matches = [way for way in connecting_ways if way.ref == current_way.ref]
        if ref_matches:
            return self._select_best_way(ref_matches, match_node, match_bearing_node)

        # Finally, select the way with the least curvature
        return self._select_best_way(connecting_ways, match_node, match_bearing_node)

    def _select_best_way(self, ways: List[Way], match_node: Coordinates, match_bearing_node: Coordinates) -> NextWayResult:
        """
        Select the best way from a list of ways based on curvature and direction.

        Args:
            ways (List[Way]): List of candidate ways.
            match_node (Coordinates): The node where the ways should connect.
            match_bearing_node (Coordinates): The node used to calculate the bearing of the current way.

        Returns:
            NextWayResult: The best next way result.
        """
        best_way = None
        best_curvature = float('inf')
        best_is_forward = False

        for way in ways:
            is_forward = way.nodes[0] == match_node
            if not is_forward and way.oneway:
                continue

            bearing_node = way.nodes[1] if is_forward else way.nodes[-2]
            curvature = abs(self._calculate_angle(match_bearing_node, match_node, bearing_node))

            if curvature < best_curvature:
                best_way = way
                best_curvature = curvature
                best_is_forward = is_forward

        if best_way is None:
            return None

        start, end = self._get_way_start_end(best_way, best_is_forward)
        return NextWayResult(best_way, best_is_forward, start, end)

    @lru_cache(maxsize=1000)
    def _find_matching_ways(self, way: Way, match_node: Coordinates) -> List[Way]:
        """Find ways that connect to the given way at the specified node."""
        return [w for w in self.ways if w != way and (w.nodes[0] == match_node or w.nodes[-1] == match_node)]

    def _next_is_forward(self, next_way: Way, match_node: Coordinates) -> bool:
        """Determine if the next way is in the forward direction."""
        return next_way.nodes[0] == match_node

    def _is_large_angle(self, match_bearing_node: Coordinates, match_node: Coordinates, next_way: Way, next_is_forward: bool) -> bool:
        """Check if the angle between the current way and the next way is large."""
        bearing_node = next_way.nodes[1] if next_is_forward else next_way.nodes[-2]
        curv = self._get_curvature(match_bearing_node, match_node, next_way, next_is_forward)
        return abs(curv) > 0.1

    def _get_curvature(self, match_bearing_node: Coordinates, match_node: Coordinates, next_way: Way, next_is_forward: bool) -> float:
        """Calculate the curvature between the current way and the next way."""
        bearing_node = next_way.nodes[1] if next_is_forward else next_way.nodes[-2]
        return self._calculate_angle(match_bearing_node, match_node, bearing_node)

    def _create_next_way_result(self, next_way: Way, next_is_forward: bool) -> NextWayResult:
        """Create a NextWayResult object for the given next way."""
        start, end = self._get_way_start_end(next_way, next_is_forward)
        return NextWayResult(next_way, next_is_forward, start, end)

    def _distance_to_end_of_way(self, pos: Position, way: Way, is_forward: bool) -> float:
        """Calculate the distance from the given position to the end of the way."""
        nodes = way.nodes if is_forward else reversed(way.nodes)
        total_distance = 0.0
        start_node = nodes[0]
        for end_node in nodes[1:]:
            segment_distance = self.haversine_distance(start_node, end_node)
            total_distance += segment_distance
            if self.point_to_line_distance(pos.latitude, pos.longitude, start_node.latitude, start_node.longitude, end_node.latitude, end_node.longitude) <= 0.1:
                return total_distance
            start_node = end_node
        return total_distance

    @staticmethod
    def _calculate_angle(p1: Coordinates, p2: Coordinates, p3: Coordinates) -> float:
        """Calculate the angle between three points."""
        v1 = (p1.latitude - p2.latitude, p1.longitude - p2.longitude)
        v2 = (p3.latitude - p2.latitude, p3.longitude - p2.longitude)
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        cos_angle = dot_product / (mag_v1 * mag_v2)
        return math.acos(max(-1, min(1, cos_angle)))  # Clamp to [-1, 1] to avoid domain errors

    @staticmethod
    def _get_way_start_end(way: Way, is_forward: bool) -> Tuple[Coordinates, Coordinates]:
        """Get the start and end coordinates of a way based on the direction."""
        return (way.nodes[0], way.nodes[-1]) if is_forward else (way.nodes[-1], way.nodes[0])

    def update_gps(self, pos: Position) -> Optional[CurrentWay]:
        self.gps_buffer.append(pos)
        if len(self.gps_buffer) > 25:
            self.gps_buffer.pop(0)

        # Check if we're still on the current way
        if self.current_way and self.on_way(self.current_way.way, pos).on_way:
            # Simple update if still on the same way
            return self._update_current_way(pos)

        # If not, find the nearest ways and check them
        nearest_ways = self._find_nearest_ways(pos)
        for way in nearest_ways:
            on_way_result = self.on_way(way, pos)
            if on_way_result.on_way:
                return self._update_current_way(pos, way, on_way_result)

        # Only use HMM if we can't find a clear match
        if self.current_timestep % 5 == 0:
            self._viterbi_step(pos)

        self.current_timestep += 1

        if not self.current_viterbi_prob:
            return None

        most_likely_state = max(self.current_viterbi_prob, key=self.current_viterbi_prob.get)
        way = next((way for way in self.ways if way.id == most_likely_state[0]), None)

        if way is None:
            return None

        on_way_result = self.on_way(way, pos)
        start, end = self._get_way_start_end(way, on_way_result.is_forward)

        current_way = CurrentWay(way, on_way_result.distance, on_way_result, start, end)

        # Use next_ways to get upcoming ways
        upcoming_ways = self.next_ways(pos, current_way, on_way_result.is_forward)

        # You could store these upcoming ways for future use or return them along with the current way
        self.upcoming_ways = upcoming_ways

        return current_way

    def _update_current_way(self, pos: Position, way: Way, on_way_result: OnWayResult) -> CurrentWay:
        start, end = self._get_way_start_end(way, on_way_result.is_forward)
        current_way = CurrentWay(way, on_way_result.distance, on_way_result, start, end)

        # Update upcoming ways
        self.upcoming_ways = self.next_ways(pos, current_way, on_way_result.is_forward)

        return current_way

    def _viterbi_step(self, pos: Position):
        new_viterbi_prob = {}
        new_viterbi_path = {}

        for state in self.hmm_states:
            way = next((w for w in self.ways if w.id == state[0]), None)
            if way is None:
                continue

            emission_prob = self._emission_probability(pos, way, state[1])
            max_prob = float('-inf')
            max_prev_state = None

            for prev_state in self.hmm_transitions.get(state, []):
                if prev_state in self.current_viterbi_prob:
                    transition_prob = self._transition_probability(prev_state, state)

                    # Consider upcoming ways in transition probability
                    if hasattr(self, 'upcoming_ways'):
                        upcoming_way_ids = [w.way.id for w in self.upcoming_ways]
                        if state[0] in upcoming_way_ids:
                            transition_prob *= 1.2  # Increase probability for upcoming ways

                    prob = self.current_viterbi_prob[prev_state] + math.log(transition_prob) + math.log(emission_prob)
                    if prob > max_prob:
                        max_prob = prob
                        max_prev_state = prev_state

            if max_prev_state is not None:
                new_viterbi_prob[state] = max_prob
                new_viterbi_path[state] = self.current_viterbi_path[max_prev_state] + [state]

        self.current_viterbi_prob = new_viterbi_prob
        self.current_viterbi_path = new_viterbi_path

    @lru_cache(maxsize=1000)
    def _emission_probability(self, pos: Position, way: Way, segment_index: int) -> float:
        """
        Calculate the emission probability for a given position and way segment.

        Args:
            pos (Position): The current GPS position.
            way (Way): The candidate way.
            segment_index (int): The index of the segment in the way.

        Returns:
            float: The emission probability.
        """
        start_node = way.nodes[segment_index]
        end_node = way.nodes[segment_index + 1]
        distance = self.point_to_line_distance(pos.latitude, pos.longitude,
                                               start_node.latitude, start_node.longitude,
                                               end_node.latitude, end_node.longitude)
        return math.exp(-distance / 20.0)  # Assuming GPS error std dev of 20 meters

    @lru_cache(maxsize=1000)
    def _transition_probability(self, prev_state: Tuple[int, int], curr_state: Tuple[int, int]) -> float:
        """
        Calculate the transition probability between two states.

        Args:
            prev_state (Tuple[int, int]): The previous state (way_id, segment_index).
            curr_state (Tuple[int, int]): The current state (way_id, segment_index).

        Returns:
            float: The transition probability.
        """
        if prev_state[0] == curr_state[0]:  # Same way
            if curr_state[1] == prev_state[1] + 1:  # Next segment
                return 0.9
            else:
                return 0.1 / (len(self.ways[prev_state[0]].nodes) - 2)  # Uniform distribution over other segments
        else:  # Different way
            return 0.1 / (len(self.ways) - 1)  # Uniform distribution over other ways

    @staticmethod
    def _is_within_bounding_box(way: Way, pos: Position) -> bool:
        """Check if a position is within the bounding box of a way."""
        bbox = way.bounding_box()
        return (bbox[0] - PADDING <= pos.latitude <= bbox[2] + PADDING and
                bbox[1] - PADDING <= pos.longitude <= bbox[3] + PADDING)

    def _precompute_way_vectors(self):
        self.way_vectors = {}
        for way in self.ways:
            coords = np.array([(node.latitude, node.longitude) for node in way.nodes])
            self.way_vectors[way.id] = coords

    def distance_to_way(self, pos: Position, way: Way) -> float:
        way_vector = self.way_vectors[way.id]
        pos_vector = np.array([pos.latitude, pos.longitude])
        distances = np.sum((way_vector[:-1] - pos_vector)**2, axis=1)
        return np.min(distances)**0.5

    @staticmethod
    def is_forward(start: Coordinates, end: Coordinates, bearing: float) -> bool:
        """Determine if the bearing is in the forward direction of the way."""
        way_bearing = AdvancedMapMatcher.calculate_bearing(start, end)
        return abs((way_bearing - bearing + 180) % 360 - 180) < 90

    @staticmethod
    def calculate_bearing(start: Coordinates, end: Coordinates) -> float:
        """Calculate the bearing between two coordinates."""
        y = math.sin(end.longitude - start.longitude) * math.cos(end.latitude)
        x = math.cos(start.latitude) * math.sin(end.latitude) - math.sin(start.latitude) * math.cos(end.latitude) * math.cos(end.longitude - start.longitude)
        return (math.atan2(y, x) * 180 / math.pi + 360) % 360

    @staticmethod
    def haversine_distance(start: Coordinates, end: Coordinates) -> float:
        """Calculate the Haversine distance between two coordinates."""
        R = 6371000  # Earth radius in meters
        lat1, lon1 = start.latitude * TO_RADIANS, start.longitude * TO_RADIANS
        lat2, lon2 = end.latitude * TO_RADIANS, end.longitude * TO_RADIANS
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    @staticmethod
    def point_to_line_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate the perpendicular distance from a point to a line segment."""
        dx, dy = x2 - x1, y2 - y1
        if dx == dy == 0:  # The segment is actually a point
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)
        if t < 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        elif t > 1:
            return math.sqrt((px - x2)**2 + (py - y2)**2)
        else:
            return abs((dy*px - dx*py + x2*y1 - y2*x1) / math.sqrt(dx**2 + dy**2))