import math
from typing import List, Dict, Tuple, Optional
from rtree import index
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TO_RADIANS = math.pi / 180.0
EARTH_RADIUS = 6371000  # Earth radius in meters
PADDING = 100  # meters
LANE_WIDTH = 3.7  # meters
MIN_WAY_DIST = 500  # meters
MAX_DISTANCE = 50  # meters, maximum distance to consider a way as matching
SEARCH_RADIUS = 100  # meters

class Coordinates:
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon

    def __eq__(self, other):
        return isinstance(other, Coordinates) and self.lat == other.lat and self.lon == other.lon

    def __hash__(self):
        return hash((self.lat, self.lon))

class Position:
    def __init__(self, lat: float, lon: float, bearing: float, speed: float):
        self.lat = lat
        self.lon = lon
        self.bearing = bearing
        self.speed = speed

class Way:
    def __init__(self, id: int, nodes: List[Coordinates], tags: Dict[str, str]):
        self.id = id
        self.nodes = nodes
        self.tags = tags
        self._bbox = self._calculate_bbox()

    def _calculate_bbox(self):
        lats = [node.lat for node in self.nodes]
        lons = [node.lon for node in self.nodes]
        return (min(lats), min(lons), max(lats), max(lons))

    @property
    def name(self):
        return self.tags.get('name', '')

    @property
    def ref(self):
        return self.tags.get('ref', '')

    @property
    def oneway(self):
        return self.tags.get('oneway', 'no').lower() == 'yes'

    @property
    def lanes(self):
        return int(self.tags.get('lanes', '2'))

class OnWayResult:
    def __init__(self, on_way: bool, distance: float, is_forward: bool):
        self.on_way = on_way
        self.distance = distance
        self.is_forward = is_forward

class CurrentWay:
    def __init__(self, way: Way, distance: float, on_way: OnWayResult, start_pos: Coordinates, end_pos: Coordinates):
        self.way = way
        self.distance = distance
        self.on_way = on_way
        self.start_position = start_pos
        self.end_position = end_pos

class NextWayResult:
    def __init__(self, way: Way, is_forward: bool, start_position: Coordinates, end_position: Coordinates):
        self.way = way
        self.is_forward = is_forward
        self.start_position = start_position
        self.end_position = end_position

class MapMatcher:
    def __init__(self, road_network: Dict):
        self.ways = self._load_ways(road_network)
        self.rtree = self._build_rtree()
        self.current_way = None

    def _load_ways(self, road_network: Dict) -> Dict[int, Way]:
        ways = {}
        nodes = {}
        for element in road_network['elements']:
            if element['type'] == 'node':
                nodes[element['id']] = Coordinates(element['lat'], element['lon'])
            elif element['type'] == 'way':
                way_nodes = [nodes[node_id] for node_id in element['nodes'] if node_id in nodes]
                if len(way_nodes) >= 2:
                    ways[element['id']] = Way(element['id'], way_nodes, element.get('tags', {}))
        logger.info(f"Loaded {len(ways)} ways")
        return ways

    def _build_rtree(self):
        idx = index.Index()
        for way_id, way in self.ways.items():
            bbox = way._bbox
            idx.insert(way_id, (bbox[1], bbox[0], bbox[3], bbox[2]))  # Swap lat and lon
        logger.info("R-tree index built")
        return idx

    def update_position(self, pos: Position) -> Optional[CurrentWay]:
        logger.debug(f"Updating position: lat={pos.lat}, lon={pos.lon}, bearing={pos.bearing}, speed={pos.speed}")

        lat_change = SEARCH_RADIUS / (EARTH_RADIUS * TO_RADIANS)
        lon_change = SEARCH_RADIUS / (EARTH_RADIUS * TO_RADIANS * math.cos(math.radians(pos.lat)))

        search_bbox = (pos.lon - lon_change, pos.lat - lat_change,
                       pos.lon + lon_change, pos.lat + lat_change)

        logger.debug(f"Search bbox: {search_bbox}")

        nearby_ways = list(self.rtree.intersection(search_bbox))
        logger.debug(f"Found {len(nearby_ways)} nearby ways")

        best_match = None
        best_distance = float('inf')

        for way_id in nearby_ways:
            way = self.ways.get(way_id)
            if way is None:
                continue
            on_way_result = self.on_way(way, pos)
            if on_way_result.on_way and on_way_result.distance < best_distance:
                best_match = (way, on_way_result)
                best_distance = on_way_result.distance

        if best_match:
            logger.info(f"Matched to way: {best_match[0].id}")
            return self._update_current_way(pos, best_match[0], best_match[1])

        logger.warning("No matching way found")
        return None

    def on_way(self, way: Way, pos: Position) -> OnWayResult:
        distance = self._distance_to_way(pos, way)

        lanes = way.lanes
        road_width_estimate = lanes * LANE_WIDTH
        max_dist = 5 + road_width_estimate

        if distance < max_dist:
            is_forward = self._is_forward(way.nodes[0], way.nodes[-1], pos.bearing)
            if not is_forward and way.oneway:
                return OnWayResult(False, distance, is_forward)
            return OnWayResult(True, distance, is_forward)

        return OnWayResult(False, distance, False)

    def _distance_to_way(self, pos: Position, way: Way) -> float:
        return min(self._point_to_line_distance(pos.lat, pos.lon, start.lat, start.lon, end.lat, end.lon)
                   for start, end in zip(way.nodes, way.nodes[1:]))

    @staticmethod
    @lru_cache(maxsize=1024)
    def _point_to_line_distance(px, py, x1, y1, x2, y2):
        px, py, x1, y1, x2, y2 = map(math.radians, [px, py, x1, y1, x2, y2])

        line_length = MapMatcher._haversine_distance(y1, x1, y2, x2)
        if line_length == 0:
            return MapMatcher._haversine_distance(py, px, y1, x1)

        a = MapMatcher._haversine_distance(py, px, y1, x1)
        b = MapMatcher._haversine_distance(py, px, y2, x2)
        c = line_length

        s = (a + b + c) / 2
        area = math.sqrt(abs(s * (s-a) * (s-b) * (s-c)))

        height = 2 * area / c

        return height

    @staticmethod
    @lru_cache(maxsize=1024)
    def _haversine_distance(lat1, lon1, lat2, lon2):
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return EARTH_RADIUS * c

    @staticmethod
    def _is_forward(start: Coordinates, end: Coordinates, bearing: float) -> bool:
        way_bearing = MapMatcher._calculate_bearing(start, end)
        bearing_diff = abs(way_bearing - bearing)
        return bearing_diff < 90 or bearing_diff > 270

    @staticmethod
    @lru_cache(maxsize=1024)
    def _calculate_bearing(start: Coordinates, end: Coordinates) -> float:
        y = math.sin(math.radians(end.lon - start.lon)) * math.cos(math.radians(end.lat))
        x = math.cos(math.radians(start.lat)) * math.sin(math.radians(end.lat)) - \
            math.sin(math.radians(start.lat)) * math.cos(math.radians(end.lat)) * math.cos(math.radians(end.lon - start.lon))
        bearing = math.atan2(y, x)
        return (math.degrees(bearing) + 360) % 360

    def _update_current_way(self, pos: Position, way: Way, on_way: OnWayResult) -> CurrentWay:
        start, end = self._get_way_start_end(way, on_way.is_forward)
        return CurrentWay(way, on_way.distance, on_way, start, end)

    @staticmethod
    def _get_way_start_end(way: Way, is_forward: bool) -> Tuple[Coordinates, Coordinates]:
        return (way.nodes[0], way.nodes[-1]) if is_forward else (way.nodes[-1], way.nodes[0])

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
            nw = self._next_way(way, forward)
            if nw is None:
                break
            next_ways.append(nw)
            way = nw.way
            start_pos = Position(nw.start_position.lat, nw.start_position.lon, 0, 0)
            forward = nw.is_forward

        if not next_ways:
            nw = self._next_way(current_way.way, is_forward)
            if nw:
                next_ways.append(nw)

        return next_ways

    def _distance_to_end_of_way(self, pos: Position, way: Way, is_forward: bool) -> float:
        nodes = way.nodes if is_forward else reversed(way.nodes)
        return sum(self._haversine_distance(prev.lat, prev.lon, curr.lat, curr.lon)
                   for prev, curr in zip(nodes, nodes[1:]))

    def _next_way(self, way: Way, is_forward: bool) -> Optional[NextWayResult]:
        match_node = way.nodes[-1] if is_forward else way.nodes[0]
        matching_ways = self._matching_ways(way, match_node)

        if not matching_ways:
            return None

        for mway in matching_ways:
            if mway.name == way.name or mway.ref == way.ref:
                next_is_forward = self._next_is_forward(mway, match_node)
                if not next_is_forward and mway.oneway:
                    continue
                start, end = self._get_way_start_end(mway, next_is_forward)
                return NextWayResult(mway, next_is_forward, start, end)

        min_curv_way = min(matching_ways, key=lambda w: self._get_curvature(way, w, match_node))
        next_is_forward = self._next_is_forward(min_curv_way, match_node)
        start, end = self._get_way_start_end(min_curv_way, next_is_forward)
        return NextWayResult(min_curv_way, next_is_forward, start, end)

    def _matching_ways(self, current_way: Way, match_node: Coordinates) -> List[Way]:
        return [way for way in self.ways.values()
                if way.id != current_way.id and
                (way.nodes[0] == match_node or way.nodes[-1] == match_node)]

    @staticmethod
    def _next_is_forward(next_way: Way, match_node: Coordinates) -> bool:
        return next_way.nodes[0] == match_node

    @staticmethod
    def _get_curvature(way1: Way, way2: Way, match_node: Coordinates) -> float:
        if len(way1.nodes) < 2 or len(way2.nodes) < 2:
            return float('inf')

        node1 = way1.nodes[-2] if way1.nodes[-1] == match_node else way1.nodes[-1]
        node2 = way2.nodes[1] if way2.nodes[0] == match_node else way2.nodes[-2]

        angle1 = MapMatcher._calculate_bearing(node1, match_node)
        angle2 = MapMatcher._calculate_bearing(match_node, node2)

        return abs(angle2 - angle1)

    @staticmethod
    def _bbox_intersects(bbox1, bbox2):
        return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                    bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])