from typing import Tuple
import math

R = 6371000  # Earth's radius in meters

def haversine_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
    lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def calculate_bearing(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate bearing of two points"""
    lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
    lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])

    dlon = lon2 - lon1

    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    initial_bearing = math.atan2(y, x)

    # Convert to degrees
    initial_bearing = math.degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360

    return bearing


def feature_is_ahead(current_bearing, bearing_to_feature, within_deg=30.):
    # Calculate the absolute angle difference
    angle_diff = abs((bearing_to_feature - current_bearing + 180) % 360 - 180)

    # Consider features within a 60-degree cone in front of the vehicle
    return angle_diff <= within_deg
