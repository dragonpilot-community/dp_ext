
import requests
import numpy as np

class OverpassAPIHelper:
    def __init__(self):
        self.url = 'https://overpass-api.de/api/interpreter'
        # self.url = 'https://overpass.kumi.systems/api/interpreter'
        self.headers = {'Accept-Encoding': 'gzip'}

    def fetch_data(self, lat, lon, radius=3500, high_speed=True):
        R = 6373000.0  # approximate radius of earth in mts
        """Fetch data from Overpass API based on the given GPS coordinates."""
        bbox_angle = np.degrees(radius / R)
        # fetch all ways and nodes on this ways in bbox
        bbox_str = f'{str(lat - bbox_angle)},{str(lon - bbox_angle)},{str(lat + bbox_angle)},{str(lon + bbox_angle)}'
        excl_way_types = 'pedestrian|footway|path|corridor|bridleway|steps|cycleway|construction|bus_guideway|escape|track'
        overpass_query = f"""
        [out:json][timeout:25][bbox:{bbox_str}];
        (
          way[highway][highway!~"^({excl_way_types})$"];
        ) -> .allways;
        (
          way[highway][highway!~"^({excl_way_types})$"][!name];
        ) -> .no_name_ways;
        (
          way[highway][highway!~"^({excl_way_types})$"][service][service~"^(alley|driveway)$"];
        ) -> .service_ways;
        (.allways; - .no_name_ways;) -> .way_result_1;
        (.way_result_1; - .service_ways;) -> .way_result_final;
        (.way_result_final;>;);
        out body;
        """
        # print(overpass_query)
        response = requests.get(self.url, params={'data': overpass_query}, headers=self.headers)
        return response.json()
