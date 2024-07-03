
import requests
import numpy as np

class OverpassAPIHelper:
    def __init__(self):
        # self.url = 'https://overpass-api.de/api/interpreter'
        self.url = 'https://overpass.kumi.systems/api/interpreter'
        self.headers = {'Accept-Encoding': 'gzip'}

    def fetch_data(self, lat, lon, radius=1300):
        R = 6373000.0  # approximate radius of earth in mts
        """Fetch data from Overpass API based on the given GPS coordinates."""
        # overpass_query = f"""
        # [out:json];
        # (
        #   way(around:{radius},{lat},{lon})
        #   [highway][highway!~"^(pedestrian|footway|path|bridleway|steps|cycleway|construction|bus_guideway|escape)$"];
        # );
        # out body;
        # """
        # overpass_query = f"""
        # [out:json];
        # (
        #   way(around:{radius},{lat},{lon})
        #   [highway][highway!~"^(pedestrian|footway|path|bridleway|steps|cycleway|construction|bus_guideway|escape)$"];
        #   >;
        # );
        # out body;
        # """
        # Calculate the bounding box coordinates for the bbox containing the circle around location.
        bbox_angle = np.degrees(radius / R)
        # fetch all ways and nodes on this ways in bbox
        bbox_str = f'{str(lat - bbox_angle)},{str(lon - bbox_angle)},{str(lat + bbox_angle)},{str(lon + bbox_angle)}'
        lat_lon = "(%f,%f)" % (lat, lon)
        q = """
            [out:json];
            way(""" + bbox_str + """)
              [highway]
              [highway!~"^(footway|path|corridor|bridleway|steps|cycleway|construction|bus_guideway|escape|service|track)$"];
            (._;>;);
            out;"""
        area_q = """is_in""" + lat_lon + """;area._[admin_level~"[24]"];
            convert area ::id = id(), admin_level = t['admin_level'],
            name = t['name'], "ISO3166-1:alpha2" = t['ISO3166-1:alpha2'];out;
            """
        overpass_query = q+area_q
        print(overpass_query)
        response = requests.get(self.url, params={'data': overpass_query}, headers=self.headers)
        return response.json()
