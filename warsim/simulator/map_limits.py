# map_limits.py
#
# A latitude-longitude rectangle that defines a (square) region on a map
#
# Author: Giacomo Del Rio
# Creation date: 10 November 2021

from typing import Tuple

from geographiclib.geodesic import Geodesic

from simulator.units_of_measure import Longitude, Latitude, Meters


class MapLimits:
    """ A (square) region on a map """

    def __init__(self, left_lon: Longitude, bottom_lat: Latitude, right_lon: Longitude, top_lat: Latitude):
        """ Builds a new MapLimits. Longitudes *must* be within [0,180] and can't overlap the prime meridian. """
        self.left_lon = left_lon
        self.bottom_lat = bottom_lat
        self.right_lon = right_lon
        self.top_lat = top_lat

    def latitude_extent(self) -> Latitude:
        return self.top_lat - self.bottom_lat

    def longitude_extent(self) -> Longitude:
        return self.right_lon - self.left_lon

    def latitude_distance(self) -> Meters:
        d1 = Geodesic.WGS84.Inverse(self.bottom_lat, self.left_lon, self.top_lat, self.left_lon,
                                    outmask=Geodesic.DISTANCE)
        return d1["s12"]

    def max_longitude_distance(self) -> Meters:
        d1 = Geodesic.WGS84.Inverse(self.bottom_lat, self.left_lon, self.bottom_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        d2 = Geodesic.WGS84.Inverse(self.top_lat, self.left_lon, self.top_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        return max(d1["s12"], d2["s12"])

    def relative_position(self, lat: Latitude, lon: Longitude) -> Tuple[float, float]:
        lat_rel = (lat - self.bottom_lat) / self.latitude_extent()
        lon_rel = (lon - self.left_lon) / self.longitude_extent()
        return lat_rel, lon_rel

    def absolute_position(self, lat_rel: float, lon_rel: float) -> Tuple[Latitude, Longitude]:
        lat = lat_rel * self.latitude_extent() + self.bottom_lat
        lon = lon_rel * self.longitude_extent() + self.left_lon
        return lat, lon

    def in_boundary(self, lat: Latitude, lon: Longitude) -> bool:
        return self.left_lon <= lon <= self.right_lon and self.bottom_lat <= lat <= self.top_lat
