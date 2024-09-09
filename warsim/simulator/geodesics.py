# geodesics.py
#
# Geodesics computations
#
# Author: Giacomo Del Rio
# Creation date: 10 November 2021


from typing import Tuple

from geographiclib.geodesic import Geodesic

from simulator.angles import normalize_angle
from simulator.units_of_measure import Longitude, Latitude, Degree, Meters


def geodetic_distance(lat_1: Latitude, lon_1: Longitude, lat_2: Latitude, lon_2: Longitude) -> Meters:
    """ Distance between two points """
    r = Geodesic.WGS84.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=Geodesic.DISTANCE)
    return r["s12"]


def geodetic_bearing(lat_1: Latitude, lon_1: Longitude, lat_2: Latitude, lon_2: Longitude) -> Degree:
    """ Bearing between two points """
    r = Geodesic.WGS84.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=Geodesic.AZIMUTH)
    return normalize_angle(r["azi1"])


def geodetic_direct(lat: Latitude, lon: Longitude, heading: Degree, distance: float) -> Tuple[Latitude, Longitude]:
    """ Compute the next point if we start from (lat, lon) and we go straight 'distance' in the 'heading' direction """
    d = Geodesic.WGS84.Direct(lat, lon, heading, distance, outmask=Geodesic.LATITUDE | Geodesic.LONGITUDE)
    return d["lat2"], d["lon2"]
