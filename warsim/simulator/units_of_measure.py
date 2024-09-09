# units_of_measure.py
#
# A definition of common units of measure types
#
# Author: Giacomo Del Rio
# Creation date: 17 May 2023

import math

# --- Constants
DEG_TO_RAD = math.pi / 180.0
PI_HALF = math.pi / 2.0

# --- Type alias
Seconds = float
Meters = float
Degree = float  # [0, 360)
Latitude = float  # [-90; +90]
Longitude = float  # [0; 180]
MetersPerSecond = float
DegreesPerSecond = float
