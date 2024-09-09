# angles.py
#
# Angles computations
#
# Author: Giacomo Del Rio
# Creation date: 10 November 2021

from simulator.units_of_measure import Degree


def normalize_angle(a: float) -> Degree:
    """ Normalize 'a' into [0, 360)

    :param a: the angle to be normalized
    :return: the normalized angle
    """
    while a >= 360.0:
        a -= 360
    while a < 0.0:
        a += 360
    return a


def sum_angles(a: Degree, b: Degree) -> Degree:
    """ Sum two angles and normalize the result

    :param a: first angle
    :param b: second angle
    :return: the normalized sum of the angles
    """
    return normalize_angle(a + b)


def signed_heading_diff(actual: Degree, desired: Degree) -> Degree:
    """ Compute the signed difference between two angles (i.e. the shortest path angle from actual to desired)

    :param actual: the actual angle in [0, 360)
    :param desired: the desired angle in [0, 360)
    :return: the signed difference between actual and desired
    """
    delta = desired - actual
    if delta < -180:
        delta = 360 + delta
    if delta > 180:
        delta = -360 + delta
    return delta
