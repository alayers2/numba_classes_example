import math

from numba import vectorize, jit_module

"""
This file consists of module level functions, and showcases the jit_module method for automatically jit-ing all 
of the functions above itself. Those functions are then used by the vectorized `calculate_heat_index` method
at the bottom of the file.
"""


def _calculate_simple_heat_index(temp: float, rh: float) -> float:
    """
    Simplified heat index regression, applicable when HI is under 80
    """
    return 0.5 * (temp + 61.0 + ((temp - 68.0) * 1.2) + (rh * 0.094))


def _calculate_full_regression(temp: float, rh: float) -> float:
    """
    Full heat index regression, applicable when HI from simple regression is over 80
    """
    return -42.379 + (2.04901523 * temp) + (10.14333127 * rh) \
           - (.22475541 * temp * rh) - (.00683783 * temp * temp) - (.05481717 * rh * rh) \
           + (.00122874 * temp * temp * rh) + (.00085282 * temp * rh * rh) - (.00000199 * temp * temp * rh * rh)


def _calculate_dry_adjustment(temp: float, rh: float) -> float:
    """
    Calculates an adjustment to heat index applicable when humidity is less than 13%
    and the temp is between 80 and 112
    """
    return ((13 - rh) / 4.0) * math.sqrt((17 - math.fabs(temp - 95.0)) / 17)


def _calculate_humid_adjustment(temp: float, rh: float) -> float:
    """
    Calculates and adjustment to the head index applicable when humidity is greater than
    85% and temperature is between 80 and 87
    """
    return ((rh - 85) / 10.0) * ((87 - temp) / 5.0)


# call the jit_module function to automatically jit the functions declared above this line within the module.
jit_module(nopython=True, error_model="numpy")


@vectorize(["double(double, double)"], nopython=True, target='parallel')
def calculate_heat_index(temp, rh):
    heat_index = _calculate_simple_heat_index(temp, rh)

    if heat_index > 80.0:
        heat_index = _calculate_full_regression(temp, rh)

        if rh < 0.13 and (80.0 < temp < 112.0):
            heat_index = heat_index - _calculate_dry_adjustment(temp, rh)

        if rh > 0.85 and (80.0 < temp < 87.0):
            heat_index = heat_index + _calculate_humid_adjustment(temp, rh)

    return heat_index
