import math

import numpy as np


class HeatIndexCalculator(object):
    """
    This is the baseline implementation of the heat index calculation. Given two input arrays of temp and rh, we iterate
    over them and apply the heat index calculation to each element.
    """

    def calculate_heat_index(self, temp: np.ndarray, rh: np.ndarray) -> np.ndarray:
        """
        Main function, creates an output array the same shape as the the inputs, and then populates it
        by iterating over each dimension and calculates the heat index for each pair of elements from the
        temp and RH arrays
        """
        hi = np.zeros_like(temp)
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                for k in range(temp.shape[2]):
                    hi[i, j, k] = self._calculate_heat_index(temp[i, j, k], rh[i, j, k])

        return hi

    def _calculate_heat_index(self, temp: float, rh: float) -> float:
        """
        Elementwise application of the heat index calculation. The basic algorithm is:
        1. Calculate a result with the simple regression
        2. If the heat index is over 80.0, apply the full regression
        3. Check the RH and temperature range to see if we need to subtract an adjustment (what I'm calling the 'dry'
            adjustment since it applies when RH is low)
        4. Check the RH and temperature range to see if we need to add an adjustment (what I'm calling the 'humid'
            adjustment ssince it applies when RH is high)
        """
        heat_index = self._calculate_simple_heat_index(temp, rh)

        if heat_index > 80.0:
            heat_index = self._calculate_full_regression(temp, rh)

            if rh < 0.13 and (80.0 < temp < 112.0):
                heat_index = heat_index - self._calculate_dry_adjustment(temp, rh)

            if rh > 0.85 and (80.0 < temp < 87.0):
                heat_index = heat_index + self._calculate_humid_adjustment(temp, rh)

        return heat_index

    def _calculate_simple_heat_index(self, temp: float, rh: float) -> float:
        """
        Simplified heat index regression, applicable when HI is under 80
        """
        return 0.5 * (temp + 61.0 + ((temp - 68.0) * 1.2) + (rh * 0.094))

    def _calculate_full_regression(self, temp: float, rh: float) -> float:
        """
        Full heat index regression, applicable when HI from simple regression is over 80
        """
        return -42.379 + (2.04901523 * temp) + (10.14333127 * rh) \
            - (.22475541 * temp * rh) - (.00683783 * temp * temp) - (.05481717 * rh * rh) \
            + (.00122874 * temp * temp * rh) + (.00085282 * temp * rh * rh) - (.00000199 * temp * temp * rh * rh)

    def _calculate_dry_adjustment(self, temp: float, rh: float) -> float:
        """
        Calculates an adjustment to heat index applicable when humidity is less than 13%
        and the temp is between 80 and 112
        """
        return ((13 - rh) / 4.0) * math.sqrt((17.0 - math.fabs(temp - 95.0)) / 17.0)

    def _calculate_humid_adjustment(self, temp: float, rh: float) -> float:
        """
        Calculates and adjustment to the head index applicable when humidity is greater than
        85% and temperature is between 80 and 87
        """
        return ((rh - 85) / 10.0) * ((87 - temp) / 5.0)
