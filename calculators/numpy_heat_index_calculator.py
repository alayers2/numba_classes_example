import numpy as np


class NumpyHeatIndexCalculator(object):
    """
    This class implements the heat index equations in much the same way as the naive implementation, but when we get
    to the conditional calculations and adjustments we use np.where to prevent breaking out into loops to apply things
    element-by-element. My personal opinion is that using np.where is less-readable than if statements.
    """

    def calculate_heat_index(self, temp: np.ndarray, rh: np.ndarray) -> np.ndarray:
        heat_index = self._calculate_simple_heat_index(temp, rh)

        # Use np.where to calculate the full regression when temp is over 80 degrees.
        full_regression = heat_index > 80.0
        heat_index = np.where(
            full_regression,
            self._calculate_full_regression(temp, rh),
            heat_index
        )

        # Use np.where to apply the dry adjustment for elements where it applies.
        heat_index = np.where(
            full_regression & (rh < 0.13) & (80 < temp) & (temp < 112),
            heat_index - self._calculate_dry_adjustment(temp, rh),
            heat_index
        )

        # And likewise for the humid adjustment.
        heat_index = np.where(
            full_regression & (rh > 0.85) & (80 < temp) & (temp < 87),
            heat_index + self._calculate_humid_adjustment(temp, rh),
            heat_index
        )

        return heat_index

    def _calculate_simple_heat_index(self, temp: np.ndarray, rh: np.ndarray) -> np.ndarray:
        return 0.5 * (temp + 61.0 + ((temp - 68.0) * 1.2) + (rh * 0.094))

    def _calculate_full_regression(self, temp: np.ndarray, rh: np.ndarray) -> np.ndarray:
        return -42.379 + (2.04901523 * temp) + (10.14333127 * rh) \
               - (.22475541 * temp * rh) - (.00683783 * temp * temp) - (.05481717 * rh * rh) \
               + (.00122874 * temp * temp * rh) + (.00085282 * temp * rh * rh) - (.00000199 * temp * temp * rh * rh)

    def _calculate_dry_adjustment(self, temp: np.ndarray, rh: np.ndarray) -> np.ndarray:
        """
        Calculates an adjustment to heat index applicable when humidity is less than 13%
        and the temp is between 80 and 112
        """
        return ((13 - rh) / 4.0) * np.sqrt((17.0 - np.fabs(temp - 95.0)) / 17.0)

    def _calculate_humid_adjustment(self, temp: np.ndarray, rh: np.ndarray) -> np.ndarray:
        """
        Calculates and adjustment to the head index applicable when humidity is greater than
        85% and temperature is between 80 and 87
        """
        return ((rh - 85) / 10.0) * ((87 - temp) / 5.0)
