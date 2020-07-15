from unittest import TestCase

import numpy as np

from calculators.heat_index_calculator import HeatIndexCalculator
from calculators.numba_heat_index_calculator import NumbaHeatIndexCalculator
from calculators.numba_vectorized_heat_index import calculate_heat_index
from calculators.numpy_heat_index_calculator import NumpyHeatIndexCalculator


class TestMethodEquality(TestCase):

    def test_methods_dry_adjustment(self):
        # But it's a dry heat!
        temp = np.zeros((2, 2, 2))
        temp[:, :, :] = 95
        rh = np.zeros((2, 2, 2))
        rh[:, :, :] = 0.10

        hi1 = HeatIndexCalculator().calculate_heat_index(temp, rh)

        hi2 = NumbaHeatIndexCalculator().calculate_heat_index(temp, rh)
        np.testing.assert_array_almost_equal(hi1, hi2)

        hi3 = NumpyHeatIndexCalculator().calculate_heat_index(temp, rh)
        np.testing.assert_array_almost_equal(hi1, hi3)

        hi4 = calculate_heat_index(temp, rh)
        np.testing.assert_array_almost_equal(hi1, hi4)

    def test_methods_humid_adjustment(self):
        # Illinois in July
        temp = np.zeros((2, 2, 2))
        temp[:, :, :] = 85
        rh = np.zeros((2, 2, 2))
        rh[:, :, :] = 0.90

        hi1 = HeatIndexCalculator().calculate_heat_index(temp, rh)

        hi2 = NumbaHeatIndexCalculator().calculate_heat_index(temp, rh)
        np.testing.assert_array_almost_equal(hi1, hi2)

        hi3 = NumpyHeatIndexCalculator().calculate_heat_index(temp, rh)
        np.testing.assert_array_almost_equal(hi1, hi3)

        hi4 = calculate_heat_index(temp, rh)
        np.testing.assert_array_almost_equal(hi1, hi4)

    def test_methods_no_adjustment(self):
        temp = np.zeros((2, 2, 2))
        temp[:, :, :] = 90
        rh = np.zeros((2, 2, 2))
        rh[:, :, :] = 0.65

        hi1 = HeatIndexCalculator().calculate_heat_index(temp, rh)

        hi2 = NumbaHeatIndexCalculator().calculate_heat_index(temp, rh)
        np.testing.assert_array_almost_equal(hi1, hi2)

        hi3 = NumpyHeatIndexCalculator().calculate_heat_index(temp, rh)
        np.testing.assert_array_almost_equal(hi1, hi3)

        hi4 = calculate_heat_index(temp, rh)
        np.testing.assert_array_almost_equal(hi1, hi4)


    def test_all_methods_equal_random(self):
        temp = np.random.default_rng().uniform(
            low=50,
            high=100,
            size=(4, 100, 300)
        )
        rh = np.random.default_rng().uniform(
            low=0.05,
            high=0.95,
            size=(4, 100, 300)
        )

        hi1 = HeatIndexCalculator().calculate_heat_index(temp, rh)

        hi2 = NumbaHeatIndexCalculator().calculate_heat_index(temp, rh)
        np.testing.assert_array_almost_equal(hi1, hi2)

        hi3 = NumpyHeatIndexCalculator().calculate_heat_index(temp, rh)
        np.testing.assert_array_almost_equal(hi1, hi3)

        hi4 = calculate_heat_index(temp, rh)
        np.testing.assert_array_almost_equal(hi1, hi4)
