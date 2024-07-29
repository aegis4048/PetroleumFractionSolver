import unittest
import sys
import pint
import numpy as np
import numpy.testing as npt


class TestNumericalMethods(unittest.TestCase):
    def assertAlmostEqualSigFig(self, actual, expected, sig_figs=7):
        npt.assert_approx_equal(actual, expected, significant=sig_figs)

    def assertNotAlmostEqualSigFig(self, actual, expected, sig_figs=7):
        try:
            npt.assert_approx_equal(actual, expected, significant=sig_figs)
        except AssertionError:
            pass  # If assertion fails, it means the values are not approximately equal, which is expected
            pass  # If assertion fails, it means the values are not approximately equal, which is expected
        else:
            raise AssertionError(
                f"Items are unexpectedly equal to {sig_figs} significant digits:\n"
                f"LEFT: {actual}\n"
                f"RIGHT: {expected}"
            )

