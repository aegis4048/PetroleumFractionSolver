import unittest
import sys
import pint
import numpy as np
import numpy.testing as npt
import pandas as pd
import warnings
import random
from scipy.optimize import newton
from thermo import ChemicalConstantsPackage

sys.path.append('.')

from pfsolver import PropertyTable, SCNProperty
from pfsolver.customExceptions import SCNPropertyWarning, PropertyTableWarning, ThermoMissingValueWarning
from pfsolver import correlations
from pfsolver import utilities
from pfsolver import eos


UREG = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)


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


class Test_PREOS(TestNumericalMethods):

    def test_PREOS_analytical_vs_numerical(self):
        comp = {'n-butane': 0.1, 'n-hexane': 0.9}
        names = list(comp.keys())
        zs = list(comp.values())

        constants = ChemicalConstantsPackage.constants_from_IDs(names)
        Pcs = np.array(constants.Pcs)
        Tcs = np.array(constants.Tcs)
        omegas = np.array(constants.omegas)
        Tbs = np.array(constants.Tbs)
        mws = np.array(constants.MWs)
        mw = np.sum(np.array(constants.MWs) * np.array(list(comp.values())))

        # test conditions
        Ts = [UREG('%.15f degF' % T).to('kelvin')._magnitude for T in np.arange(0, 401, 40)] # Test T between 0 ~ 400 by 40 increment
        Ps = [UREG('%.15f psi' % P).to('pascal')._magnitude for P in np.arange(10, 1011, 100)]  # Test P between 10 ~ 2000 by 50 increment

        for T, P in zip(Ts, Ps):
            PR_analytical = eos.PR(T, P, Tcs[0], Pcs[0], omegas[0], mws[0], analytical=True)
            PR_numerical = eos.PR(T, P, Tcs[0], Pcs[0], omegas[0], mws[0], analytical=False)
            PR78_analytical = eos.PR78(T, P, Tcs[0], Pcs[0], omegas[0], mws[0], analytical=True)
            PR78_numerical = eos.PR78(T, P, Tcs[0], Pcs[0], omegas[0], mws[0], analytical=False)

            PRMIX_analytical = eos.PRMIX(T, P, Tcs, Pcs, omegas, zs, mws=mws, analytical=True)
            PRMIX_numerical = eos.PRMIX(T, P, Tcs, Pcs, omegas, zs, mws=mws, analytical=False)
            PRMIX78_analytical = eos.PRMIX78(T, P, Tcs, Pcs, omegas, zs, mws=mws, analytical=True)
            PRMIX78_numerical = eos.PRMIX78(T, P, Tcs, Pcs, omegas, zs, mws=mws, analytical=False)

            sig_figs = 8
            self.assertAlmostEqualSigFig(PR_analytical.V_liq, PR_numerical.V_liq, sig_figs=sig_figs)
            self.assertAlmostEqualSigFig(PR78_analytical.V_gas, PR78_numerical.V_gas, sig_figs=sig_figs)
            self.assertAlmostEqualSigFig(PRMIX_analytical.rho_liq, PRMIX_numerical.rho_liq, sig_figs=sig_figs)
            self.assertAlmostEqualSigFig(PRMIX78_analytical.rho_gas, PRMIX78_numerical.rho_gas, sig_figs=sig_figs)

    def test_PR_vs_PR78(self):
        comp = {'n-butane': 1}
        names = list(comp.keys())
        zs = list(comp.values())

        constants = ChemicalConstantsPackage.constants_from_IDs(names)
        Pc = constants.Pcs[0]
        Tc = constants.Tcs[0]
        omega = constants.omegas[0]
        omega_2 = 0.5  # PR78 is different from PR when omega > 0.491
        mw = constants.MWs[0]

        Ts = [UREG('%.15f degF' % T).to('kelvin')._magnitude for T in np.arange(0, 401, 40)] # Test T between 0 ~ 400 by 40 increment
        Ps = [UREG('%.15f psi' % P).to('pascal')._magnitude for P in np.arange(10, 1011, 100)]  # Test P between 10 ~ 2000 by 50 increment

        for T, P in zip(Ts, Ps):
            PR = eos.PR(T, P, Tc, Pc, omega, mw)
            PR78 = eos.PR78(T, P, Tc, Pc, omega, mw)

            sig_figs = 8
            self.assertAlmostEqualSigFig(PR.V_liq, PR78.V_liq, sig_figs=sig_figs)
            self.assertAlmostEqualSigFig(PR.V_gas, PR78.V_gas, sig_figs=sig_figs)
            self.assertAlmostEqualSigFig(PR.rho_liq, PR78.rho_liq, sig_figs=sig_figs)
            self.assertAlmostEqualSigFig(PR.rho_gas, PR78.rho_gas, sig_figs=sig_figs)

            PR = eos.PR(T, P, Tc, Pc, omega_2, mw)
            PR78 = eos.PR78(T, P, Tc, Pc, omega_2, mw)
            self.assertNotAlmostEqualSigFig(PR.V_liq, PR78.V_liq, sig_figs=sig_figs)
            self.assertNotAlmostEqualSigFig(PR.V_gas, PR78.V_gas, sig_figs=sig_figs)
            self.assertNotAlmostEqualSigFig(PR.rho_liq, PR78.rho_liq, sig_figs=sig_figs)
            self.assertNotAlmostEqualSigFig(PR.rho_gas, PR78.rho_gas, sig_figs=sig_figs)

    def test_PRMIX_vs_PRMIX78(self):
        comp = {'n-butane': 0.1, 'n-hexane': 0.9}
        names = list(comp.keys())
        zs = list(comp.values())

        constants = ChemicalConstantsPackage.constants_from_IDs(names)
        Pcs = np.array(constants.Pcs)
        Tcs = np.array(constants.Tcs)
        omegas = np.array(constants.omegas)
        omegas_2 = np.array([0.6, 0.6])
        Tbs = np.array(constants.Tbs)
        mws = np.array(constants.MWs)
        mw = np.sum(np.array(constants.MWs) * np.array(list(comp.values())))

        Ts = [UREG('%.15f degF' % T).to('kelvin')._magnitude for T in np.arange(0, 401, 40)]
        Ps = [UREG('%.15f psi' % P).to('pascal')._magnitude for P in np.arange(10, 1011, 100)]

        for T, P in zip(Ts, Ps):
            PRMIX = eos.PRMIX(T, P, Tcs, Pcs, omegas, zs, mws=mws)
            PRMIX78 = eos.PRMIX78(T, P, Tcs, Pcs, omegas, zs, mws=mws)

            sig_figs = 8
            self.assertAlmostEqualSigFig(PRMIX.V_liq, PRMIX78.V_liq, sig_figs=sig_figs)
            self.assertAlmostEqualSigFig(PRMIX.V_gas, PRMIX78.V_gas, sig_figs=sig_figs)
            self.assertAlmostEqualSigFig(PRMIX.rho_liq, PRMIX78.rho_liq, sig_figs=sig_figs)
            self.assertAlmostEqualSigFig(PRMIX.rho_gas, PRMIX78.rho_gas, sig_figs=sig_figs)

            PRMIX = eos.PRMIX(T, P, Tcs, Pcs, omegas_2, zs, mws=mws)
            PRMIX78 = eos.PRMIX78(T, P, Tcs, Pcs, omegas_2, zs, mws=mws)

            self.assertNotAlmostEqualSigFig(PRMIX.V_liq, PRMIX78.V_liq, sig_figs=sig_figs)
            self.assertNotAlmostEqualSigFig(PRMIX.V_gas, PRMIX78.V_gas, sig_figs=sig_figs)
            self.assertNotAlmostEqualSigFig(PRMIX.rho_liq, PRMIX78.rho_liq, sig_figs=sig_figs)
            self.assertNotAlmostEqualSigFig(PRMIX.rho_gas, PRMIX78.rho_gas, sig_figs=sig_figs)

    def test_PR_vs_PRMIX(self):
        comp = {'n-butane': 1}
        names = list(comp.keys())
        zs = list(comp.values())

        constants = ChemicalConstantsPackage.constants_from_IDs(names)
        Pcs = np.array(constants.Pcs)
        Tcs = np.array(constants.Tcs)
        omegas = np.array(constants.omegas)
        mws = np.array(constants.MWs)

        Ts = [UREG('%.15f degF' % T).to('kelvin')._magnitude for T in np.arange(0, 401, 40)] # Test T between 0 ~ 400 by 40 increment
        Ps = [UREG('%.15f psi' % P).to('pascal')._magnitude for P in np.arange(10, 1011, 100)]  # Test P between 10 ~ 2000 by 50 increment

        for T, P in zip(Ts, Ps):
            PR = eos.PR(T, P, Tcs[0], Pcs[0], omegas[0], mws[0])
            PRMIX = eos.PRMIX(T, P, Tcs, Pcs, omegas, zs, mws=mws)

            sig_figs = 8
            self.assertAlmostEqualSigFig(PR.V_liq, PRMIX.V_liq, sig_figs=sig_figs)
            self.assertAlmostEqualSigFig(PR.V_gas, PRMIX.V_gas, sig_figs=sig_figs)
            self.assertAlmostEqualSigFig(PR.rho_liq, PRMIX.rho_liq, sig_figs=sig_figs)
            self.assertAlmostEqualSigFig(PR.rho_gas, PRMIX.rho_gas, sig_figs=sig_figs)

    def test_PR(self):
        comp = {'n-butane': 1}
        names = list(comp.keys())
        zs = list(comp.values())

        constants = ChemicalConstantsPackage.constants_from_IDs(names)
        Pc = constants.Pcs[0]
        Tc = constants.Tcs[0]
        omega = constants.omegas[0]
        mw = constants.MWs[0]

        T = UREG('%.15f degF' % 60).to('kelvin')._magnitude
        P = UREG('%.15f psi' % 200).to('pascal')._magnitude
        obj = eos.PR(T, P, Tc, Pc, omega, mw)
        self.assertAlmostEqualSigFig(obj.rho_liq, 617098.5, sig_figs=7)  # Promax: 616991.184024, liquid phase
        self.assertAlmostEqualSigFig(obj.rho_gas, 73846.51, sig_figs=7)

        T = UREG('%.15f degF' % 400).to('kelvin')._magnitude
        P = UREG('%.15f psi' % 200).to('pascal')._magnitude
        obj = eos.PR(T, P, Tc, Pc, omega, mw)
        self.assertAlmostEqualSigFig(obj.rho_liq, 548558.5, sig_figs=7)
        self.assertAlmostEqualSigFig(obj.rho_gas, 22394.93, sig_figs=7)  # Promax: 22402.33066525, vapor phase

        T = UREG('%.15f degF' % 500).to('kelvin')._magnitude
        P = UREG('%.15f psi' % 600).to('pascal')._magnitude
        obj = eos.PR(T, P, Tc, Pc, omega, mw)
        self.assertAlmostEqualSigFig(obj.rho_liq, 853268.5, sig_figs=7)
        self.assertAlmostEqualSigFig(obj.rho_gas, 67358.87, sig_figs=7)  # Promax: 67426.14919366, supercritical

    def test_PRMIX(self):
        comp = {'n-butane': 0.1, 'n-hexane': 0.9}
        names = list(comp.keys())
        zs = list(comp.values())

        constants = ChemicalConstantsPackage.constants_from_IDs(names)
        Pcs = np.array(constants.Pcs)
        Tcs = np.array(constants.Tcs)
        omegas = np.array(constants.omegas)
        Tbs = np.array(constants.Tbs)
        mws = np.array(constants.MWs)
        mw = np.sum(np.array(constants.MWs) * np.array(list(comp.values())))

        T = UREG('%.15f degF' % 60).to('kelvin')._magnitude
        P = UREG('%.15f psi' % 200).to('pascal')._magnitude
        obj = eos.PRMIX(T, P, Tcs, Pcs, omegas, zs, mws=mws)
        self.assertAlmostEqualSigFig(obj.rho_liq, 670566.7, sig_figs=7)  # Promax: 667992.3025038, liquid phase
        self.assertAlmostEqualSigFig(obj.rho_gas, 110272.4, sig_figs=7)

        T = UREG('%.15f degF' % 300).to('kelvin')._magnitude
        P = UREG('%.15f psi' % 100).to('pascal')._magnitude
        obj = eos.PRMIX(T, P, Tcs, Pcs, omegas, zs, mws=mws)
        self.assertAlmostEqualSigFig(obj.rho_liq, 503867.7, sig_figs=7)
        self.assertAlmostEqualSigFig(obj.rho_gas, 19475.67, sig_figs=7)  # Promax: 19479.7551467, vapor phase

        T = UREG('%.15f degF' % 600).to('kelvin')._magnitude
        P = UREG('%.15f psi' % 600).to('pascal')._magnitude
        obj = eos.PRMIX(T, P, Tcs, Pcs, omegas, zs, mws=mws)
        self.assertAlmostEqualSigFig(obj.rho_liq, 673208.1, sig_figs=7)
        self.assertAlmostEqualSigFig(obj.rho_gas, 100304.8, sig_figs=7)  # Promax: 101420.8790536, supercritical








