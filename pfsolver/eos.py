import numpy as np
import pint
from thermo import ChemicalConstantsPackage
from matplotlib import pyplot as plt
import time

from pfsolver import utilities
from pfsolver import config


UREG = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)


class calculatorPR:
    @staticmethod
    def calc_Tr(T, Tc):
        return T / Tc

    @staticmethod
    def calc_a(Tc, Pc):
        return 0.45724 * (config.constants['R'] ** 2) * (Tc ** 2) / Pc

    @staticmethod
    def calc_b(Tc, Pc):
        return 0.07780 * config.constants['R'] * Tc / Pc

    @staticmethod
    def calc_kappa(omega):
        return 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2

    @staticmethod
    def calc_alpha(kappa, Tr):
        return (1 + kappa * (1 - np.sqrt(Tr))) ** 2

    @staticmethod
    def calc_a_alpha(a, alpha):
        return a * alpha

    @staticmethod
    def calc_A(a_alpha, T, P):
        return a_alpha * P / (config.constants['R'] ** 2 * T ** 2)

    @staticmethod
    def calc_B(b, T, P):
        return b * P / (config.constants['R'] * T)

    @staticmethod
    def calc_V_liq(Z_liq, T, P):
        return Z_liq * config.constants['R'] * T / P  # m^3/mol

    @staticmethod
    def calc_V_gas(Z_gas, T, P):
        return Z_gas * config.constants['R'] * T / P  # m^3/mol

    @staticmethod
    def solve_cubic(A, B):
        # Coefficients of the cubic equation
        coeffs = [1, -(1 - B), A - 3 * B ** 2 - 2 * B, -(A * B - B ** 2 - B ** 3)]
        return np.array(sorted(np.roots(coeffs)))

    @staticmethod
    def solve_cubic_analytical(A, B):
        a = -(1 - B)
        b = A - 3 * B ** 2 - 2 * B
        c = -(A * B - B ** 2 - B ** 3)

        Q = (a ** 2 - 3 * b) / 9
        K = (2 * a ** 3 - 9 * a * b + 27 * c) / 54

        if K ** 2 < Q ** 3:
            theta = np.arccos(K / np.sqrt(Q ** 3))
            x1 = -2 * np.sqrt(Q) * np.cos(theta / 3) - a / 3
            x2 = -2 * np.sqrt(Q) * np.cos((theta + 2 * np.pi) / 3) - a / 3
            x3 = -2 * np.sqrt(Q) * np.cos((theta - 2 * np.pi) / 3) - a / 3
        else:
            S = np.cbrt(-K + np.sqrt(K ** 2 - Q ** 3))
            T = np.cbrt(-K - np.sqrt(K ** 2 - Q ** 3))
            x1 = S + T - a / 3
            x2 = -(S + T) / 2 - a / 3 + 1j * np.sqrt(3) * (S - T) / 2
            x3 = -(S + T) / 2 - a / 3 - 1j * np.sqrt(3) * (S - T) / 2

        return np.array(sorted([x1, x2, x3]))

    @staticmethod
    def Z_obj_func(Z, A, B):
        return Z ** 3 - (1 - B) * Z ** 2 + (A - 2 * B - 3 * B ** 2) * Z - (
                A * B - B ** 2 - B ** 3)

    @staticmethod
    def calc_rho(mw, V_liq):
        return mw / V_liq  # g/m^3  

    
    @staticmethod
    def identify_phase(roots_real, V_liq, V_gas, T, P, tol=0.2):
        if len(roots_real) == 3:
            phase = 'two phase'
        elif len(roots_real) == 1:
            assert V_liq == V_gas
            V_real_gas = utilities.real_gas_molar_volume(roots_real[0], T, P)
            
            # if V_liq is within 20% proximity of V_real_gas, then it is a gas phase
            if abs(V_liq - V_real_gas) / V_real_gas < tol:
                phase = 'gas'
            else:
                phase = 'liquid'
        else:
            raise ValueError('Number of roots should be either 1 or 3. This is a bug')
        return phase


class calculatorPR78(calculatorPR):

    def __init__(self):
        super().__init__()

    @staticmethod
    def calc_kappa(omega):
        if omega <= 0.491:
            kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
        else:
            kappa = 0.379642 + 1.48503 * omega - 0.164423 * omega ** 2 + 0.016666 * omega ** 3
        return kappa


class calculatorPRMIX(calculatorPR):

    def __init__(self):
        super().__init__()

    @staticmethod
    def calc_a_alpha(zs, ais_alphas):
        a_alpha = 0
        for i in range(len(zs)):
            for j in range(len(zs)):
                a_alpha += zs[i] * zs[j] * (ais_alphas[i] * ais_alphas[j]) ** 0.5
        return a_alpha

    @staticmethod
    def calc_b(zs, Tcs, Pcs):
        bis = calculatorPR.calc_b(Tcs, Pcs)
        b = 0
        for i in range(len(zs)):
            b += zs[i] * bis[i]
        return b

    @staticmethod
    def calc_kappas(omegas):
        kappas = []
        for omega in omegas:
            kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
            kappas.append(kappa)
        return kappas


class calculatorPRMIX78(calculatorPRMIX):

    def __init__(self):
        super().__init__()

    @staticmethod
    def calc_kappas(omegas):
        kappas = []
        for omega in omegas:
            if omega <= 0.491:
                kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
            else:
                kappa = 0.379642 + 1.48503 * omega - 0.164423 * omega ** 2 + 0.016666 * omega ** 3
            kappas.append(kappa)
        return kappas


def plot_Z(x, y, roots, roots_real, T, P, eos_name):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(x, y, label='f(Z)', color='k')
    ax.axhline(0, color='r', label='y=0')
    for i in range(len(roots)):
        real_part = roots[i].real
        imag_part = roots[i].imag

        if np.abs(imag_part) < 1e-5:
            label = 'root = %.3f' % real_part
        else:
            sign = '+' if imag_part > 0 else '-'
            label = f"root(i) = {real_part:.3f}{sign}j{np.abs(imag_part):.3f}"

        ax.axvline(roots_real[i], color='b', linestyle='--', label=label)

    ax.legend(fontsize=10, ncol=1, loc='upper right')
    ax.grid(axis='y', linestyle='--', color='#acacac', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ymax_ = abs(min(y) * 5)
    y_margin = abs(ymax_ - min(y)) * 0.05
    ymin = min(y) - y_margin
    ymax = ymax_ + y_margin
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel('Compressibility: Z', fontsize=13)
    ax.set_ylabel('Objective function: f(Z)', fontsize=13)
    ax.text(0.99, 0.1, 'aegis4048.github.io', fontsize=12, ha='right', va='center',
            transform=ax.transAxes, color='grey', alpha=0.5)

    T_F = round(UREG('%.15f kelvin' % T).to('degF')._magnitude, 1)
    P_psia = round(UREG('%.15f pascal' % P).to('psi')._magnitude, 1)

    bold_txt = utilities.setbold('Roots of the Cubic Equation')
    plain_txt = r', %s @%sF, %spsia' % (eos_name, T_F, P_psia)
    fig.suptitle(bold_txt + plain_txt, verticalalignment='top', x=0.01, horizontalalignment='left', fontsize=13,
                 y=0.96)
    yloc = 0.895
    ax.annotate('', xy=(0.01, yloc + 0.01), xycoords='figure fraction', xytext=(1.02, yloc + 0.01),
                arrowprops=dict(arrowstyle="-", color='k', lw=0.7))

    fig.tight_layout()
    return fig, ax


class PR(object):

    def __init__(self, T, P, Tc, Pc, omega, mw=None, analytical=True, _update_now=True):

        self.eos_name = 'Peng-Robinson EOS 1976'

        self.T = T  # K
        self.P = P  # pascal
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega

        self.mw = mw
        self.R = config.constants['R']  # 8.31446261815324 ((m^3-Pa)/(mol-K))

        self.Tr = calculatorPR.calc_Tr(self.T, self.Tc)
        self.a = calculatorPR.calc_a(self.Tc, self.Pc)
        self.b = calculatorPR.calc_b(self.Tc, self.Pc)
        self.kappa = calculatorPR.calc_kappa(self.omega)
        self.alpha = calculatorPR.calc_alpha(self.kappa, self.Tr)
        self.a_alpha = calculatorPR.calc_a_alpha(self.a, self.alpha)
        self.A = calculatorPR.calc_A(self.a_alpha, self.T, self.P)
        self.B = calculatorPR.calc_B(self.b, self.T, self.P)

        if analytical:
            self.roots = calculatorPR.solve_cubic_analytical(self.A, self.B)
        else:
            self.roots = calculatorPR.solve_cubic(self.A, self.B)

        self.roots_real = self.roots[self.roots.imag == 0].real
        self.Z_liq = min(self.roots_real)  # lowest real root is liquid
        self.Z_gas = max(self.roots_real)
        self.V_liq = calculatorPR.calc_V_liq(self.Z_liq, self.T, self.P)  # m^3/mol
        self.V_gas = calculatorPR.calc_V_gas(self.Z_gas, self.T, self.P)

        if self.mw is not None:
            self.rho_liq = calculatorPR.calc_rho(self.mw, self.V_liq)
            self.rho_gas = calculatorPR.calc_rho(self.mw, self.V_gas)

    def plot(self):
        x = np.linspace(0, 1.05)
        y = calculatorPR.Z_obj_func(x, self.A, self.B)
        fig, ax = plot_Z(x, y, self.roots, self.roots_real, self.T, self.P, self.eos_name)
        return fig, ax


class PR78:
    def __init__(self, T, P, Tc, Pc, omega, mw=None, analytical=True):

        self.eos_name = 'Peng-Robinson EOS 1978'

        self.T = T  # K
        self.P = P  # pascal
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega

        self.mw = mw
        self.R = config.constants['R']  # 8.31446261815324 ((m^3-Pa)/(mol-K))

        self.Tr = calculatorPR78.calc_Tr(self.T, self.Tc)
        self.a = calculatorPR78.calc_a(self.Tc, self.Pc)
        self.b = calculatorPR78.calc_b(self.Tc, self.Pc)
        self.kappa = calculatorPR78.calc_kappa(self.omega)
        self.alpha = calculatorPR78.calc_alpha(self.kappa, self.Tr)
        self.a_alpha = calculatorPR78.calc_a_alpha(self.a, self.alpha)
        self.A = calculatorPR78.calc_A(self.a_alpha, self.T, self.P)
        self.B = calculatorPR78.calc_B(self.b, self.T, self.P)

        if analytical:
            self.roots = calculatorPR78.solve_cubic_analytical(self.A, self.B)
        else:
            self.roots = calculatorPR78.solve_cubic(self.A, self.B)

        self.roots_real = self.roots[self.roots.imag == 0].real
        self.Z_liq = min(self.roots_real)  # lowest real root is liquid
        self.Z_gas = max(self.roots_real)
        self.V_liq = calculatorPR78.calc_V_liq(self.Z_liq, self.T, self.P)  # m^3/mol
        self.V_gas = calculatorPR78.calc_V_gas(self.Z_gas, self.T, self.P)

        if self.mw is not None:
            self.rho_liq = calculatorPR78.calc_rho(self.mw, self.V_liq)
            self.rho_gas = calculatorPR78.calc_rho(self.mw, self.V_gas)
        
        self.phase = calculatorPR78.identify_phase(self.roots_real, self.V_liq, self.V_gas, self.T, self.P)
        if self.phase == 'gas':
            self.V_liq = None
            self.Z_liq = None
            self.rho_liq = None
        if self.phase == 'liquid':
            self.V_gas = None
            self.Z_gas = None
            self.rho_gas = None

    def plot(self):
        x = np.linspace(0, 1.05)
        y = calculatorPR.Z_obj_func(x, self.A, self.B)
        fig, ax = plot_Z(x, y, self.roots, self.roots_real, self.T, self.P, self.eos_name)
        return fig, ax


class PRMIX:

    def __init__(self, T, P, Tcs, Pcs, omegas, zs, mws=None, analytical=True, kijs=None, _update_now=True):

        self.eos_name = 'Peng-Robinson Mixture EOS 1976'
        self.n = len(Tcs)  # number of components

        # Todo: make an Javascript Kijs matrix constructor on a documentation page
        self.kijs = kijs

        self.T = T  # K
        self.P = P
        self.Tcs = np.array(Tcs)
        self.Pcs = np.array(Pcs)
        self.omegas = np.array(omegas)
        self.zs = utilities.normalize_composition_list(zs)[0]

        self.mws = mws
        self.R = config.constants['R']  # 8.31446261815324 ((m^3-Pa)/(mol-K))

        self.Trs = calculatorPRMIX.calc_Tr(self.T, self.Tcs)
        self.ais = calculatorPRMIX.calc_a(self.Tcs, self.Pcs)
        self.b = calculatorPRMIX.calc_b(self.zs, self.Tcs, self.Pcs)
        self.kappas = calculatorPRMIX.calc_kappas(self.omegas)
        self.alphas = calculatorPRMIX.calc_alpha(self.kappas, self.Trs)
        self.ais_alphas = calculatorPR.calc_a_alpha(self.ais, self.alphas)
        self.a_alpha = calculatorPRMIX.calc_a_alpha(self.zs, self.ais_alphas)
        self.A = calculatorPRMIX.calc_A(self.a_alpha, self.T, self.P)
        self.B = calculatorPRMIX.calc_B(self.b, self.T, self.P)

        if analytical:
            self.roots = calculatorPRMIX.solve_cubic_analytical(self.A, self.B)
        else:
            self.roots = calculatorPRMIX.solve_cubic(self.A, self.B)

        self.roots_real = self.roots[self.roots.imag == 0].real
        self.Z_liq = min(self.roots_real)
        self.Z_gas = max(self.roots_real)
        self.V_liq = calculatorPRMIX.calc_V_liq(self.Z_liq, self.T, self.P)
        self.V_gas = calculatorPRMIX.calc_V_gas(self.Z_gas, self.T, self.P)

        if self.mws is not None:
            self.mw = np.sum(np.array(self.mws) * np.array(self.zs))
            self.rho_liq = calculatorPRMIX.calc_rho(self.mw, self.V_liq)
            self.rho_gas = calculatorPRMIX.calc_rho(self.mw, self.V_gas)

        self.phase = calculatorPRMIX.identify_phase(self.roots_real, self.V_liq, self.V_gas, self.T, self.P)
        if self.phase == 'gas':
            self.V_liq = None
            self.Z_liq = None
            self.rho_liq = None
        if self.phase == 'liquid':
            self.V_gas = None
            self.Z_gas = None
            self.rho_gas = None

    def plot(self):
        x = np.linspace(0, 1.05)
        y = calculatorPRMIX.Z_obj_func(x, self.A, self.B)
        fig, ax = plot_Z(x, y, self.roots, self.roots_real, self.T, self.P, self.eos_name)
        return fig, ax


class PRMIX78:
    def __init__(self, T, P, Tcs, Pcs, omegas, zs, mws=None, analytical=True, kijs=None):

        self.eos_name = 'Peng-Robinson Mixture EOS 1978'
        self.n = len(Tcs)  # number of components

        # Todo: make an Javascript Kijs matrix constructor
        self.kijs = kijs

        self.T = T  # K
        self.P = P
        self.Tcs = np.array(Tcs)
        self.Pcs = np.array(Pcs)
        self.omegas = np.array(omegas)
        self.zs = utilities.normalize_composition_list(zs)[0]

        self.mws = mws
        self.R = config.constants['R']  # 8.31446261815324 ((m^3-Pa)/(mol-K))

        self.Trs = calculatorPRMIX78.calc_Tr(self.T, self.Tcs)
        self.ais = calculatorPRMIX78.calc_a(self.Tcs, self.Pcs)
        self.bis = calculatorPRMIX78.calc_b(self.zs, self.Tcs, self.Pcs)
        self.b = calculatorPRMIX78.calc_b(self.zs, self.Tcs, self.Pcs)
        self.kappas = calculatorPRMIX78.calc_kappas(self.omegas)
        self.alphas = calculatorPRMIX78.calc_alpha(self.kappas, self.Trs)
        self.ais_alphas = calculatorPR78.calc_a_alpha(self.ais, self.alphas)
        self.a_alpha = calculatorPRMIX78.calc_a_alpha(self.zs, self.ais_alphas)
        self.A = calculatorPRMIX78.calc_A(self.a_alpha, self.T, self.P)
        self.B = calculatorPRMIX78.calc_B(self.b, self.T, self.P)

        if analytical:
            self.roots = calculatorPRMIX78.solve_cubic_analytical(self.A, self.B)
        else:
            self.roots = calculatorPRMIX78.solve_cubic(self.A, self.B)

        self.roots_real = self.roots[self.roots.imag == 0].real
        self.Z_liq = min(self.roots_real)
        self.Z_gas = max(self.roots_real)
        self.V_liq = calculatorPRMIX78.calc_V_liq(self.Z_liq, self.T, self.P)
        self.V_gas = calculatorPRMIX78.calc_V_gas(self.Z_gas, self.T, self.P)

        if self.mws is not None:
            self.mw = np.sum(np.array(self.mws) * np.array(self.zs))
            self.rho_liq = calculatorPRMIX78.calc_rho(self.mw, self.V_liq)
            self.rho_gas = calculatorPRMIX78.calc_rho(self.mw, self.V_gas)
            
        self.phase = calculatorPRMIX78.identify_phase(self.roots_real, self.V_liq, self.V_gas, self.T, self.P)
        if self.phase == 'gas':
            self.V_liq = None
            self.Z_liq = None
            self.rho_liq = None
        if self.phase == 'liquid':
            self.V_gas = None
            self.Z_gas = None
            self.rho_gas = None

    def plot(self):
        x = np.linspace(0, 1.05)
        y = calculatorPRMIX78.Z_obj_func(x, self.A, self.B)
        fig, ax = plot_Z(x, y, self.roots, self.roots_real, self.T, self.P, self.eos_name)
        return fig, ax




