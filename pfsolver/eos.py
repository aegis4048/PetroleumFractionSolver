import numpy as np
import pint
from thermo import ChemicalConstantsPackage
from matplotlib import pyplot as plt

from pfsolver import utilities
from pfsolver import config


UREG = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)


class PR(object):

    def __init__(self, T, P, Tc, Pc, omega, mw=None, _update_now=True):

        self.eos_name = 'Peng-Robinson EOS 1976'

        self.T = T  # K
        self.P = P  # pascal
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega

        self.mw = mw
        self.R = config.constants['R']  # 8.31446261815324 ((m^3-Pa)/(mol-K))
        self.Tr = self.T / self.Tc

        self.a = None
        self.b = None
        self.kappa = None
        self.alpha = None
        self.a_alpha = None
        self.A = None
        self.B = None
        self.roots = None
        self.roots_real = None
        self.Z_liq = None  # lowest real root is liquid
        self.Z_gas = None
        self.V_liq = None  # m^3/mol
        self.V_gas = None
        self.rho_liq = None
        self.rho_gas = None

        # this line is implemented to avoid duplicate calculations when PR class is inherited with __super__ in
        # other classes
        if _update_now:
            self.initialize_properties()

    def calc_a(self):
        return 0.45724 * (self.R ** 2) * (self.Tc ** 2) / self.Pc

    def calc_b(self):
        return 0.07780 * self.R * self.Tc / self.Pc

    def calc_kappa(self):
        return 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2

    def calc_alpha(self):
        return (1 + self.kappa * (1 - np.sqrt(self.Tr))) ** 2

    def calc_a_alpha(self):
        return self.a * self.alpha

    def calc_A(self):
        return self.a_alpha * self.P / (self.R ** 2 * self.T ** 2)

    def calc_B(self):
        return self.b * self.P / (self.R * self.T)

    def calc_V_liq(self):
        return self.Z_liq * self.R * self.T / self.P  # m^3/mol

    def calc_V_gas(self):
        return self.Z_gas * self.R * self.T / self.P  # m^3/mol

    def calc_rho_liq(self):
        return self.P / (self.R * self.T * self.Z_liq) * self.mw  # g/m^3

    def calc_rho_gas(self):
        return self.P / (self.R * self.T * self.Z_gas) * self.mw  # g/m^3

    def solve_cubic(self):
        # Coefficients of the cubic equation
        coeffs = [1, -(1 - self.B), self.A - 3 * self.B ** 2 - 2 * self.B,
                  -(self.A * self.B - self.B ** 2 - self.B ** 3)]
        return np.roots(coeffs)

    def Z_obj_func(self, Z):
        return Z**3 - (1 - self.B) * Z**2 + (self.A - 2*self.B - 3*self.B**2) * Z - (
                self.A * self.B - self.B**2 - self.B**3)

    def initialize_properties(self):
        self.a = self.calc_a()
        self.b = self.calc_b()
        self.kappa = self.calc_kappa()
        print('kappa orig:', self.kappa)
        self.alpha = self.calc_alpha()
        self.a_alpha = self.calc_a_alpha()
        self.A = self.calc_A()
        self.B = self.calc_B()
        self.roots = self.solve_cubic()
        self.roots_real = self.roots.real
        self.Z_liq = min(self.roots_real)  # lowest real root is liquid
        self.Z_gas = max(self.roots_real)
        self.V_liq = self.calc_V_liq()  # m^3/mol

        if self.mw is not None:
            self.rho_liq = self.calc_rho_liq()
            self.rho_gas = self.calc_rho_gas()

    def plot(self):

        x = np.linspace(0, 1.05)
        y = self.Z_obj_func(x)

        fig, ax = plt.subplots(figsize=(8, 4.5))

        ax.plot(x, y, label='f(Z)', color='k')
        ax.axhline(0, color='r', label='y=0')
        for i in range(len(self.roots)):
            real_part = self.roots[i].real
            imag_part = self.roots[i].imag

            if np.abs(imag_part) < 1e-5:
                label = 'root = %.3f' % real_part
            else:
                sign = '+' if imag_part > 0 else '-'
                label = f"root(i) = {real_part:.3f}{sign}j{np.abs(imag_part):.3f}"

            ax.axvline(self.roots_real[i], color='b', linestyle='--', label=label)

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

        def setbold(txt):
            return ' '.join([r"$\bf{" + item + "}$" for item in txt.split(' ')])

        # Todo: Think about how to implement unit management
        T_F = round(UREG('%.15f kelvin' % self.T).to('degF')._magnitude, 1)
        P_psia = round(UREG('%.15f pascal' % self.P).to('psi')._magnitude, 1)

        bold_txt = setbold('Roots of the Cubic Equation')
        plain_txt = r', %s @%sF, %spsia' % (self.eos_name, T_F, P_psia)
        fig.suptitle(bold_txt + plain_txt, verticalalignment='top', x=0.01, horizontalalignment='left', fontsize=13,
                     y=0.96)
        yloc = 0.895
        ax.annotate('', xy=(0.01, yloc + 0.01), xycoords='figure fraction', xytext=(1.02, yloc + 0.01),
                    arrowprops=dict(arrowstyle="-", color='k', lw=0.7))

        fig.tight_layout()
        return fig, ax


class PR78(PR):
    def __init__(self, T, P, Tc, Pc, omega, mw=None):
        super().__init__(T, P, Tc, Pc, omega, mw, _update_now=False)

        self.eos_name = 'Peng-Robinson EOS 1978'
        self.kappa = self.calc_kappa()
        self.initialize_properties()

    def calc_kappa(self):
        if self.omega <= 0.491:
            return 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2
        else:
            return 0.379642 + 1.48503 * self.omega - 0.164423 * self.omega ** 2 + 0.016666 * self.omega ** 3




