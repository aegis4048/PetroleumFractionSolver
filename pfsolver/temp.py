import correlations
import pandas as pd
import numpy as np
from scipy.optimize import newton
from scipy.optimize import minimize


class SCNProperty(object):

    def __init__(self, scn, subtract='naphthenes'):

        self.scn = scn
        self.sg_liq = correlations.calc_sg_liq(scn)
        self.MW = correlations.calc_MW(scn)
        self.Tb_K = correlations.calc_Tb(scn)
        self.Tb_R = correlations.kelvin_to_rankine(self.Tb_K)
        self.Tb = self.Tb_R

        if self.MW > 200:
            self.type = 'heavy'
            raise TypeError('Computed MW=%.2f. SCN > 14.5714 (MW > 200) is not supported. The PNA composition model implemented in '
                            'this library  is intended for modeling plus fractions of natural gas samples, which '
                            'realistically never exceeds MW>200.' % self.MW)
        else:
            self.type = 'light'

        self.SUS_100 = None
        self.VGC = None
        self.VGF = None

        self.RI = correlations.calc_RI(self.Tb, self.sg_liq)
        self.RI_intercept = correlations.calc_RI_intercept(self.sg_liq, self.RI)
        self.v100, self.v210 = correlations.calc_v100_v210(self.Tb, self.sg_liq)

        self.warning = {
            'PNA': {
                'MW': False,
            }
        }

        if self.type == 'heavy':
            self.SUS_100 = correlations.calc_SUS_100(self.v100)
            self.VGC = correlations.calc_VGC(self.sg_liq, self.SUS_100)
            self.VG = self.VGC
        else:
            self.VGF = correlations.calc_VGF(self.sg_liq, self.v100)
            self.VG = self.VGF

        # forcing MW = 100 to force light fraction.
        self.xp, self.xn, self.xa = correlations.calc_PNA_comp(100, self.VG, self.RI_intercept)

        if subtract == 'naphthenes':
            self.xn = 1 - self.xp - self.xa
        elif subtract == 'paraffins':
            self.xp = 1 - self.xn - self.xa
        elif subtract == 'aromatics':
            self.xa = 1 - self.xp - self.xn
        else:
            raise ValueError('subtract must be one of "naphthenes", "paraffins", or "aromatics"')

    @classmethod
    def build_table(cls, arr):
        SCNs = arr
        sgs_liq = []
        Tbs = []
        MWs = []
        xps = []
        xns = []
        xas = []
        RIs = []
        v100s = []
        v210s = []
        for item in SCNs:
            SCN_obj = SCNProperty(scn=item)
            sgs_liq.append(SCN_obj.sg_liq)
            Tbs.append(SCN_obj.Tb)
            MWs.append(SCN_obj.MW)
            xps.append(SCN_obj.xp)
            xns.append(SCN_obj.xn)
            xas.append(SCN_obj.xa)
            RIs.append(SCN_obj.RI)
            v100s.append(SCN_obj.v100)
            v210s.append(SCN_obj.v210)

        table = pd.DataFrame.from_dict({
            'SCN': SCNs,
            'sg_liq': sgs_liq,
            'Tb [°R]': Tbs,
            'MW': MWs,
            'xp': xps,
            'xn': xns,
            'xa': xas,
            'RI': RIs,
            'v100': v100s,
            'v210': v210s
        })
        return table

    @staticmethod
    def solveForSCN(target, value):
        """
        Solve for SCN that results in the specified target variable reaching a specified value.

        Parameters:
        - target: A string specifying the target variable (e.g., 'MW').
        - value: The desired value for the target variable (e.g., 94 for MW=94).

        Returns:
        - scn: The SCN value that results in the target variable reaching the desired value.

        Raises:
        - ValueError: If the provided target is not a valid target variable.

        Example:
            solved_scn = SCNProperty.solveForSCN(target='Xa', value=0.086459)
        """

        target = target.lower()
        valid_targets = ['scn', 'sg_liq', 'mw', 'tb', 'xp', 'xn', 'xa', 'RI', 'v100', 'v210']
        if target not in valid_targets:
            raise ValueError(f"Invalid target '{target}'. Valid targets are {valid_targets}.")

        mapping = {
            'scn': 'scn',
            'sg_liq': 'sg_liq',
            'mw': 'MW',
            'tb': 'Tb',
            'xp': 'xp',
            'xn': 'xn',
            'xa': 'xa',
            'ri': 'RI',
            'v100': 'v100',
            'v210': 'v210'
        }

        def target_difference(scn):
            temp_obj = SCNProperty(scn)
            current_value = getattr(temp_obj, mapping[target])
            return current_value - value

        initial_guess_scn = 7
        solved_scn = newton(target_difference, initial_guess_scn)
        return solved_scn

scns = []
for mw in np.arange(80, 200, 5):
    solved_scn = SCNProperty.solveForSCN(target='mw', value=mw)
    scns.append(solved_scn)

print(SCNProperty.build_table(scns).to_string())
print('-------------------------------------------------------')


temp = SCNProperty(7.071429)

print(temp.v100)
print(temp.v210)
print(temp.VGF)
print(temp.RI)
print(temp.RI_intercept)

vgf = -1.816 + 3.484 * temp.sg_liq - 0.1156 * np.log(temp.v100)
vgf = -1.948 + 3.535 * temp.sg_liq - 0.1613 * np.log(temp.v210)
print(vgf)

print('-------------------------------------------------------')