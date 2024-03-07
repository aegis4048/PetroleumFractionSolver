import correlations
import pandas as pd
import numpy as np
from scipy.optimize import newton
from scipy.optimize import minimize


class SCNProperty(object):

    def __init__(self, scn, subtract='naphthenes', PNA=True):

        self.scn = scn
        self.sg = correlations.calc_sg_liq(scn)
        self.MW = correlations.calc_MW(scn)
        self.Tb_K = correlations.calc_Tb(scn)
        self.Tb_R = correlations.kelvin_to_rankine(self.Tb_K)
        self.Tb = self.Tb_R
        self.PNA = PNA

        if self.MW > 200:
            self.type = 'heavy'
            """
            raise TypeError('Computed MW=%.2f. SCN > 14.5714 (MW > 200) is not supported. The PNA composition model implemented in '
                'this library  is intended for modeling plus fractions of natural gas samples, which '
                'realistically never exceeds MW>200.' % self.MW)
            """

        else:
            self.type = 'light'

        if PNA is True:
            self.SUS_100 = None
            self.VGC = None
            self.VGF = None

            self.RI = correlations.calc_RI(self.Tb, self.sg)
            self.RI_intercept = correlations.calc_RI_intercept(self.sg, self.RI)
            self.v100, self.v210 = correlations.calc_v100_v210(self.Tb, self.sg)

            self.warning = {
                'PNA': {
                    'MW': False,
                }
            }

            if self.type == 'heavy':
                self.SUS_100 = correlations.calc_SUS_100(self.v100)
                self.VGC = correlations.calc_VGC(self.sg, self.SUS_100)
                self.VG = self.VGC
            else:
                self.VGF = correlations.calc_VGF(self.sg, self.v100)
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
    def build_table(cls, arr, PNA=True):
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
            SCN_obj = SCNProperty(scn=item, PNA=PNA)
            sgs_liq.append(SCN_obj.sg)
            Tbs.append(SCN_obj.Tb)
            MWs.append(SCN_obj.MW)
            if SCN_obj.PNA:
                xps.append(SCN_obj.xp)
                xns.append(SCN_obj.xn)
                xas.append(SCN_obj.xa)
                RIs.append(SCN_obj.RI)
                v100s.append(SCN_obj.v100)
                v210s.append(SCN_obj.v210)

        table = pd.DataFrame.from_dict({
            'SCN': SCNs,
            'MW': MWs,
            'sg': sgs_liq,
            'Tb [°R]': Tbs
        })
        if PNA:
            table['xp'] = xps
            table['xn'] = xns
            table['xa'] = xas
            table['RI'] = RIs
            table['v100'] = v100s
            table['v210'] = v210s

        return table

    @staticmethod
    def calc_property(target, input_key, input_val):

        target = SCNProperty._target_str_mapping(target)
        input_key = SCNProperty._target_str_mapping(input_key)

        scn = SCNProperty.solveForSCN(target=input_key, value=input_val)
        scn_obj = SCNProperty(scn)
        return getattr(scn_obj, target)

    @staticmethod
    def _target_str_mapping(target_str):

        target = target_str.lower()
        valid_targets = ['scn', 'sg', 'mw', 'tb', 'xp', 'xn', 'xa', 'RI', 'v100', 'v210']
        if target not in valid_targets:
            raise ValueError(f"Invalid target '{target}'. Valid targets are {valid_targets}.")

        mapping = {
            'scn': 'scn',
            'sg': 'sg',
            'mw': 'MW',
            'tb': 'Tb',
            'xp': 'xp',
            'xn': 'xn',
            'xa': 'xa',
            'ri': 'RI',
            'v100': 'v100',
            'v210': 'v210'
        }
        return mapping[target]

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
        target = SCNProperty._target_str_mapping(target)
        
        def target_difference(scn):
            temp_obj = SCNProperty(scn)
            current_value = getattr(temp_obj, target)
            return current_value - value

        initial_guess_scn = 7
        solved_scn = newton(target_difference, initial_guess_scn)
        return solved_scn

print(SCNProperty.build_table(np.arange(6, 15, 1)).to_string())
print('-------------------------------------------------------')

scns = []
for mw in np.arange(80, 300, 5):
    solved_scn = SCNProperty.solveForSCN(target='mw', value=mw)
    scns.append(solved_scn)

print(SCNProperty.build_table(scns, PNA=False).to_string())
print('-------------------------------------------------------')

print(SCNProperty.calc_property(target='sg', input_key='MW', input_val=100))
print(SCNProperty.calc_property(target='tb', input_key='MW', input_val=100))