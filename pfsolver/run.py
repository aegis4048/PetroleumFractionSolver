import correlations
import pandas as pd
import numpy as np
from scipy.optimize import newton


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

print(SCNProperty.build_table(np.arange(6, 15, 1)).to_string())
print('-------------------------------------------------------')

# this method works, assuming that there's no BTEX composition

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Molecular weights of the compounds
mws = np.array([86.18, 100.21, 114.23])  # n-Hexane, n-Heptane, n-Octane
mws = np.array([86.18, 100.21, 114.23, 92.14])  # n-Hexane, n-Heptane, n-Octane, Toluene (aromatics)
mws = np.array([86.18, 100.21, 114.23, 128, 92.14])  # n-Hexane, n-Heptane, n-Octane, n-Nonane, aromatics (aromatics)

# Specific gravities of the compounds (dummy values, replace with actual values)
sgs = np.array([0.659, 0.684, 0.70])  # n-Hexane, n-Heptane, n-Octane
sgs = np.array([0.659, 0.684, 0.70, 0.866])  # n-Hexane, n-Heptane, n-Octane, Toluene (BTEX avg)
sgs = np.array([0.659, 0.684, 0.70, 0.718, 0.866])  # n-Hexane, n-Heptane, n-Octane, n-Nonane, aromatics (BTEX avg)


def calculate_liquid_sg(x):
    # Simplified liquid SG calculation based on mole fractions and specific gravities
    # Replace this with your actual SG calculation
    return np.dot(x, sgs) / np.sum(x)


def optimize_mixture(target_mw):
    # Objective function to minimize: the difference between target and calculated MW
    def objective(x):
        return np.abs(target_mw - np.dot(x, mws) / np.sum(x))

    solved_scn = SCNProperty.solveForSCN(target='mw', value=target_mw)

    mole_frac_aromatics = 1 - SCNProperty(scn=solved_scn).xa
    # Constraints: sum of mole fractions should be 1
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - mole_frac_aromatics})

    # Initial guess (normalized ratios)
    # initial_ratios = np.array([2, 1, 0.5])
    # initial_ratios = np.array([2, 1, 0.5, 0.5])
    initial_ratios = np.array([2, 1, 0.5, 0.25, 0.5])
    x0 = initial_ratios / np.sum(initial_ratios)

    # Bounds for each variable: between 0 and 1
    bounds = [(0, 1) for _ in range(len(mws))]

    # Minimize the objective function
    result = minimize(objective, x0, bounds=bounds, constraints=cons)

    # Calculate liquid SG based on the optimized mole fractions
    liquid_sg = calculate_liquid_sg(result.x)

    # Return the optimized mole fractions and liquid SG
    return result.x, liquid_sg


# Prepare dataframe
data = []
for target_mw in np.arange(85, 121, 1):
    mole_fractions, liquid_sg = optimize_mixture(target_mw)

    mole_fractions = np.round(mole_fractions * 100, 2)

    data.append([target_mw, liquid_sg] + list(mole_fractions))

columns = ['MW', 'Liquid SG'] + [f'{compound} [%]' for compound in ['C6', 'C7', 'C8']]
columns = ['MW', 'Liquid SG'] + [f'{compound} [%]' for compound in ['C6', 'C7', 'C8', 'Aromatics']]
columns = ['MW', 'Liquid SG'] + [f'{compound} [%]' for compound in ['C6', 'C7', 'C8', 'C9', 'Aromatics']]
df = pd.DataFrame(data, columns=columns)

print(df.to_string())