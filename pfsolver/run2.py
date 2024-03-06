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


#


def calculate_liquid_sg(x):
    # Simplified liquid SG calculation based on mole fractions and specific gravities
    # Replace this with your actual SG calculation
    return np.dot(x, sgs) / np.sum(x)


def optimize_mixture(target_mw, mws):

    # Objective function to minimize: the difference between target and calculated MW
    def objective(x, xa, target_mw, mws_p, mw_a):
        return np.abs(target_mw - np.dot(x, mws_p) - xa*mw_a)

    solved_scn = SCNProperty.solveForSCN(target='mw', value=target_mw)
    xa = SCNProperty(scn=solved_scn).xa

    n = len(mws) - 1
    initial_ratios = 2 * 0.5 ** np.arange(n)
    x0 = initial_ratios / np.sum(initial_ratios)

    # Bounds for each variable: between 0 and 1
    bounds = [(0, 1) for _ in range(len(mws[:-1]))]
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - (1-xa)})

    # Minimize the objective function
    result = minimize(objective, x0, args=(xa, target_mw, mws[:-1], mws[-1]), constraints=cons, bounds=bounds)

    # Return the optimized mole fractions and liquid SG
    return list(result.x) + [xa]


print(SCNProperty.build_table(np.arange(6, 15, 1)).to_string())
print('-------------------------------------------------------')

scns = []
for mw in np.arange(80, 200, 5):
    solved_scn = SCNProperty.solveForSCN(target='mw', value=mw)
    scns.append(solved_scn)

print(SCNProperty.build_table(scns).to_string())
print('-------------------------------------------------------')


x_C6 = 0.422
x_C7 = 0.361
x_C8 = 0.076
x_C9 = 0.023
x_C10 = 0.007

print(x_C6 / 0.889)
print(x_C7 / 0.889)
print(x_C8 / 0.889)
print(x_C9 / 0.889)
print(x_C10 / 0.889)


# Molecular weights and specific gravities of the compounds
# n-Hexane, n-Heptane, n-Octane, n-Nonane, aromatics (BTEX avg)
mws = np.array([86.18, 100.21, 114.23, 128.2, 78])
sgs = np.array([0.659, 0.684, 0.70, 0.718, 0.8786])

# Benzene MW = 78.11, sg = 0.8786
# Toluene MW = 92.14, sg = 0.866
# 65% Benzene and 35% Toluene mixture MW = 83

# the liquid gravity is under-predicted because naphthenes have higher sg at low MW.

# Todo:
# For naphthene fraction, assume a pseudo-naphthenic-compound composed of 60% C6H12, 30% C7H14, 10% C8H16 to get common sg and MW.
# For aromatic fraction, assume a pseudo-aromatic-compound composed of 60% Benzene and 40% Toluene to get common sg and MW.
# Write an article for SCN and PNA composition. I need a summary of this section to move on to naphthene and aromatic fraction for sg_liq calculation

# For C6, cyclohexane             , according to Fesco GLYCALC format
# For C7, methylcyclohexane       , according to Fesco GLYCALC format
# For C8, Trimethylcyclopentane   , pick a random one. There's no definite ones, there are so many of them. But this shouldn't matter as they compose negligible fraction of the mixture.

# split my article into two: 1) SG Liq correlation from MW

# The Brazos liquid condensate sample analysis assume 60% 30% 10% fraction. The SG and MW match exactly.


def calc_sg_mixture(mole_fractions, sgs):
    masses = mole_fractions * sgs
    volumes = masses / sgs
    total_mass = np.sum(masses)
    total_volume = np.sum(volumes)
    mixture_sg = total_mass / total_volume
    return mixture_sg

# Example usage
data = []
for target_mw in np.arange(86, 102, 1):
    mole_fractions = optimize_mixture(target_mw, mws)
    sg_liq = calc_sg_mixture(mole_fractions, sgs)
    data.append([target_mw] + [sg_liq] + list(mole_fractions))

#columns = ['MW', 'Liquid SG'] + [f'{compound} [%]' for compound in ['Hexane', 'Heptane', 'Octane', 'Nonane', 'Aromatics']]
columns = ['MW', 'Liquid SG'] + [f'{compound} [%]' for compound in ['Hexane', 'Heptane', 'Octane', 'Nonane', 'Aromatics']]
df = pd.DataFrame(data, columns=columns)
#df['Total'] = df.apply(lambda x: x.iloc[1] + x.iloc[2] + x.iloc[3] + x.iloc[4] + x.iloc[5], axis=1)
df['Total'] = df.apply(lambda x: x.iloc[2] + x.iloc[3] + x.iloc[4] + x.iloc[5] + x.iloc[6], axis=1)
print(np.dot(mws, mole_fractions))
print(df.to_string(index=False))

