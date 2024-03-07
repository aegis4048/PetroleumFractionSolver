import numpy as np
import correlations
from scipy.optimize import minimize, newton
import warnings


class SCNProperty(object):

    def __init__(self, sg=None, mw=None, Tb=None, xa=None, xn=None, xp=None, PNA=True, subtract='naphthenes', warnings=True):

        # Tb in rankine
        self.sg = None
        self.mw = None
        self.Tb = None
        self.xa = None
        self.xn = None
        self.xp = None
        self._SUS_100 = None
        self._VGC = None
        self._VGF = None
        self.ri = None
        self._RI_intercept = None
        self.v100 = None
        self.v210 = None
        self._subtract = subtract

        self._attributes = {
            'sg': sg,
            'mw': mw,
            'Tb': Tb,
        }

        self._correlations = {
            correlations.Tb_mw_model_SCN: ['Tb', 'mw'],
            correlations.sg_liq_mw_model_SCN: ['sg', 'mw'],
        }
        self.resolve_dependencies()
        self.assign_attributes()

        if PNA:
            self.xp, self.xn, self.xa = self.calc_PNA_composition()

        # round to n significant figures
        n = 4
        self._attributes = {key: float(f"{value:.{n}g}") if isinstance(value, float) else value for key, value in self._attributes.items()}
        self._round_attributes(n)

        self._range_warnings = {
            'sg': (0.69, 0.947),
            'mw': (82, 698),
            'Tb': (606.6, 1531.8)
        }
        if warnings:
            self._check_ranges()

    def _check_ranges(self):
        for attr, (min_val, max_val) in self._range_warnings.items():
            value = getattr(self, attr, None)
            if value is not None and not (min_val <= value <= max_val):
                warnings.warn(f"{attr} value {value} is out of working range [{min_val}, {max_val}]. Set warnings=False to suppress this warning.")

    @classmethod
    def build_table(cls, arr, col='mw', PNA=True):
        col = SCNProperty._target_str_mapping(col)
        SCN_dicts = []
        for item in arr:
            SCN_obj = SCNProperty(**{col: item}, PNA=PNA)
            SCN_dict = {key: value for key, value in vars(SCN_obj).items() if not key.startswith('_')}
            SCN_dicts.append(SCN_dict)

        return pd.DataFrame(SCN_dicts)

    @staticmethod
    def _target_str_mapping(target_str):

        target = target_str.lower()
        valid_targets = ['sg', 'mw', 'tb', 'xp', 'xn', 'xa', 'ri', 'v100', 'v210']
        if target not in valid_targets:
            raise ValueError(f"Invalid target '{target}'. Valid targets are {valid_targets}.")

        mapping = {
            'sg': 'sg',
            'mw': 'mw',
            'tb': 'Tb',
            'xp': 'xp',
            'xn': 'xn',
            'xa': 'xa',
            'ri': 'ri',
            'v100': 'v100',
            'v210': 'v210'
        }
        return mapping[target]

    def _round_attributes(self, n):
        for attr in vars(self):
            value = getattr(self, attr)
            if isinstance(value, float):
                setattr(self, attr, float(f"{value:.{n}g}"))

    def calc_PNA_composition(self):

        self.ri = correlations.calc_RI(self.Tb, self.sg)
        self._RI_intercept = correlations.calc_RI_intercept(self.sg, self.ri)
        self.v100, self.v210 = correlations.calc_v100_v210(self.Tb, self.sg)

        if self.mw > 200:
            self._SUS_100 = correlations.calc_SUS_100(self.v100)
            self._VGC = correlations.calc_VGC(self.sg, self._SUS_100)
            self._VG = self._VGC
        else:
            self._VGF = correlations.calc_VGF(self.sg, self.v100)
            self._VG = self._VGF

        xp, xn, xa = correlations.calc_PNA_comp(self.mw, self._VG, self._RI_intercept)

        if self._subtract == 'naphthenes':
            self.xn = 1 - xp - xa
        elif self._subtract == 'paraffins':
            self.xp = 1 - xn - xa
        elif self._subtract == 'aromatics':
            self.xa = 1 - xp - xn
        else:
            raise ValueError('subtract must be one of "naphthenes", "paraffins", or "aromatics"')

        return xp, xn, xa

    def assign_attributes(self):
        for key, value in self._attributes.items():
            setattr(self, key, value)

    def resolve_dependencies(self):
        resolved = set([attr for attr, value in self._attributes.items() if value is not None])

        # Resolve other dependencies
        while len(resolved) < len(self._attributes):
            resolved_this_iteration = False
            for correlation_func, variables in self._correlations.items():

                unresolved_vars = [var for var in variables if var not in resolved]
                if len(unresolved_vars) == 1:
                    unresolved_var = unresolved_vars[0]
                    resolved_vars = [self._attributes[var] for var in variables if var in resolved]
                    try:
                        self._attributes[unresolved_var] = newton(lambda x: correlation_func(*self.prepare_args(correlation_func, x, resolved_vars)), x0=self.get_initial_guess(unresolved_var))
                        resolved.add(unresolved_var)
                        resolved_this_iteration = True
                    except RuntimeError as e:
                        print("Error in calculating {}: {}".format(unresolved_var, e))
                        return

            if not resolved_this_iteration:
                break

    def prepare_args(self, correlation_func, x, resolved_vars):
        arg_order = self._correlations[correlation_func]
        args = []
        for arg in arg_order:
            if arg in self._attributes and self._attributes[arg] is not None:
                args.append(self._attributes[arg])
            else:
                args.append(x)
        return args

    def get_initial_guess(self, variable):
        initial_guesses = {'mw': 100, 'sg': 0.8, 'Tb': 600}
        return initial_guesses.get(variable, 1.0)


def objective(mw, GHV_gas):
    xa = SCNProperty(mw=mw).xa  # provide guess value of xa to improve model accuracy
    return abs(correlations.mw_GHV_gas_xa(mw, GHV_gas, xa))

# Assuming GHV_gas is given
ghv_gas = 4774
mw_guess = newton(lambda mw: correlations.mw_ghv_paraffinic(mw, ghv_gas), x0=100)  # initial MW guess assuming 100% paraffinic composition

# Optimization setup
initial_guess = np.array([mw_guess])
bounds = [(0, None)]

# Perform the optimization
result = minimize(lambda mw: objective(mw[0], ghv_gas), initial_guess, bounds=bounds, tol=0.01)


if result.success:
    mw_solved = result.x[0]
    xa_solved = SCNProperty(mw=mw_solved).xa
    print(f"Solved mw: {mw_solved}, Corresponding xa: {xa_solved}")
else:
    print(f"Solution not found: {result.message}")
