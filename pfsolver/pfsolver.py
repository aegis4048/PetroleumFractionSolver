import numpy as np
import pandas as pd
import pint
from thermo import ChemicalConstantsPackage
import correlations
import warnings
from scipy.optimize import newton, minimize
import config
import inspect


UREG = pint.UnitRegistry()

MAPPING = config.GPA_table_column_mapping
CONSTANTS = config.constants

def normalize_composition(comp_dict):
    """
    :param comp_dict: un-normalized dictionary of composition. {"CH4": 3, "C2H6", 6}
    :return: normalized dictionary of composition. {"CH4": 0.3333, "C2H6", 0.66666666}
    """
    total_comp = sum(comp_dict.values())
    if total_comp > 0:
        keys = list(comp_dict.keys())
        last_key = keys[-1]
        normalized_values = [v / total_comp for v in comp_dict.values()]

        # Normalize all but the last element
        comp_dict = {keys[i]: normalized_values[i] for i in range(len(keys) - 1)}

        # Adjust the last element so that the sum is exactly 1
        comp_dict[last_key] = 1 - sum(comp_dict.values())

    return comp_dict


def is_fraction(s):
    """
    string detector for petroleum fractions. Specialized codes for fractions are triggered if detected.
    """
    substrings = ['fraction', 'fractions', 'plus', '+']
    return any(substring in s.lower() for substring in substrings)


def ideal_gas_molar_volume():
    """
    PV=nRT, where number of moles n=1. Rearranging -> V=RT/P
    R = 8.31446261815324 ((m^3-Pa)/(mol-K))
    T = 288.7056 K, 60F, standard temperature
    P = 101325 Pa, 1 atm, standard pressure
    :return: ideal gas molar volume in a standard condition (m^3/mol)
    """
    return CONSTANTS['R'] * CONSTANTS['T_STANDARD'] / CONSTANTS['P_STANDARD']


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


class PropertyTable(object):

    def __init__(self, comp_dict, summary=False, warnings=True):

        self.comp_dict = normalize_composition(comp_dict)
        self.df_GPA = pd.read_pickle("GPA 2145-16 Compound Properties Table - English.pkl")
        self.names = list(self.comp_dict.keys())
        self.zs = list(self.comp_dict.values())

        self.summary = summary
        self.warnings = warnings
        self.warning_msgs = []

        self.comp_dict_pure, self.comp_dict_fraction = self._split_known_unknown()
        self.names_pure = list(self.comp_dict_pure.keys())
        self.names_fraction = list(self.comp_dict_fraction.keys())
        self.zs_pure = list(self.comp_dict_pure.values())
        self.zs_fraction = list(self.comp_dict_fraction.values())
        self.n_pure = len(self.names_pure)
        self.n_fraction = len(self.names_fraction)

        self.target_props = ['ghv', 'sg_liq_60F', 'sg_gas_60F', 'mw']

        self.constants_pure = ChemicalConstantsPackage.constants_from_IDs(self.names_pure)
        self.check_properties_exists()

        self.V_molar = ideal_gas_molar_volume()

        self.ghvs_gas_pure, self.ghvs_liq_pure, self.sgs_liq_pure, self.sgs_gas_pure, self.mws_pure = self.get_properties_pure_compounds()
        self.ghvs_gas_fraction, self.ghvs_liq_fraction, self.sgs_fraction, self.sgs_gas_fraction, self.mws_fraction, self.scns_fraction = [], [], [], [], [], []

        # table without summary statistics line
        self.table_ = pd.DataFrame.from_dict({
            'Name': self.names_pure + self.names_fraction,
            'CAS': self.constants_pure.CASs + [None] * self.n_fraction,
            'Mole Fraction': self.zs_pure + self.zs_fraction,
            'MW': self.mws_pure.tolist() + [None] * self.n_fraction,
            'Mass Fraction': [None] * len(self.names),
            'GHV_gas': self.ghvs_gas_pure.tolist() + [None] * self.n_fraction,
            'GHV_liq': self.ghvs_liq_pure.tolist() + [None] * self.n_fraction,
            'SG_gas': self.sgs_gas_pure.tolist() + [None] * self.n_fraction,
            'SG_liq': self.sgs_liq_pure.tolist() + [None] * self.n_fraction,
        })

        self.column_mapping = {
            'Name': ['name', 'compound', 'chemical'],
            'CAS': ['cas', 'casrn', 'cas number'],
            'Mole Fraction': ['mole fraction', 'mole frac', 'mole'],
            'MW': ['mw', 'molar mass', 'molecular weight'],
            'Mass Fraction': ['mass fraction', 'mass frac', 'mass'],
            'GHV_gas': ['ghv_gas', 'ghv gas', 'gas ghv', 'vapor ghv', 'ghv_vapor', 'ghv vapor'],
            'GHV_liq': ['ghv_liq', 'ghv liq', 'liq ghv', 'liquid ghv', 'ghv_liquid', 'ghv liquid'],
            'SG_gas': ['sg_gas', 'sg gas', 'gas sg', 'vapor sg', 'sg_vapor', 'sg vapor'],
            'SG_liq': ['sg_liq', 'sg liq', 'liq sg', 'liquid sg', 'sg_liquid', 'sg liquid']
        }

        self.summary_stats_method = {
            "Name": None,
            "CAS": None,
            "Mole Fraction": 'sum',
            "MW": 'mole_frac_mean',
            "Mass Fraction": 'sum',
            "GHV_gas": 'mole_frac_mean',
            "GHV_liq": 'mole_frac_mean',
            "SG_gas": 'mass_frac_mean',
            "SG_liq": 'mass_frac_mean',
        }
        self._handle_summary()
        self._print_warnings()

    def _map_input_to_key(self, user_input):
        user_input_lower = user_input.lower()
        for key, values in self.column_mapping.items():
            if user_input_lower in values:
                return key
        return None  # Or raise an exception if preferred

    def _print_warnings(self):
        if self.warning_msgs:
            for message in self.warning_msgs:
                print(f"\033[91m{message}\033[0m")  # Print each warning message in red

    def _handle_warnings(self, working_range, custom_warning_msg, values_dict):

        if set(working_range.keys()) != set(values_dict.keys()):
            raise ValueError("The keys in working_range do not match exactly with the keys in values_dict.")

        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_back.f_code.co_name
        warning_base = f"Warning from {class_name}.{method_name}:"

        for key, value in values_dict.items():
            if key in working_range:
                min_val, max_val = working_range[key]

                # Handling None as an unbounded range
                if min_val is not None and value < min_val:
                    full_warning = f"{warning_base} {key}={value} is below the minimum working range of {min_val}."
                    self.warning_msgs.append(full_warning)
                elif max_val is not None and value > max_val:
                    full_warning = f"{warning_base} {key}={value} exceeds the maximum working range of {max_val}."
                    self.warning_msgs.append(full_warning)

        # If a custom warning message is provided, append it as a new item.
        if custom_warning_msg:
            custom_warning = f"{warning_base} {custom_warning_msg}"
            self.warning_msgs.append(custom_warning)

        self.warning_msgs.append('Set PropertyTable(warnings=False) to suppress these warnings.')
        self.warning_msgs = list(set(self.warning_msgs))  # Remove duplicates when this function is executed multiple times

    def calc_summary(self):

        summary_row = {}
        for column, operation in self.summary_stats_method.items():

            # If any NaN values are present in the column, summary stats can't be calculated
            if pd.isnull(self.table_[column]).any():
                summary_row[column] = None
                continue

            if operation == 'sum':
                summary_row[column] = self.table_[column].sum()
            elif operation == 'mean':
                summary_row[column] = self.table_[column].mean()
            elif operation == 'mole_frac_mean':
                if pd.isnull(self.table_['Mole Fraction']).any():
                    summary_row[column] = None
                else:
                    weighted_sum = (self.table_['Mole Fraction'] * self.table_[column]).sum()
                    summary_row[column] = weighted_sum / self.table_['Mole Fraction'].sum()
            elif operation == 'mass_frac_mean':
                if pd.isnull(self.table_['Mass Fraction']).any():
                    summary_row[column] = None
                else:
                    weighted_sum = (self.table_['Mass Fraction'] * self.table_[column]).sum()
                    summary_row[column] = weighted_sum / self.table_['Mass Fraction'].sum()
            else:
                summary_row[column] = None

        summary_row['Name'] = 'Total'
        summary_df = pd.DataFrame([summary_row], index=[self.n_fraction + self.n_pure])

        return summary_df

    # update directly from the user input
    def update_property(self, name, props_dict):

        if name not in self.names:
            raise ValueError("Chemical name '%s' is found in the provided composition." % name)

        row_index = self.table_[self.table_['Name'] == name].index



        for key, value in props_dict.items():
            key = self._map_input_to_key(key)
            if key is not None:
                self.table_.loc[row_index, key] = value
            #if key in self.table_.columns:
            #    self.table_.loc[row_index, key] = value
            else:
                raise ValueError(f"Column '{key}' does not exist in the DataFrame. Valid columns are {self.table_.columns}")

        # three iterations are needed to calculate column properties from left to right
        self._internal_update_property()
        self._internal_update_property()
        self._internal_update_property()
        self._print_warnings()

    def update_property_total(self):
        if "Total" not in self.table_summary['Name'].values:
            self.table_summary.loc[self.table_summary.index.max() + 1] = ["Total"] + [np.nan] * (
                        self.table_summary.shape[1] - 1)
        total_row_index = self.table_summary[self.table_summary['Name'] == "Total"].index[0]
        self.table_summary.at[total_row_index, 'Mole Fraction'] = self.table_summary['Mole Fraction'].sum()
        self.table_summary.at[total_row_index, 'Mass Fraction'] = self.table_summary.dropna(subset=['Mass Fraction'])['Mass Fraction'].sum()
        self._print_warnings()

    def calc_mass_fraction(self, row_index):
        MW_total = self.table_summary.at[self.table_summary.index.max(), 'MW']
        if not pd.isnull(MW_total):
            self.table_.at[row_index, 'Mass Fraction'] = self.table_.at[row_index, 'Mole Fraction'] * self.table_.at[row_index, 'MW'] / MW_total
        else:
            pass

    def _internal_update_property(self):
        self.table_summary = self.calc_summary()  # Update Total row first
        rules = {
            'MW': {
                'weighted_avg': {
                    'required_columns': [['Mole Fraction']],
                    'total_required': [['MW']],
                    'required_others': [[None]],
                    'funcs': [None],
                },
                'correlation': {
                    'required_columns': [['SG_gas'], ['SG_liq'], ['GHV_gas']],
                    'total_required': [[None], [None], [None]],
                    'required_others': [[None], [None], [None]],
                    'funcs': [
                        self._calc_mw_from_sg_gas,
                        self._calc_mw_from_sg_liq,
                        self._calc_mw_from_ghv_gas,
                    ],
                },
            },
            'Mass Fraction': {
                'weighted_avg': {  # notes: weighted average method needs to be removed.
                    'required_columns': [['Mass Fraction']],
                    'total_required': [['Mass Fraction']],
                    'required_others': [[None]],
                    'funcs': [None],
                },
                'correlation': {
                    'required_columns': [['Mole Fraction', 'MW']],
                    'total_required': [['MW']],
                    'required_others': [[None]],
                    'funcs': [self._calc_mass_frac_from_mole_frac_mw],
                },
            },
            'GHV_gas': {
                'weighted_avg': {
                    'required_columns': [['Mole Fraction']],
                    'total_required': [['GHV_gas']],
                    'required_others': [[None]],
                    'funcs': [None],
                },
                'correlation': {
                    'required_columns': [['MW']],
                    'total_required': [[None]],
                    'required_others': [[None]],
                    'funcs': [self._calc_GHV_gas_from_mw],
                },
            },
            'GHV_liq': {
                'weighted_avg': {
                    'required_columns': [['Mass Fraction']],
                    'total_required': [['GHV_liq']],
                    'required_others': [[None]],
                    'funcs': [None],
                },
                'correlation': {
                    'required_columns': [['GHV_gas', 'MW']],
                    'total_required': [[None]],
                    'required_others': [[None]],  # maybe self.SCNs
                    'funcs': [self._calc_GHV_liq_from_GHV_gas_MW],
                },
            },
            'SG_gas': {
                'weighted_avg': {
                    'required_columns': [['Mass Fraction']],
                    'total_required': [['SG_gas']],
                    'required_others': [[None]],
                    'funcs': [None],
                },
                'correlation': {
                    'required_columns': [['MW']],
                    'total_required': [[None]],
                    'required_others': [[None]],
                    'funcs': [self._calc_sg_gas_from_mw],
                },
            },
            'SG_liq': {
                'weighted_avg': {
                    'required_columns': [['Mass Fraction']],
                    'total_required': [['SG_liq']],
                    'required_others': [[None]],
                    'funcs': [None],
                },
                'correlation': {
                    'required_columns': [['MW']],
                    'total_required': [[False]],
                    'required_others': [[None]],
                    'funcs': [self._calc_sg_liq_from_mw],
                },
            }
        }

        for property, methods in rules.items():
            for method_name, method_details in methods.items():
                # Check if the 'weighted_avg' method can be computed or not, skip to 'correlation' if not

                #if self._handle_weighted_avg_method(property, methods.get('weighted_avg', {})):
                #    continue  # Skip to next property if weighted_avg was successful

                if method_name == 'correlation':
                    self._handle_correlation_method(property, method_details)

        self._handle_summary()

    def _handle_summary(self):
        if self.summary:
            self.table_summary = self.calc_summary()
            warnings.simplefilter(action='ignore', category=FutureWarning)
            self.table = pd.concat([self.table_, self.table_summary])
        else:
            self.table = self.table_

    def _handle_correlation_method(self, property, method_details):
        for idx, row in self.table_.iterrows():
            if pd.isna(row[property]):  # Skip rows where property is already calculated
                n_correlations = len(method_details['required_columns'])
                for i in range(n_correlations):

                    required_columns_satisfied = True
                    total_required_satisfied = True
                    required_others_satisfied = True

                    if all(method_details['required_columns'][i]):
                        required_columns_satisfied = all(pd.notna(row[col]) for col in method_details['required_columns'][i])
                    if all(method_details['total_required'][i]):
                        total_required_satisfied = all(pd.notna(self.table_summary[col].iloc[0]) for col in method_details['total_required'][i])
                    if all(method_details['required_others'][i]):
                        required_others_satisfied = True  # implement actual function later

                    if all([required_columns_satisfied, total_required_satisfied, required_others_satisfied]):

                        args = []
                        if all(method_details['required_columns'][i]):
                            args = [row[col] for col in method_details['required_columns'][i]]  # this returns KeyError: None for list containing None, so all() method is used to avoid this error.

                        args_total = []
                        if all(method_details['total_required'][i]):
                            args_total = [self.table_summary[col].iloc[0] for col in method_details['total_required'][i]]

                        args_others = []
                        if all(method_details['required_others'][i]):
                            args_others = []

                        func = method_details['funcs'][i]
                        if func == None:
                            print('property %s is not computed cuz no function' % property)
                            continue

                        calculated_value = func(*args, *args_others, *args_total)

                        if calculated_value is not None:
                            self.table_.loc[idx, property] = calculated_value
                            break  # Stop after the first successful calculation

    def _calc_GHV_gas_from_mw(self, mw):
        scn_obj = SCNProperty(mw=mw, warnings=self.warnings)
        aromatic_fraction = scn_obj.xa
        aromatic_fraction = 0
        print('----')
        print(aromatic_fraction)
        GHV_gas = newton(lambda ghv_gas: correlations.mw_ghv(mw, ghv_gas, aromatic_fraction), x0=5000, maxiter=50)
        return GHV_gas

    def _calc_sg_liq_from_mw(self, mw):
        scn_obj = SCNProperty(mw=mw, warnings=self.warnings)
        sg_liq = scn_obj.sg

        working_range = {
            'MW': (82, 698),
            'SG_liq': (0.69, 0.947),
        }
        custom_warning_msg = None
        self._handle_warnings(working_range, custom_warning_msg, {'MW': mw, 'SG_liq': sg_liq})

        return sg_liq

    def _calc_sg_gas_from_mw(self, mw):
        sg_gas = newton(lambda sg_gas: correlations.mw_sg_gas(mw, sg_gas), x0=0.65, maxiter=50)
        return sg_gas

    def _calc_mw_from_sg_gas(self, sg_gas):
        mw = newton(lambda mw: correlations.mw_sg_gas(mw, sg_gas), x0=90, maxiter=50)
        return mw

    def _calc_mw_from_sg_liq(self, sg_liq):
        scn_obj = SCNProperty(sg=sg_liq, warnings=False)
        mw = scn_obj.mw

        working_range = {
            'MW': (82, 698),
            'SG_liq': (0.69, 0.947),
        }
        custom_warning_msg = None
        self._handle_warnings(working_range, custom_warning_msg, {'MW': mw, 'SG_liq': sg_liq})

        return mw

    def _calc_mw_from_ghv_gas(self, ghv_gas):

        def objective(mw, GHV_gas):
            xa = SCNProperty(mw=mw, warnings=self.warnings).xa  # provide guess value of xa to improve model accuracy
            return abs(correlations.mw_GHV_gas_xa(mw, GHV_gas, xa))

        # initial MW guess assuming 100% paraffinic composition
        initial_mw_guess = np.array([newton(lambda mw: correlations.mw_ghv_paraffinic(mw, ghv_gas), x0=100)])
        bounds = [(0, None)]
        result = minimize(lambda mw: objective(mw[0], ghv_gas), initial_mw_guess, bounds=bounds, tol=0.01)
        mw = result.x[0]

        working_range = {
            'MW': (82, 698),
        }
        custom_warning_msg = None
        self._handle_warnings(working_range, custom_warning_msg, {'MW': mw})

        return mw

    def _calc_mass_frac_from_mole_frac_mw(self, mole_frac, mw, mw_total):
        return mole_frac * mw / mw_total

    def _calc_GHV_liq_from_GHV_gas_MW(self, ghv_gas, mw):
        return newton(lambda ghv_liq: correlations.ghv_liq_ghv_gas_mw(ghv_liq, ghv_gas, mw), x0=0.65, maxiter=50)

    def get_properties_pure_compounds(self):
        """
        :param constants: thermo's constants object
        :param GPA_data: pandas dataframe of the GPA 2145-16 Table
        :return:
        """
        ghvs_ideal_gas = []
        ghvs_liq = []
        sgs_liq = []
        sgs_gas = []
        mws = []

        for cas, name, Hc, mw, rhol_60F_mass in zip(self.constants_pure.CASs, self.constants_pure.names, self.constants_pure.Hcs, self.constants_pure.MWs, self.constants_pure.rhol_60Fs_mass):

            matching_row = self.df_GPA[self.df_GPA['CAS'] == cas]

            # chemical is found in the GPA data table
            if not matching_row.empty:
                ghv_ideal_gas = matching_row[MAPPING['ghv_gas']].iloc[0]
                ghv_liq = matching_row[MAPPING['ghv_liq']].iloc[0]
                sg_liq = matching_row[MAPPING['sg_liq_60F']].iloc[0]
                sg_gas = matching_row[MAPPING['sg_gas_60F']].iloc[0]
                mw = matching_row[MAPPING['mw']].iloc[0]

                if pd.isna(ghv_ideal_gas):
                    if Hc != 0:  # GPA table is missing GHV_gas data for some compounds: 1-propyne
                        ghv_ideal_gas = Hc / self.V_molar
                        ghv_ideal_gas = UREG('%.15f joule/m^3' % ghv_ideal_gas).to('Btu/ft^3')._magnitude * -1
                        ghv_liq = Hc * mw
                        ghv_liq = UREG('%.15f joule/g' % ghv_liq).to('Btu/lb')._magnitude * -1
                    elif Hc is None:
                        raise ValueError("Chemical name '%s' is recognized but missing a required data (%s)." % (
                        name, 'Hcs, heat of combustion [J/mol]'))
                    else:
                        ghv_ideal_gas = 0
                        ghv_liq = 0

            # chemical is NOT identified in the GPA datatable
            else:
                if Hc != 0:
                    ghv_ideal_gas = Hc / self.V_molar
                    ghv_ideal_gas = UREG('%.15f joule/m^3' % ghv_ideal_gas).to('Btu/ft^3')._magnitude * -1
                    ghv_liq = Hc * mw
                    ghv_liq = UREG('%.15f joule/g' % ghv_liq).to('Btu/lb')._magnitude * -1
                elif Hc is None:
                    raise ValueError("Chemical name '%s' is recognized but missing a required data (%s)." % (
                    name, 'Hcs, heat of combustion [J/mol]'))
                else:
                    ghv_ideal_gas = 0
                    ghv_liq = 0

                sg_liq = rhol_60F_mass / CONSTANTS['RHO_WATER']
                sg_gas = mw / CONSTANTS['MW_AIR']

            ghvs_ideal_gas.append(ghv_ideal_gas)
            ghvs_liq.append(ghv_liq)
            sgs_liq.append(sg_liq)
            sgs_gas.append(sg_gas)
            mws.append(mw)

        return np.array(ghvs_ideal_gas), np.array(ghvs_liq), np.array(sgs_liq), np.array(sgs_gas), np.array(mws)

    def _split_known_unknown(self):
        comp_dict_pure = {}
        comp_dict_fraction = {}
        for key, value in self.comp_dict.items():
            if is_fraction(key):
                comp_dict_fraction[key] = value
            else:
                comp_dict_pure[key] = value
        return comp_dict_pure, comp_dict_fraction

    def check_properties_exists(self):
        """
        :param constants: constants object of the thermo library
        Molecular weight and normal boiling points are minimum information needed to characterize a fluid, incase of any
        missing properties. For example, docosane is missing Heat of combustion (J/mol), but it can be correlated.
        """
        rhol_60Fs_mass = self.constants_pure.rhol_60Fs_mass
        mws = self.constants_pure.MWs
        names = self.constants_pure.names
        Hcs = self.constants_pure.Hcs

        for rhol_60F_mass, mw, Hc, name in zip(rhol_60Fs_mass, mws, Hcs, names):
            if 'sg_liq_60F' in self.target_props or 'sg_gas_60F' in self.target_props:
                if rhol_60F_mass is None:
                    raise ValueError("Chemical name '%s' is recognized but missing a required data (%s)." % (name, 'rhol_60Fs_mass, liquid mass density at 60F'))
            if 'mw' in self.target_props:
                if mw is None:
                    raise ValueError("Chemical name '%s' is recognized but missing a required data (%s)." % (name, 'MWs, molecular weight [g/mol]'))
            if 'ghv' in self.target_props:
                if Hc is None:
                    raise ValueError("Chemical name '%s' is recognized but missing a required data (%s)." % (name, 'Hcs, heat of combustion [J/mol]'))


df = SCNProperty.build_table([i for i in range(82, 94, 1)])
print(df.to_string())
print('----------------------------------------------------------')


brazos_gas = dict([
    ('hydrogen sulfide', 0.001),
    ('nitrogen', 2.304),
    ('carbon dioxide', 1.505),
    ('methane', 71.432),
    ('ethane', 11.732),
    ('propane', 7.595),
    ('isobutane', 0.827),
    ('n-butane', 2.540),
    ('i-pentane', 0.578),
    ('n-pentane', 0.597),
    ('Hexanes+', 0.889),
    #('Heptanes+', 0.4),
])

brazos_cond = dict([
    ('nitrogen', 0),
    ('carbon dioxide', 0.007),
    ('methane', 0.052),
    ('ethane', 0.495),
    ('propane', 3.985),
    ('isobutane', 1.935),
    ('n-butane', 11.854),
    ('i-pentane', 8.710),
    ('n-pentane', 12.783),
    ('Hexanes+', 60.179),
    #('Heptanes+', 0.4),
])

ptable = PropertyTable(brazos_gas, summary=True, warnings=False)
#ptable.update_property('Hexanes+', {'GHV_liq': 20408})
#ptable.update_property('Hexanes+', {'GHV_gas': 4849})
#ptable.update_property('Hexanes+', {'liquid sg': 0.7142})
ptable.update_property('Hexanes+', {'mw': 90.161})


#ptable.update_property('Hexanes+', {'MW':90.161, 'GHV_gas': 5000})
#ptable.update_property('Hexanes+', {'MW': 90.161, 'GHV_gas': 4849})
#ptable.update_property('Heptanes+', {'MW': 100.5, 'GHV_gas': 6000})

#ptable.update_property('Hexanes+', {'GHV_liq': 20677})  # Brazos condensates
#ptable.update_property('Hexanes+', {'GHV_liq': 20888, 'MW': 93.189})  # Brazos condensates
#ptable.update_property('Hexanes+', {'GHV_gas': 5129})  # 6:3:1 Hexanes+
#ptable.update_property('Hexanes+', {'GHV_gas': 4849})  # extended Hexanes+
#ptable.update_property('Hexanes+', {'GHV_gas': 4774})  # extended Hexanes+ for Shamrock

#ptable.update_property_total({'MW': 81.5})

#ptable.update_property('Hexanes+', {'MW': 90.161, 'GHV_gas': 4849})  # Brazos gas
print(ptable.table.to_string())

