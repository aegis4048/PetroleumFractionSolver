import numpy as np
import pandas as pd
import pint
from thermo import ChemicalConstantsPackage
import warnings
from scipy.optimize import newton, minimize
import math
import copy
import os
import inspect

from . import correlations
from .customExceptions import SCNPropertyWarning, PropertyTableWarning
from . import config
from . import utilities


UREG = pint.UnitRegistry()
MAPPING = config.GPA_table_column_mapping
CONSTANTS = config.constants


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FEATHER_PATH = os.path.join(DIR_PATH, "GPA 2145-16 Compound Properties Table - English.feather")


def is_fraction(s):
    """
    string detector for petroleum fractions. Specialized codes for fractions are triggered if detected.
    """
    substrings = ['fraction', 'fractions', 'plus', '+']
    return any(substring in s.lower() for substring in substrings)


class SCNProperty(object):

    def __init__(
            self,
            sg=None,
            mw=None,
            Tb=None,  # Tb in Rankine
            model='kf',
            init_guess=None,
            subtract='naphthenes',
            warning=True,
            round=None,
    ):

        if sg is None and mw is None and Tb is None:
            raise ValueError("At least one of sg, mw, or Tb must be provided. From: {}".format(self.__class__.__name__))

        self.warning = warning
        config.update_settings({'warning': warning})

        self.model = model
        self.subtract = subtract
        self.kwargs_options = {
            'model': ['kf', 'ra'],  # Katz & Firoozabadi (1978), Riazi & Al-Sahhaf (1996)
            'subtract': ['naphthenes', 'paraffins', 'aromatics'],
        }
        self._validate_kwargs()

        default_guess = {'mw': 100, 'sg': 0.8, 'Tb': 600}
        self.init_guess = self._validate_return_kwargs_dict(default_guess, init_guess)

        # this _range_warnings structure needs to be preserved to pass unittest
        tol = 0.01
        if self.model == 'kf':
            self._range_warnings = {
                'sg': (0.685 - 0.685 * tol, 0.937 + 0.937 * tol),
                'mw': (84 - 84 * tol, 626 + 626 * tol),
                'Tb': (606.7 - 606.7 * tol, 1486.7 + 1486.7 * tol)
            }
        else:
            self._range_warnings = {
                'sg': (0.690 - 0.690 * tol, 0.947 + 0.947 * tol),
                'mw': (82 - 82 * tol, 698 + 698 * tol),
                'Tb': (606.6 - 606.6 * tol, 1531.8 + 1531.8 * tol)
            }

        # Tb in rankine
        self.sg = None
        self.mw = None
        self.Tb = None
        self.xp = None
        self.xn = None
        self.xa = None
        self.SUS_100 = None
        self.VGC = None
        self.VGF = None
        self.ri = None
        self.RI_intercept = None
        self.v100 = None
        self.v210 = None

        self._attributes = {
            'sg': sg,
            'mw': mw,
            'Tb': Tb,
        }

        if self.model == 'kf':
            func_Tb_mw = correlations.Tb_mw_model_SCN_KF
            func_sg_mw = correlations.sg_mw_model_SCN_KF
        else:
            func_Tb_mw = correlations.Tb_mw_model_SCN_RA
            func_sg_mw = correlations.sg_mw_model_SCN_RA

        self._correlations = {
            func_Tb_mw: ['Tb', 'mw'],
            func_sg_mw: ['sg', 'mw'],
        }
        self._resolve_dependencies()
        self._assign_attributes()

        self.xp, self.xn, self.xa = self._calc_PNA_composition()

        # round to n significant figures
        if round is not None:
            self._attributes = {key: float(f"{value:.{round}g}") if isinstance(value, float) else value for key, value in self._attributes.items()}
            self._round_attributes(round)

        if self.warning:
            properties_to_check = {attr: getattr(self, attr) for attr in list(self._range_warnings.keys())}
            utilities.check_ranges(
                target_dict=properties_to_check,
                ref_dict=self._range_warnings,
                class_name=self.__class__.__name__,
                warning_obj=SCNPropertyWarning
            )

    def _validate_kwargs(self):
        for model_name, model_choices in self.kwargs_options.items():
            user_choice = getattr(self, model_name)
            if user_choice is None or user_choice not in model_choices:
                user_choice_repr = f"'{user_choice}'" if isinstance(user_choice, str) else user_choice
                raise ValueError(
                    f"Invalid {model_name}: {user_choice_repr}. Valid options: {model_choices}. From: {self.__class__.__name__}")

    def _validate_return_kwargs_dict(self, default_guess, init_guess):
        if init_guess is not None:
            if not isinstance(init_guess, dict):
                raise ValueError(f"init_guess must be a dictionary. From: {self.__class__.__name__}")
            invalid_keys = [key for key in init_guess if key not in default_guess]
            if invalid_keys:
                raise ValueError(f"Invalid keys in init_guess: {invalid_keys}. Valid options: {list(default_guess.keys())}.")
            init_guess = {**default_guess, **init_guess}  # Merge default with user-provided, prioritizing user values
        else:
            init_guess = default_guess
        return init_guess

    @classmethod
    def build_table(cls, arr, col='mw', output_keys=None, **kwargs):
        available_keys = ['sg', 'mw', 'Tb', 'xp', 'xn', 'xa', 'ri', 'v100', 'v210', 'SUS_100', 'VGC', 'VGF', 'RI_intercept']
        available_input_keys = ['sg', 'mw', 'Tb']

        for key in available_input_keys:
            kwargs.pop(key, None)  # prevents duplicate inputs

        if output_keys is None:
            output_keys = ['sg', 'mw', 'Tb', 'xp', 'xn', 'xa', 'ri', 'v100', 'v210']

        if not all(key in available_keys for key in output_keys):
            invalid_keys = [key for key in output_keys if key not in available_keys]
            raise ValueError(f"Invalid keys in output_keys: {invalid_keys}. Valid options: {available_keys}.")

        if col not in available_input_keys:
            raise ValueError(f"Invalid col specified: {col}. Valid options: {available_input_keys}.")

        if col in output_keys:
            output_keys.remove(col)

        SCN_dicts = []
        for item in arr:
            SCN_obj = SCNProperty(**{col: item}, **kwargs)
            SCN_dicts.append({key: getattr(SCN_obj, key) for key in [col] + output_keys})

        return_df = pd.DataFrame(SCN_dicts)

        return return_df

    # round to n significant figures
    def _round_attributes(self, n):
        for attr in vars(self):
            value = getattr(self, attr)
            if isinstance(value, float):
                setattr(self, attr, float(f"{value:.{n}g}"))

    def _calc_PNA_composition(self):
        self.ri = correlations.calc_RI(self.Tb, self.sg)
        self.RI_intercept = correlations.calc_RI_intercept(self.sg, self.ri)
        self.v100, self.v210 = correlations.calc_v100_v210(self.Tb, self.sg)

        if utilities.handle_rounding_error(self.mw, 'mw') > 200:  # decimal points for floating point errors
            self.SUS_100 = correlations.calc_SUS_100(self.v100)
            self.VGC = correlations.calc_VGC(self.sg, self.SUS_100)
            self.VG = self.VGC
        else:
            self.VGF = correlations.calc_VGF(self.sg, self.v100)
            self.VG = self.VGF

        xp, xn, xa = correlations.calc_PNA_comp(self.mw, self.VG, self.RI_intercept)

        if self.subtract == 'naphthenes':
            xn = 1 - xp - xa
        elif self.subtract == 'paraffins':
            xp = 1 - xn - xa
        else:  # self.subtract == 'aromatics':
            xa = 1 - xp - xn

        if pd.isnull(xp) or pd.isnull(xn) or pd.isnull(xa):
            xp = 1
            xn = 0
            xa = 0

            if self.warning:
                msgs = f"PNA composition failed to solve. Replacing with default xp={xp} (paraffin), " \
                       f"xn={xn} (naphthenic), and xa={xa} (aromatic). From: {self.__class__.__name__}"
                utilities.issue_unique_warning(msgs, SCNPropertyWarning)


        return xp, xn, xa

    def _assign_attributes(self):
        for key, value in self._attributes.items():
            setattr(self, key, value)

    def _resolve_dependencies(self):
        resolved_vars_keys = set([attr for attr, value in self._attributes.items() if value is not None])
        resolved_vars_dict = {key: self._attributes[key] for key in resolved_vars_keys if key in self._attributes}
        resolved_vars_values = list(resolved_vars_dict.values())

        # Resolve other dependencies
        while len(resolved_vars_keys) < len(self._attributes):
            resolved_this_iteration = False
            for correlation_func, variables in self._correlations.items():
                unresolved_vars = [var for var in variables if var not in resolved_vars_keys]
                if len(unresolved_vars) == 1:
                    unresolved_var = unresolved_vars[0]

                    # warning messages need to be printed here prior to non-linear solver, as the solver can trigger
                    # runtime Error and failing to run _check_ranges in the end __init__.
                    if self.warning:
                        utilities.check_ranges(target_dict=resolved_vars_dict,
                                               ref_dict=self._range_warnings,
                                               class_name=self.__class__.__name__,
                                               warning_obj=SCNPropertyWarning)


                    try:
                        self._attributes[unresolved_var] = newton(lambda x: correlation_func(*self._prepare_args(correlation_func, x, resolved_vars_values)), x0=self._get_init_guess(unresolved_var))
                    except RuntimeError as e:
                        custom_message = f"{e}\nCustomMessage: Failed to solve with newton's method for " \
                                         f"unresolved_var: '{unresolved_var}', " \
                                         f"using correlation_func: {correlation_func.__name__}, " \
                                         f"initial guess for unresolved_var: {self._get_init_guess(unresolved_var)}, " \
                                         f"resolved_vars_dict: {resolved_vars_dict}. From: {self.__class__.__name__}"
                        raise RuntimeError(custom_message) from e

                    #self._attributes[unresolved_var] = newton(lambda x: correlation_func(*self._prepare_args(correlation_func, x, resolved_vars_values)), x0=self._get_init_guess(unresolved_var))
                    resolved_vars_keys.add(unresolved_var)
                    resolved_this_iteration = True

            if not resolved_this_iteration:
                break

    def _prepare_args(self, correlation_func, x, resolved_vars_values):
        arg_order = self._correlations[correlation_func]
        args = []
        for arg in arg_order:
            if arg in self._attributes and self._attributes[arg] is not None:
                args.append(self._attributes[arg])
            else:
                args.append(x)
        return args

    def _get_init_guess(self, variable):
        return self.init_guess.get(variable, None)  # return None if key not found, default


class PropertyTable(object):

    def __init__(self, comp_dict, summary=True, warning=True, SCNProperty_kwargs=None):

        self.warning = warning
        self.warning_msgs = []
        self.summary = summary
        self.table_summary = None
        self.target_compound = None

        if SCNProperty_kwargs is None:
            self.SCNProperty_kwargs = {}
        else:
            self.SCNProperty_kwargs = SCNProperty_kwargs
        self.SCNProperty_kwargs.setdefault('warning', self.warning)
        self._validate_SCNProperty_kwargs()

        self.comp_dict, self.unnormalized_sum = utilities.normalize_composition(comp_dict)
        if not (math.isclose(self.unnormalized_sum, 1, abs_tol=1e-9) or math.isclose(self.unnormalized_sum, 100, abs_tol=1e-9)):
            if self.warning:
                comp_dict_items = ",\n".join(f"    '{key}': {value * 100}" for key, value in self.comp_dict.items())
                comp_dict_formatted = f"{{\n{comp_dict_items}\n}}"
                msgs = f"The sum of the composition is not 100 ({self.unnormalized_sum}). The composition has been " \
                       f"normalized. To suppress this warning, replace with a normalized composition, Or set " \
                       f"warning=False.\nSuggested normalized dict:\n{comp_dict_formatted}" \
                       f"\nFrom: {self.__class__.__name__}"
                utilities.issue_unique_warning(msgs, PropertyTableWarning)


        self.df_GPA = pd.read_feather(FEATHER_PATH)
        self.names = list(self.comp_dict.keys())
        self.zs = list(self.comp_dict.values())

        self.comp_dict_pure, self.comp_dict_fraction = self._split_known_unknown()
        self.names_pure = list(self.comp_dict_pure.keys())
        self.names_fraction = list(self.comp_dict_fraction.keys())
        self.zs_pure = list(self.comp_dict_pure.values())
        self.zs_fraction = list(self.comp_dict_fraction.values())
        self.n_pure = len(self.names_pure)
        self.n_fraction = len(self.names_fraction)

        self.target_props = ['ghv', 'sg_liq_60F', 'sg_gas_60F', 'mw']  # reference to config.py GPA_table_column_mapping
        self.constants_pure = ChemicalConstantsPackage.constants_from_IDs(self.names_pure)
        self._check_properties_exists()

        self.V_molar = utilities.ideal_gas_molar_volume()

        self.ghvs_gas_pure, self.ghvs_liq_pure, self.sgs_liq_pure, self.sgs_gas_pure, self.mws_pure = self._get_properties_pure_compounds()
        self.ghvs_gas_fraction, self.ghvs_liq_fraction, self.sgs_fraction, self.sgs_gas_fraction, self.mws_fraction, self.scns_fraction = [], [], [], [], [], []

        # table without summary statistics line
        self.table_ = pd.DataFrame.from_dict({
            'Name': self.names_pure + self.names_fraction,
            'CAS': self.constants_pure.CASs + [np.nan] * self.n_fraction,
            'Mole_Fraction': self.zs_pure + self.zs_fraction,
            'MW': self.mws_pure.tolist() + [np.nan] * self.n_fraction,
            'Mass_Fraction': [np.nan] * len(self.names),
            'GHV_gas': self.ghvs_gas_pure.tolist() + [np.nan] * self.n_fraction,
            'GHV_liq': self.ghvs_liq_pure.tolist() + [np.nan] * self.n_fraction,
            'SG_gas': self.sgs_gas_pure.tolist() + [np.nan] * self.n_fraction,
            'SG_liq': self.sgs_liq_pure.tolist() + [np.nan] * self.n_fraction,
        })
        self.table_ = self.table_.set_index('Name').reindex(self.names).reset_index()  # reorder the table to match the input order for fractions

        #self.fraction_indices_dict = {item: idx for idx, item in enumerate(self.table_.Name) if is_fraction(item)}
        #self.pure_indices_dict = {item: idx for idx, item in enumerate(self.table_.Name) if not is_fraction(item)}
        self.compound_indices_dict = {item: idx for idx, item in enumerate(self.table_.Name)}

        self.column_mapping = {
            'Name': ['name', 'compound', 'chemical'],
            'CAS': ['cas', 'casrn', 'cas number'],
            'Mole_Fraction': ['mole_fraction', 'mole fraction', 'mole frac', 'mole', 'mole_frac'],
            'MW': ['mw', 'molar mass', 'molecular weight'],
            'Mass_Fraction': ['mass_fraction', 'mass_frac', 'mass fraction', 'mass frac', 'mass'],
            'GHV_gas': ['ghv_gas', 'ghv gas', 'gas ghv', 'vapor ghv', 'ghv_vapor', 'ghv vapor'],
            'GHV_liq': ['ghv_liq', 'ghv liq', 'liq ghv', 'liquid ghv', 'ghv_liquid', 'ghv liquid'],
            'SG_gas': ['sg_gas', 'sg gas', 'gas sg', 'vapor sg', 'sg_vapor', 'sg vapor'],
            'SG_liq': ['sg_liq', 'sg liq', 'liq sg', 'liquid sg', 'sg_liquid', 'sg liquid'],
        }
        self.summary_stats_method = {
            "Name": None,
            "CAS": None,
            "Mole_Fraction": 'sum',
            "MW": 'mole_frac_mean',
            "Mass_Fraction": 'sum',
            "GHV_gas": 'mole_frac_mean',  # mass_frac_mean for other option
            "GHV_liq": 'mass_frac_mean',
            "SG_gas": 'mole_frac_mean',
            "SG_liq": 'mole_frac_mean',
        }
        self._handle_summary()
        self._internal_update_property()

    def _validate_SCNProperty_kwargs(self):

        constructor_signature = inspect.signature(SCNProperty.__init__)
        allowed_keys = [param.name for param in constructor_signature.parameters.values() if param.name != 'self']

        # Special keywords that are prohibited. These values must be provided in other ways.
        prohibited_keywords = ['sg', 'mw', 'Tb']

        unexpected_keys = []
        prohibited_keys = []

        # Check for unexpected keywords
        for key in self.SCNProperty_kwargs:
            if key not in allowed_keys:
                unexpected_keys.append(f"'{key}'")

        # Check for the use of prohibited keywords
        for key in prohibited_keywords:
            if key in self.SCNProperty_kwargs:
                prohibited_keys.append(f"'{key}'")

        # If there are any unexpected or prohibited keywords, raise appropriate exceptions
        if unexpected_keys:
            raise TypeError(f"__init__() got an unexpected keys(s): {', '.join(unexpected_keys)}")

        if prohibited_keys:
            raise TypeError(f"The use of {', '.join(prohibited_keys)} is prohibited when provided as SCNProperty_kwargs. From: {self.__class__.__name__}")


    def _map_input_to_key(self, user_input):
        user_input_lower = user_input.lower()
        for key, values in self.column_mapping.items():
            if user_input_lower in values:
                return key
        raise ValueError(f"Invalid column name: '{user_input}'. Valid options are "
                         f"{self.column_mapping.keys()}. From: {self.__class__.__name__}")

    def update_total(self, props_dict, target_compound=None, recalc=True):

        if target_compound is not None:
            self.target_compound = target_compound
        #else:
        #    if self.target_compound is None:
        #         pass

        # check if summary is true
        if not self.summary:
            raise ValueError(f"Summary is set to False. Set summary=True to use this method. From: {self.__class__.__name__}")

        excluded_cols = ['Name', 'CAS', 'Mole_Fraction']
        # code to update column value of table_summary with chemical_name as column name. table_summary is always a 1 row df.
        for key, value in props_dict.items():
            column = self._map_input_to_key(key)
            excluded_cols.append(column)
            self.table_summary[column] = value

            operation = self.summary_stats_method[column]
            unknowns = self.table_[column][self.table_[column].isna()]
            unknown_names = self.table_.loc[unknowns.index, 'Name']

            # get index of the target plus fraction to solve for.
            if len(unknowns) > 1:

                # these three lines of codes are just to create temporary table to print error message better
                self.table_summary.loc[self.table_summary.index[0], column] = value
                warnings.simplefilter(action='ignore', category=FutureWarning)
                warning_table = pd.concat([self.table_, self.table_summary])

                raise ValueError(f"More than one ({len(unknowns)}) NaN values found in column '{column}' "
                                 f"for compounds: {list(unknown_names.values)}. Provide their values with "
                                 f"update_property() so there's only 1 unknown value.\n"
                                 f"{warning_table.to_string()}\n"
                                 f"From: {self.__class__.__name__}")
            elif len(unknowns) == 0:
                if self.target_compound is None:
                    self.target_compound = self.names_fraction[0]
                    target_idx = self.compound_indices_dict[self.target_compound]
                    if len(self.names_fraction) != 1:
                        if self.warning:
                            msg = f"Ambiguous 'target_compound' to adjust to match the target Total properties: {props_dict}. The first " \
                                  f"fraction compound '{self.names_fraction[0]}' is assumed to be the 'target_compound'. " \
                                  f"It is recommended to explicitly set 'target_compound' from one of the followings to " \
                                  f"avoid this warning: {self.names_fraction}. Set warning=False to suppress this warning. " \
                                  f"From: {self.__class__.__name__}"
                            utilities.issue_unique_warning(msg, PropertyTableWarning)

                else:
                    target_idx = self.compound_indices_dict[self.target_compound]
            else:  # len(unknowns) == 1
                target_idx = unknowns.index[0]
                self.target_compound = next(key for key, value in self.compound_indices_dict.items() if value == target_idx)

            knowns = self.table_[column].drop(target_idx)

            if operation == 'sum':
                solution = value - knowns.sum()
            elif operation == 'mole_frac_mean':
                mole_frac_knowns = self.table_['Mole_Fraction'].drop(target_idx)
                weighted_sum = (mole_frac_knowns * knowns).sum()
                target_mole_frac = self.table_.loc[target_idx, 'Mole_Fraction']
                solution = (value - weighted_sum) / target_mole_frac
            elif operation == 'mass_frac_mean':
                weighted_sum = (self.table_['Mass_Fraction'] * knowns).sum()
                solution = (value - weighted_sum) / self.table_.loc[target_idx, 'Mass_Fraction']
            elif operation is None:
                solution = None
            else:
                raise ValueError(f"Invalid operation: '{operation}'. Valid options are 'sum', 'mean', 'mole_frac_mean', 'mass_frac_mean'. From: {self.__class__.__name__}")

            self.table_.loc[target_idx, column] = solution

        if recalc is True:
            # _internal_update_property() triggers calculations only on cells with nan values
            self.table_.loc[target_idx, self.table_.columns.difference(excluded_cols)] = np.nan



        # three iterations are needed to calculate column properties from left to right
        # this should be fast because only the empty cells are calculated. Non-empty cells are skipped
        warnings.simplefilter('once')
        self._internal_update_property()
        self._internal_update_property()
        self._internal_update_property()
        warnings.resetwarnings()

        # code to check if self.table.loc[max(self.compound_indices_dict.values()) + 1, column] is np.nan
        if pd.isna(self.table.loc[max(self.compound_indices_dict.values()) + 1, column]):
            self.table.loc[max(self.compound_indices_dict.values()) + 1, column] = value

    # update directly from the user input
    def update_property(self, name, props_dict, recalc=True):

        self._validate_chemical_name(name)

        row_index = self.table_[self.table_['Name'] == name].index

        properties = []
        for key, value in props_dict.items():
            key = self._map_input_to_key(key)
            self.table_.loc[row_index, key] = value
            properties.append(key)

        if recalc is True:
            excluded_cols = ['Name', 'CAS', 'Mole_Fraction', *properties]
            target_idx = self.compound_indices_dict[name]
            self.table_.loc[target_idx, self.table_.columns.difference(excluded_cols)] = np.nan

        # three iterations are needed to calculate column properties from left to right
        # this should be fast because only the empty cells are calculated. Non-empty cells are skipped
        self._internal_update_property()
        self._internal_update_property()
        self._internal_update_property()

    def _validate_chemical_name(self, name):
        if name not in self.names:
            raise ValueError(f"Chemical name '{name}' is not found in the provided composition: "
                             f"{self.names}. From: {self.__class__.__name__}")

    def _internal_update_property(self):

        rules = {
            'MW': {
                'weighted_avg': {
                    'required_columns': [['Mole_Fraction']],
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
            'Mass_Fraction': {
                'weighted_avg': {  # notes: weighted average method needs to be removed.
                    'required_columns': [['Mass_Fraction']],
                    'total_required': [['Mass_Fraction']],
                    'required_others': [[None]],
                    'funcs': [None],
                },
                'correlation': {
                    'required_columns': [['Mole_Fraction', 'MW']],
                    'total_required': [['MW']],
                    'required_others': [[None]],
                    'funcs': [self._calc_mass_frac_from_mole_frac_mw],
                },
            },
            'GHV_gas': {
                'weighted_avg': {
                    'required_columns': [['Mole_Fraction']],
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
                    'required_columns': [['Mass_Fraction']],
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
                    'required_columns': [['Mole_Fraction']],
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
                    'required_columns': [['Mass_Fraction']],
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

    def _calc_summary(self):

        summary_row = {}
        for column, operation in self.summary_stats_method.items():

            # If any NaN values are present in the column, summary stats can't be calculated
            if pd.isnull(self.table_[column]).any():
                summary_row[column] = np.nan
                continue

            summary_row[column] = np.nan
            if operation == 'sum':
                summary_row[column] = self.table_[column].sum()
            elif operation == 'mole_frac_mean':
                if not pd.isnull(self.table_['Mole_Fraction']).any():
                    weighted_sum = (self.table_['Mole_Fraction'] * self.table_[column]).sum()
                    summary_row[column] = weighted_sum / self.table_['Mole_Fraction'].sum()
            elif operation == 'mass_frac_mean':
                if not pd.isnull(self.table_['Mass_Fraction']).any():
                    weighted_sum = (self.table_['Mass_Fraction'] * self.table_[column]).sum()
                    summary_row[column] = weighted_sum / self.table_['Mass_Fraction'].sum()
            elif operation is None:
                pass
            else:
                raise ValueError(f"Invalid operation: '{operation}'. Valid options are 'sum', 'mean', 'mole_frac_mean', 'mass_frac_mean'. From: {self.__class__.__name__}")

        summary_row['Name'] = 'Total'
        summary_df = pd.DataFrame([summary_row], index=[self.n_fraction + self.n_pure])

        return summary_df

    def _handle_summary(self):
        if self.summary:
            self.table_summary = self._calc_summary()
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
                        if func is None:
                            raise NotImplementedError(
                                f"property '{property}' is not computed because its function is not implemented yet. This message should not be triggered. Please submit Issues on github if you see this message.")

                        try:
                            calculated_value = func(*args, *args_others, *args_total)
                        except RuntimeError as e:
                            if self.warning:
                                msgs = str(e)
                                if 'From: ' not in msgs:
                                    msgs += f" From: {self.__class__.__name__}"
                                msgs += f"\nValue is set as NaN. Please check the input values and try again. " \
                                        f"To suppress this warning, set warning=False."
                                utilities.issue_unique_warning(msgs, PropertyTableWarning)

                            calculated_value = None

                        if calculated_value is not None:
                            self.table_.loc[idx, property] = calculated_value
                            break  # Stop after the first successful calculation

    def _calc_GHV_gas_from_mw(self, mw):
        scn_obj = SCNProperty(mw=mw, **self.SCNProperty_kwargs)
        aromatic_fraction = scn_obj.xa
        GHV_gas = newton(lambda ghv_gas: correlations.mw_ghv(mw, ghv_gas, aromatic_fraction), x0=5000, maxiter=50)
        return GHV_gas

    def _calc_sg_liq_from_mw(self, mw):
        scn_obj = SCNProperty(mw=mw, **self.SCNProperty_kwargs)
        sg_liq = scn_obj.sg
        return sg_liq

    def _calc_sg_gas_from_mw(self, mw):
        sg_gas = newton(lambda sg_gas: correlations.mw_sg_gas(mw, sg_gas), x0=0.65, maxiter=50)
        return sg_gas

    def _calc_mw_from_sg_gas(self, sg_gas):
        mw = newton(lambda mw: correlations.mw_sg_gas(mw, sg_gas), x0=90, maxiter=50)
        return mw

    def _calc_mw_from_sg_liq(self, sg_liq):
        scn_obj = SCNProperty(sg=sg_liq, **self.SCNProperty_kwargs)
        mw = scn_obj.mw
        return mw

    def _calc_mw_from_ghv_gas(self, ghv_gas):

        def objective(mw, GHV_gas):
            # turn off warning because this is party of iterative convergence
            SCNProperty_kwargs_copy = copy.deepcopy(self.SCNProperty_kwargs)
            SCNProperty_kwargs_copy['warning'] = False
            xa = SCNProperty(mw=mw, **self.SCNProperty_kwargs).xa  # provide guess value of xa to improve model accuracy
            return abs(correlations.mw_GHV_gas_xa(mw, GHV_gas, xa))

        # initial MW guess assuming 100% paraffinic composition
        initial_mw_guess = np.array([newton(lambda mw: correlations.mw_ghv_paraffinic(mw, ghv_gas), x0=100)])
        bounds = [(0, None)]
        result = minimize(lambda mw: objective(mw[0], ghv_gas), initial_mw_guess, bounds=bounds, tol=0.01)
        mw = result.x[0]

        # final check to raise warning if anything is outside the working range
        SCNProperty(mw=mw, **self.SCNProperty_kwargs)

        return mw

    def _calc_mass_frac_from_mole_frac_mw(self, mole_frac, mw, mw_total):
        return mole_frac * mw / mw_total

    def _calc_GHV_liq_from_GHV_gas_MW(self, ghv_gas, mw):
        return newton(lambda ghv_liq: correlations.ghv_liq_ghv_gas_mw(ghv_liq, ghv_gas, mw), x0=0.65, maxiter=50)

    def _get_properties_pure_compounds(self):
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
                        raise ValueError(
                            f"Chemical name '{name}' is recognized but missing a required data ('Hc, heat of combustion [J/mol]'). From: {self.__class__.__name__}")
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
                    raise ValueError(
                        f"Chemical name '{name}' is recognized but missing a required data ('Hc, heat of combustion [J/mol]'). From: {self.__class__.__name__}")
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

    def _check_properties_exists(self):
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
                    raise ValueError(
                        f"Chemical name '{name}' is recognized but missing a required data ('rhol_60F_mass, liquid mass density at 60F'). From: {self.__class__.__name__}")
            if 'mw' in self.target_props:
                if mw is None:
                    raise ValueError(
                        f"Chemical name '{name}' is recognized but missing a required data ('MW, molecular weight [g/mol]'). From: {self.__class__.__name__}")
            if 'ghv' in self.target_props:
                if Hc is None:
                    raise ValueError(
                        f"Chemical name '{name}' is recognized but missing a required data ('Hc, heat of combustion [J/mol]'). From: {self.__class__.__name__}")


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

ovintiv_tomlin = dict([
    ('nitrogen', 6.436),
    ('carbon dioxide', 0.848),
    ('methane', 30.232),
    ('ethane', 14.229),
    ('propane', 16.889),
    ('isobutane', 3.797),
    ('n-butane', 11.066),
    ('i-pentane', 3.141),
    ('n-pentane', 5.070),
    ('n-hexane', 6.776),
    ('heptane', 1.462),
    ('n-octane', 0.053),
    ('n-nonane', 0.001),
    #('Heptanes+', 0.4),
])

# Shamrock
shamrock = dict([
    ('nitrogen', 0.86),  # 0.86
    ('carbon dioxide', 0.374),
    ('methane', 71.351),
    ('ethane', 12.287),
    ('propane', 9.147),
    ('isobutane', 0.985),
    ('n-butane', 3.180),
    ('i-pentane', 0.628),
    ('n-pentane', 0.743),  # 0.743
    ('Hexanes+', 0.445),
    #('Heptanes+', 0.2),
    #('heneicosane', 0.01),
])


#df = SCNProperty.build_table([i for i in range(82, 94, 1)], warning=False, col='mw', output_keys=['mw', 'v100', 'v210', 'SUS_100', 'VGC'], model='kf')
#df = SCNProperty.build_table([i for i in range(84, 94, 1)], col='mw', warning=True, model='kf')
#print(df.to_string())


# complete test scripts
# ptable = PropertyTable(shamrock, summary=True, warning=True, SCNProperty_kwargs={'warning': False, 'model': 'kf'})


# Brazos Trees Gas
#ptable = PropertyTable(brazos_gas, summary=True, warning=True, SCNProperty_kwargs={'warning': True, 'model': 'kf'})
#ptable.update_property('Hexanes+', {'GHV_liq': 20408})
#ptable.update_property('Hexanes+', {'GHV_gas': 4849})
#ptable.update_property('Hexanes+', {'liquid sg': 0.7142})
#ptable.update_property('Hexanes+', {'gas sg': 3.1228})
#ptable.update_property('Hexanes+', {'gas sg': 3.396934})
#ptable.update_property('Hexanes+', {'mw': 86.161})  # 86.161
#ptable.update_property('Hexanes+', {'liquid sg': 0.65})
#ptable.update_property('Heptanes+', {'mw': 32.24})

#ptable.update_total({'mw': 25.251}, target_compound='Hexanes+') #23.251, 23.323927
#ptable.update_total({'mw': 25.251}, target_compound='Heptanes+')
#ptable.update_total({'mw': 23.323927}) #23.251, 23.323927
#ptable.update_total({'liquid sg': 0.3740})  #0.3740, 0.358661, the error
#ptable.update_total({'liquid sg': 0.358661})
#ptable.update_total({'gas sg': 0.8053})  # computed from C6+ mw=90.161 is 0.802769
#ptable.update_total({'gas sg': 0.81}, recalc=True)
#ptable.update_total({'gas sg': 0.91}, recalc=True)

#ptable.update_total({'GHV_gas': 1325.1})
#ptable.update_total({'GHV_liq': 51546}, recalc=False)


#ptable.update_total({'gas sg': 0.81})
#ptable.update_total({'gas sg': 0.81, 'mw': 113.7})

#ptable.update_property('Hexanes+', {'gas sg': 3.1228}, recalc=True)




# Todo: implement multiple inputs at one time scenario

# ovintiv_tomlin
#ptable = PropertyTable(ovintiv_tomlin, summary=True, warning=True)
#ptable = PropertyTable(ovintiv_tomlin, summary=False, warning=True)

# Shamrock
# notes: make a note that total method is not recommended. It's extremely sensitive to error in inputs for all properties.


#ptable = PropertyTable(shamrock, summary=True, warning=True, SCNProperty_kwargs={'warning': True, 'model': 'kf'})
#ptable.update_property('Hexanes+', {'GHV_liq': 20677})
#ptable.update_property('Hexanes+', {'GHV_gas': 4774})
#ptable.update_property('Hexanes+', {'liquid sg': 0.6933})
#ptable.update_property('Hexanes+', {'gas sg': 3.0180})
#ptable.update_property('Hexanes+', {'mw': 87.665})  # 87.665
#ptable.update_total({'mw': 23.380})
#ptable.update_total({'liquid sg': 0.3694})
#ptable.update_total({'gas sg': 0.8054})
#ptable.update_total({'GHV_gas': 1380})
#ptable.update_total({'mw': 23.380, 'GHV_liq': 22402})

# MW is required for GHV_liq calculation, this is a Todo

#ptable.update_total({'GHV_liq': 22402, 'mw': 23.380})

# ptable.update_property('Heptanes+', {'mw': 96.665})  # 87.665

#ptable.update_property('Hexanes+', {'mw': 87.665, 'liquid sg': 0.6933})
#ptable.update_property('Hexanes+', {'liquid sg': 0.6933, 'mw': 87.665})
#ptable.update_property('Hexanes+', {'liquid sg': 0.6933})
#ptable.update_property('Hexanes+', {'liquid sg': 0.6933, 'mw': 87.665})


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
#print(ptable.table.to_string())

# Todo: Write example run for ovintiv tomlin by defining the 4 pseudos
# Todo: Do StateCordell, which explicitly shows 6:3:1
# Todo: make dictionary objects for each of Plus fraction, and total properties
