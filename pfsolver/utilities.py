from . import config
import warnings
import numpy as np


CONSTANTS = config.constants
ISSUED_WARNINGS_REGISTRY = set()


def validate_kwargs(kwargs_options, obj):
    for model_name, model_choices in kwargs_options.items():
        user_choice = getattr(obj, model_name)
        if user_choice is None or user_choice not in model_choices:
            user_choice_repr = f"'{user_choice}'" if isinstance(user_choice, str) else user_choice
            raise ValueError(
                f"Invalid {model_name}: {user_choice_repr}. Valid options: {model_choices}. From: {obj.__class__.__name__}")


def issue_unique_warning(message, warning_type=None):
    """
    Issues a warning if it hasn't been issued before.

    Args:
    - message (str): The warning message.
    - warning_type (Warning): The type of the warning.
    """
    warning_signature = (message, warning_type)

    if warning_signature not in ISSUED_WARNINGS_REGISTRY:
        warnings.warn(message, warning_type)
        ISSUED_WARNINGS_REGISTRY.add(warning_signature)


def handle_rounding_error(value, key):
    n = config.rounding_error_decimal_points[key]
    return round(value, n)


def normalize_composition_dict(comp_dict):
    """
    :param comp_dict: un-normalized dictionary of composition. {"CH4": 3, "C2H6", 6}
    :return: normalized dictionary of composition. {"CH4": 0.3333, "C2H6", 0.66666666}
    """
    total_comp = sum(comp_dict.values())
    total_comp = round(total_comp, 9)  # warning is triggered if total comp is not 1 or 100

    keys = list(comp_dict.keys())
    last_key = keys[-1]
    normalized_values = np.array([v / total_comp for v in comp_dict.values()])

    # Adjust the last element so that the sum is exactly 1
    comp_dict = {keys[i]: normalized_values[i] for i in range(len(keys) - 1)}
    comp_dict[last_key] = 1 - sum(comp_dict.values())
    return comp_dict, total_comp


def normalize_composition_list(zs):
    """
    :param zs: un-normalized list of composition. [3, 6]
    :return: normalized list of composition. [0.3333, 0.66666666]
    """
    total_comp = sum(zs)
    total_comp = round(total_comp, 9)  # warning is triggered if total comp is not 1 or 100
    normalized_values = np.array([v / total_comp for v in zs])
    normalized_values[-1] = 1 - normalized_values[:-1].sum()

    return np.array(normalized_values), total_comp


def ideal_gas_molar_volume():
    """
    PV=nRT, where number of moles n=1. Rearranging -> V=RT/P
    R = 8.31446261815324 ((m^3-Pa)/(mol-K))
    T = 288.7056 K, 60F, standard temperature
    P = 101325 Pa, 1 atm, standard pressure
    :return: ideal gas molar volume in a standard condition (m^3/mol) = 0.023690421108823113
    """
    return CONSTANTS['R'] * CONSTANTS['T_STANDARD'] / CONSTANTS['P_STANDARD']


def check_ranges(target_dict, ref_dict, class_name=None, warning_obj=None):
    for target_key, target_val in target_dict.items():
        min_val = ref_dict[target_key][0]
        max_val = ref_dict[target_key][1]
        if target_val is not None and not (min_val <= target_val <= max_val):
            msg = f"{target_key} value {target_val} is out of recommended working range [{min_val}, {max_val}]. Set warning=False to suppress this warning."
            if class_name is not None:
                msg += f" From: {class_name}"
            #issue_unique_warning(msg, warning_obj)
            warnings.warn(msg, warning_obj)


def setbold(txt):
    return ' '.join([r"$\bf{" + item + "}$" for item in txt.split(' ')])

