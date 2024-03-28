from . import config
import warnings


CONSTANTS = config.constants


def handle_rounding_error(value, key):
    n = config.rounding_error_decimal_points[key]
    return round(value, n)

def normalize_composition(comp_dict):
    """
    :param comp_dict: un-normalized dictionary of composition. {"CH4": 3, "C2H6", 6}
    :return: normalized dictionary of composition. {"CH4": 0.3333, "C2H6", 0.66666666}
    """
    total_comp = sum(comp_dict.values())
    total_comp = round(total_comp, 9)
    if total_comp > 0:
        keys = list(comp_dict.keys())
        last_key = keys[-1]
        normalized_values = [v / total_comp for v in comp_dict.values()]

        # Normalize all but the last element
        comp_dict = {keys[i]: normalized_values[i] for i in range(len(keys) - 1)}

        # Adjust the last element so that the sum is exactly 1
        comp_dict[last_key] = 1 - sum(comp_dict.values())

    return comp_dict, total_comp


def ideal_gas_molar_volume():
    """
    PV=nRT, where number of moles n=1. Rearranging -> V=RT/P
    R = 8.31446261815324 ((m^3-Pa)/(mol-K))
    T = 288.7056 K, 60F, standard temperature
    P = 101325 Pa, 1 atm, standard pressure
    :return: ideal gas molar volume in a standard condition (m^3/mol)
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
            warnings.warn(msg, warning_obj)

