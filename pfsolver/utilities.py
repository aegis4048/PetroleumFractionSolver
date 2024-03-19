import config


CONSTANTS = config.constants


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