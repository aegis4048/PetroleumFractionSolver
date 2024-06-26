# Configuration dictionary with default values
constants = {
    "T_STANDARD": 288.7056,  # Temperature in Kelvin
    "P_STANDARD": 101325.0,    # Pressure in Pascal
    "R": 8.31446261815324,
    "MW_AIR": 28.9625,  # molecular weight of air at standard conditions, g/mol, GPA 2172-19
    "RHO_WATER": 999.0170125317171,  # density of water @60F, 1atm (kg/m^3) according to IAPWS-95 standard. Calculate rho at different conditions by:  chemicals.iapws95_rho(288.706, 101325) (K, pascal)
}
GPA_table_column_mapping = {
    'ghv_gas': 'Gross Heating Value Ideal Gas [Btu/ft^3]',
    'ghv_liq': 'Gross Heating Value Ideal Gas [Btu/lbm]',
    'sg_liq_60F': 'Liq. Relative Density @60F:1atm',
    'sg_gas_60F': 'Ideal Gas Relative Density @60F:1atm',
    'mw': 'Molar Mass [g/mol]',
    'Tb': 'Boiling T. [F]',
    'Tc': 'Crit T. [F]',
    'Pc': 'Crit. P. [psia]',
    'omega': 'h',
}
rounding_error_decimal_points = {
    'mw': 6,
}
settings = {
    'warning': True,
}


def update_config(user_config):
    """
    Update configuration values using a user-provided dictionary.
    :param user_config: A dictionary containing configuration keys and their new values
    """
    constants.update(user_config)


def update_settings(settings_dict):
    global settings
    settings.update(settings_dict)


