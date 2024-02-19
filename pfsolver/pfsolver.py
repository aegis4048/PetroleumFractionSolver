import numpy as np
import pandas as pd
import pint
from thermo import ChemicalConstantsPackage


CONSTANTS = {
    "T_STANDARD": 288.7056,  # Temperature in Kelvin
    "P_STANDARD": 101325.0,    # Pressure in Pascal
    "R": 8.31446261815324,
    "MW_AIR": 28.97,  # molecular weight of air at standard conditions, g/mol
    "RHO_WATER": 999.0170125317171,  # density of water @60F, 1atm (kg/m^3) according to IAPWS-95 standard. Calculate rho at different conditions by:  chemicals.iapws95_rho(288.706, 101325) (K, pascal)
}
# simpler string to match the GPA table column names.
MAPPING = {
    'ghv': 'Gross Heating Value Ideal Gas [Btu/ft^3]',
    'sg_liq_60F': 'Liq. Relative Density @60F:1atm',
    'sg_gas_60F': 'Ideal Gas Relative Density @60F:1atm',
    'mw': 'Molar Mass [g/mol]',
}

UREG = pint.UnitRegistry()


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


class PropertyTable(object):

    def __init__(self, comp_dict):

        self.comp_dict = normalize_composition(comp_dict)
        self.df_GPA = pd.read_pickle("GPA 2145-16 Compound Properties Table - English.pkl")
        self.names = list(self.comp_dict.keys())
        self.zs = list(self.comp_dict.values())

        self.comp_dict_pure, self.comp_dict_fraction = self._split_known_unknown()
        self.names_pure = list(self.comp_dict_pure.keys())
        self.names_fraction = list(self.comp_dict_fraction.keys())
        self.zs_pure = list(self.comp_dict_pure.values())
        self.zs_fraction = list(self.comp_dict_fraction.values())

        self.target_props = ['ghv', 'sg_liq_60F', 'sg_gas_60F', 'mw']

        self.constants_pure = ChemicalConstantsPackage.constants_from_IDs(self.names_pure)
        self.check_properties_exists()

        self.V_molar = ideal_gas_molar_volume()

        self.ghvs_pure, self.sgs_liq_pure, self.sgs_gas_pure, self.mws_pure = self.get_properties_pure_compounds()

        self.table = pd.DataFrame(data=[
            self.names_pure,
            self.constants_pure.CASs,
            self.zs_pure
        ])

        self.table = pd.DataFrame.from_dict({
            'Name': self.names_pure + self.names_fraction,
            'CAS': self.constants_pure.CASs + [None],
            'Mole Fraction': self.zs_pure + self.zs_fraction,
        })

    def solve_ghv_gas(self, ghv_total, update=True):
        wghtd_ghv_pure = np.sum(self.ghvs_pure * self.zs_pure)
        wghtd_ghv_fraction = ghv_total - wghtd_ghv_pure
        ghv_fraction = wghtd_ghv_fraction / sum(self.zs_fraction)

        if update is True:
            self.table['GHV'] = np.append(self.ghvs_pure, ghv_fraction)

        return wghtd_ghv_fraction / sum(self.zs_fraction)

    def calc_mw(self, SCN=7, update=True):
        pass # calculate with scn value

    def get_properties_pure_compounds(self):
        """
        :param constants: thermo's constants object
        :param GPA_data: pandas dataframe of the GPA 2145-16 Table
        :return:
        """
        ghvs_ideal_gas = []
        sgs_liq = []
        sgs_gas = []
        mws = []

        for cas, name, Hc, mw, rhol_60F_mass in zip(self.constants_pure.CASs, self.constants_pure.names, self.constants_pure.Hcs, self.constants_pure.MWs, self.constants_pure.rhol_60Fs_mass):

            matching_row = self.df_GPA[self.df_GPA['CAS'] == cas]

            # chemical is found in the GPA data table
            if not matching_row.empty:
                ghv_ideal_gas = matching_row[MAPPING['ghv']].iloc[0]
                sg_liq = matching_row[MAPPING['sg_liq_60F']].iloc[0]
                sg_gas = matching_row[MAPPING['sg_gas_60F']].iloc[0]
                mw = matching_row[MAPPING['mw']].iloc[0]

                if pd.isna(ghv_ideal_gas):
                    if Hc != 0:  # GPA table is missing GHV data for some compounds: hydrogen chloride
                        ghv_ideal_gas = Hc / self.V_molar
                        ghv_ideal_gas = UREG('%.15f joule/m^3' % ghv_ideal_gas).to('Btu/ft^3')._magnitude * -1
                    elif Hc is None:
                        raise ValueError("Chemical name '%s' is recognized but missing a required data (%s)." % (
                        name, 'Hcs, heat of combustion [J/mol]'))
                    else:
                        ghv_ideal_gas = 0

            # chemical is NOT identified in the GPA datatable
            else:
                if Hc != 0:
                    ghv_ideal_gas = Hc / self.V_molar
                    ghv_ideal_gas = UREG('%.15f joule/m^3' % ghv_ideal_gas).to('Btu/ft^3')._magnitude * -1
                elif Hc is None:
                    raise ValueError("Chemical name '%s' is recognized but missing a required data (%s)." % (
                    name, 'Hcs, heat of combustion [J/mol]'))
                else:
                    ghv_ideal_gas = 0

                sg_liq = rhol_60F_mass / CONSTANTS['RHO_WATER']
                sg_gas = mw / CONSTANTS['MW_AIR']

            ghvs_ideal_gas.append(ghv_ideal_gas)
            sgs_liq.append(sg_liq)
            sgs_gas.append(sg_gas)
            mws.append(mw)

        return np.array(ghvs_ideal_gas), np.array(sgs_liq), np.array(sgs_gas), np.array(mws)

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


brazos = dict([
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
    ('fractions', 0.889),
])

ptable = PropertyTable(brazos)

table = ptable.table

# print the dataframe table nicely
print(ptable.solve_ghv_gas(ghv_total=1320))

print(table.to_string())








