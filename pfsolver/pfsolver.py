import numpy as np
import pandas as pd
import pint
from thermo import ChemicalConstantsPackage
import correlations
import warnings
from scipy.optimize import newton


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


class PropertyTable(object):

    def __init__(self, comp_dict, summary=False):

        self.comp_dict = normalize_composition(comp_dict)
        self.df_GPA = pd.read_pickle("GPA 2145-16 Compound Properties Table - English.pkl")
        self.names = list(self.comp_dict.keys())
        self.zs = list(self.comp_dict.values())

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

        self.ghvs_liq_pure, self.sgs_liq_pure, self.sgs_gas_pure, self.mws_pure = self.get_properties_pure_compounds()
        self.ghvs_liq_fraction, self.sgs_fraction, self.sgs_gas_fraction, self.mws_fraction, self.scns_fraction = [], [], [], [], []

        # self.ws_pure = self.zs_pure * self.mws_pure

        # table without summary statistics line
        self.table_ = pd.DataFrame.from_dict({
            'Name': self.names_pure + self.names_fraction,
            'CAS': self.constants_pure.CASs + [None] * self.n_fraction,
            'Mole Fraction': self.zs_pure + self.zs_fraction,
            'MW': self.mws_pure.tolist() + [None] * self.n_fraction,
            'Mass Fraction': [None] * len(self.names),
            'GHV_gas': self.ghvs_liq_pure.tolist() + [None] * self.n_fraction,
            'GHV_liq': self.ghvs_liq_pure.tolist() + [None] * self.n_fraction,
            'SG_gas': self.sgs_gas_pure.tolist() + [None] * self.n_fraction,
            'SG_liq': self.sgs_liq_pure.tolist() + [None] * self.n_fraction,
        })

        self.summary_stats_method = {
            "Name": None,
            "CAS": None,
            "Mole Fraction": 'sum',
            "MW": 'mole_frac_mean',
            "Mass Fraction": 'sum',
            "GHV_gas": 'mole_frac_mean',
            "GHV_liq": 'mole_frac_mean',
            "SG_gas": 'mass_frac_mean',
            "SG_liq": 'mass_frac_mean'
        }
        self.table_summary = self.calc_summary()

        self.table = self.table_

        #warnings.simplefilter(action='ignore', category=FutureWarning)
        #print(pd.concat([self.table_, self.table_summary]).to_string())

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

        # https://stackoverflow.com/questions/77254777/alternative-to-concat-of-empty-dataframe-now-that-it-is-being-deprecated
        # this warning keeps triggering even if the df has no all-nan columns. The solutions in the SO didn't fix so im suppressing the warning.
        # warnings.simplefilter(action='ignore', category=FutureWarning)

        summary_row['Name'] = 'Total'
        summary_df = pd.DataFrame([summary_row], index=[self.n_fraction + self.n_pure])
        #updated_df = pd.concat([self.table_, summary_df])

        return summary_df

    # update directly from the user input
    def update_property(self, name, props_dict):

        if name not in self.names:
            raise ValueError("Chemical name '%s' is found in the provided composition." % name)

        row_index = self.table_[self.table_['Name'] == name].index
        for key, value in props_dict.items():
            if key in self.table_.columns:
                self.table_.loc[row_index, key] = value
            else:
                raise ValueError(f"Column '{key}' does not exist in the DataFrame.")

        self._internal_update_property()


    def update_total_row(self):
        if "Total" not in self.table_summary['Name'].values:
            self.table_summary.loc[self.table_summary.index.max() + 1] = ["Total"] + [np.nan] * (
                        self.table_summary.shape[1] - 1)
        total_row_index = self.table_summary[self.table_summary['Name'] == "Total"].index[0]
        self.table_summary.at[total_row_index, 'Mole Fraction'] = self.table_summary['Mole Fraction'].sum()
        self.table_summary.at[total_row_index, 'Mass Fraction'] = self.table_summary.dropna(subset=['Mass Fraction'])['Mass Fraction'].sum()

    def calc_mass_fraction(self, row_index):
        MW_total = self.table_summary.at[self.table_summary.index.max(), 'MW']
        if not pd.isnull(MW_total):
            self.table_.at[row_index, 'Mass Fraction'] = self.table_.at[row_index, 'Mole Fraction'] * self.table_.at[row_index, 'MW'] / MW_total
        else:
            pass

    def _internal_update_property(self):
        self.table_summary = self.calc_summary()  # Update Total row first
        rules = {
            'Mass Fraction': {
                'weighted_avg': {
                    'required_columns': ['Mass Fraction'],
                    'total_required': True,
                    'required_others': None,
                },
                'correlation': {
                    'required_columns': ['Mole Fraction', 'MW'],
                    'total_required': True,
                    'required_others': None,
                },
            },
            'GHV_gas': {
                'weighted_avg': {
                    'required_columns': ['Mole Fraction'],
                    'total_required': True,
                    'required_others': None,
                },
                'correlation': {
                    'required_columns': ['MW'],
                    'total_required': False,
                    'required_others': ['SCN'], # maybe self.SCNs
                },
            },
            'SG_gas': {
                'weighted_avg': {
                    'required_columns': ['Mass Fraction'],
                    'total_required': True,
                    'required_others': None,
                },
                'correlation': {
                    'required_columns': ['MW'],
                    'total_required': True,
                    'required_others': ['SCN'],
                },
            },
            'SG_liq': {
                'weighted_avg': {
                    'required_columns': ['Mass Fraction'],
                    'total_required': True,
                    'required_others': None,
                },
                'correlation': {
                    'required_columns': ['GHV_gas', 'MW'],
                    'total_required': True,
                    'required_others': None,
                },
            }
        }

        for property, methods in rules.items():
            for method_name, method_details in methods.items():
                # Check if the 'weighted_avg' method can be computed or not, skip to 'correlation' if not
                if method_name == 'correlation':
                    self._handle_correlation_method(property, method_details)

        self.table_summary = self.calc_summary()

        warnings.simplefilter(action='ignore', category=FutureWarning)
        #print(pd.concat([self.table_, self.table_summary]).to_string())

        self.table = pd.concat([self.table_, self.table_summary])
        #print(self.table_.to_string())
        #print('---------------------------------------------------------')
        #print(self.table_summary.to_string())

    def _handle_correlation_method(self, property, method_details):

        required_columns = method_details['required_columns']
        total_required = method_details['total_required']

        # Iterate over each row in the dataframe
        for idx, row in self.table_.iterrows():
            # Check if required columns for the current row have valid (non-NaN) values
            if all(pd.notna(row[col]) for col in required_columns):
                # Check if total is required and valid for calculation
                if total_required and all(pd.notna(self.table_summary[col].values[0]) for col in required_columns):
                    # Perform the correlation calculation here
                    # This is a placeholder for your specific calculation logic

                    if pd.isna(row[property]):  # perform calculation if the cell is empty, otherwise, dont
                        calculated_value = self._calculate_correlation(row, property)
                        self.table_.loc[idx, property] = calculated_value

    def _calculate_correlation(self, row, property):

        if property == 'Mass Fraction':
            MW_total = self.table_summary.at[self.table_summary.index.max(), 'MW']
            calculated_value = row['Mole Fraction'] * row['MW'] / MW_total
        elif property == 'SG_gas':
            calculated_value = newton(lambda sg: correlations.mw_sg_gas(row['MW'], sg), x0=0.65, maxiter=50)
        elif property == 'SG_liq':

            # ghv needs to be in liquid ghv
            calculated_value = newton(lambda sg_liq: correlations.ghv_liq_sg_liq(row['GHV_gas'], sg_liq), x0=0.65, maxiter=50)
        else:
            calculated_value = None

        return calculated_value


    def solve_ghv_gas(self, ghv_total, update=True, name=None):

        if len(self.names_fraction) > 1:
            if name is None:
                raise ValueError("More than 1 fractions are identified: %s. Please specify the name parameter. For example: name='%s'" % (self.names_fraction, self.names_fraction[0]))
            else:
                name = self.names_fraction[0]

        z_fraction = self.table_[self.table_['Name'] == name]['Mole Fraction'].iloc[0]
        z_fraction_index = self.table_[self.table_['Name'] == name].index[0]

        wghtd_ghv_pure = np.sum(self.ghvs_pure * self.zs_pure)
        wghtd_ghv_fraction = ghv_total - wghtd_ghv_pure
        ghv_fraction = wghtd_ghv_fraction / z_fraction

        if update is True:
            #self.table_['GHV_gas'] = np.append(self.ghvs_pure, ghv_fraction)
            self.ghvs_fraction.append(ghv_fraction)
            # call self._internal_update_property() to update the table for consistent update method.



        return ghv_fraction
        #return wghtd_ghv_fraction / sum(self.zs_fraction)

    def calc_mw(self, SCN=7, name=None):




        MW = correlations.calc_MW_from_GHV(1, aromatic_fraction=0)

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

                    if Hc != 0:  # GPA table is missing GHV_gas data for some compounds: 1-propyne
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
    ('Hexanes+', 0.889),
    #('Heptanes+', 0.4),
])

ptable = PropertyTable(brazos, summary=False)

# print the dataframe table nicely


#ptable.update_property('Hexanes+', {'MW':90.161, 'GHV_gas': 4849, 'SG_gas': 3.1228, 'SG_liq': 0.7142})


#ptable.update_property('ethane', {'MW':100000, 'GHV_gas': 4849, 'SG_gas': 3.1228, 'SG_liq': 0.7142})
print(ptable.table.to_string())

print('----------------------------------------------------------')
ptable.update_property('Hexanes+', {'MW':90.161, })


print(ptable.table.to_string())

# print(ptable.solve_ghv_gas(ghv_total=1320, name='Hexanes+'))

# print(ptable.table.to_string())
#print(ptable.ghvs_fraction)

scn_obj = SCNProperty(scn=7)
#print(scn_obj.xa)

df = SCNProperty.build_table([i for i in range(6, 15)])

# Todo: implement a weighted average linear regression prediction

# This print line needs to replace the Table 1 of my article
# Description: MW, Tb, and SG predicted from SCN with methods introduced in [1]. PNA composition predicted with methods introduced in [2] and [4].
# print(df.to_string())

