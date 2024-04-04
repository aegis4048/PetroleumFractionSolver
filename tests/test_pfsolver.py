import unittest
import sys
import pint
import numpy as np
import pandas as pd
import warnings
import random
from scipy.optimize import newton
from thermo import ChemicalConstantsPackage

sys.path.append('.')

from pfsolver import PropertyTable, SCNProperty
from pfsolver.customExceptions import SCNPropertyWarning, PropertyTableWarning, ThermoMissingValueWarning
from pfsolver import correlations
from pfsolver import utilities


UREG = pint.UnitRegistry()


# shamrock
sample_1 = dict([
    ('nitrogen', 0.86),
    ('carbon dioxide', 0.374),
    ('methane', 71.351),
    ('ethane', 12.287),
    ('propane', 9.147),
    ('isobutane', 0.985),
    ('n-butane', 3.180),
    ('i-pentane', 0.628),
    ('n-pentane', 0.743),
    ('Hexanes+', 0.445),
])

# shamrock order changed
sample_1a = dict([
    ('nitrogen', 0.86),
    ('carbon dioxide', 0.374),
    ('methane', 71.351),
    ('ethane', 12.287),
    ('propane', 9.147),
    ('isobutane', 0.985),
    ('n-butane', 3.180),
    ('i-pentane', 0.628),
    ('Hexanes+', 0.445),
    ('n-pentane', 0.743),  # 0.743
])

# shamrock, 2 fractions
sample_1b = dict([
    ('nitrogen', 0.86),  # 0.86
    ('carbon dioxide', 0.374),
    ('methane', 71.351),
    ('ethane', 12.287),
    ('propane', 9.147),
    ('isobutane', 0.985),
    ('n-butane', 3.180),
    ('i-pentane', 0.628),
    ('n-pentane', 0.743),
    ('Hexanes+', 0.245),
    ('Heptanes+', 0.2),
])

# Ovintiv Tomlin
sample_2 = dict([
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
])

# brazos condensates
sample_3 = dict([
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
])

air = {
    'nitrogen': 78.10222649645685,
    'oxygen': 20.946060743576158,
    'argon': 0.9160026564077035,
    'carbon dioxide': 0.03300009570027753,
    'neon': 0.0018200052780153063,
    'helium': 0.0005200015080043732,
    'methane': 0.00015000043500126153,
    'krypton': 0.00011000031900092511,
    'hydrogen': 5.000014500042051e-05,
    'nitrous oxide': 3.0000087000252303e-05,
    'carbon monoxide': 2.00000580001682e-05,
    'xenon': 1.0000028982659614e-05
}

# Todo: need another sample to test non-GPA properties

# Documents\GasCompressibiltiyFactor-py>python -m unittest tests.test_gascomp
# python -m unittest discover .




class Test_SCNProperty(unittest.TestCase):

    def test_SCNProperty(self):

        # default kwargs
        kwargs = {
            'mw': 90,
            'model': 'kf',
            'init_guess': None,
            'subtract': 'naphthenes',
            'round': None,
            'warning': True
        }

        # calculation check 1
        test1 = SCNProperty(**{**kwargs, 'model': 'kf'})
        self.assertAlmostEqual(test1.sg, 0.705407, places=3)
        self.assertAlmostEqual(test1.Tb, 638.868494, places=3)
        self.assertAlmostEqual(test1.xp, 0.648636, places=3)
        self.assertAlmostEqual(test1.xn, 0.307898, places=3)
        self.assertAlmostEqual(test1.xa, 0.043466, places=3)
        self.assertAlmostEqual(test1.ri, 1.392851, places=3)
        self.assertAlmostEqual(test1.v100, 0.465191, places=3)
        self.assertAlmostEqual(test1.v210, 0.313680, places=3)
        self.assertAlmostEqual(test1.xa + test1.xn + test1.xp, 1, places=3)

        # calculation check 1 - upper limit
        test11 = SCNProperty(**{**kwargs, 'mw': 626, 'model': 'kf'})
        self.assertAlmostEqual(test11.sg, 0.937260, places=3)
        self.assertAlmostEqual(test11.Tb, 1483.440185, places=3)
        self.assertAlmostEqual(test11.xp, 0.560982, places=3)
        self.assertAlmostEqual(test11.xn, 0.358220, places=3)
        self.assertAlmostEqual(test11.xa, 0.080799, places=3)
        self.assertAlmostEqual(test11.ri, 1.500194, places=3)
        self.assertAlmostEqual(test11.v100, 1117.940534, places=3)
        self.assertAlmostEqual(test11.v210, 33.219443, places=3)
        self.assertAlmostEqual(test11.xa + test11.xn + test11.xp, 1, places=3)

        # calculation check 1 - lower limit
        test12 = SCNProperty(**{**kwargs, 'mw': 84, 'model': 'kf'})
        self.assertAlmostEqual(test12.sg, 0.684977, places=3)
        self.assertAlmostEqual(test12.Tb, 615.533924, places=3)
        self.assertAlmostEqual(test12.xp, 0.729559, places=3)
        self.assertAlmostEqual(test12.xn, 0.264685, places=3)
        self.assertAlmostEqual(test12.xa, 0.005756, places=3)
        self.assertAlmostEqual(test12.ri, 1.382405, places=3)
        self.assertAlmostEqual(test12.v100, 0.420901, places=3)
        self.assertAlmostEqual(test12.v210, 0.292877, places=3)
        self.assertAlmostEqual(test12.xa + test12.xn + test12.xp, 1, places=3)

        # calculation check 2
        test2 = SCNProperty(**{**kwargs, 'model': 'ra'})
        self.assertAlmostEqual(test2.sg, 0.714185, places=3)
        self.assertAlmostEqual(test2.Tb, 639.374343, places=3)
        self.assertAlmostEqual(test2.xp, 0.605375, places=3)
        self.assertAlmostEqual(test2.xn, 0.334476, places=3)
        self.assertAlmostEqual(test2.xa, 0.060150, places=3)
        self.assertAlmostEqual(test2.ri, 1.397132, places=3)
        self.assertAlmostEqual(test2.v100, 0.469588, places=3)
        self.assertAlmostEqual(test2.v210, 0.314390, places=3)
        self.assertAlmostEqual(test2.xa + test2.xn + test2.xp, 1, places=3)

        # calculation check 2 - upper limit
        test21 = SCNProperty(**{**kwargs, 'mw': 698, 'model': 'ra'})
        self.assertAlmostEqual(test21.sg, 0.94701, places=3)
        self.assertAlmostEqual(test21.Tb, 1531.864931, places=3)
        self.assertAlmostEqual(test21.xp, 0.559841, places=3)
        self.assertAlmostEqual(test21.xn, 0.368586, places=3)
        self.assertAlmostEqual(test21.xa, 0.071573, places=3)
        self.assertAlmostEqual(test21.ri, 1.501941, places=3)
        self.assertAlmostEqual(test21.v100, 3205.849618, places=3)
        self.assertAlmostEqual(test21.v210, 55.62652, places=3)
        self.assertAlmostEqual(test21.xa + test21.xn + test21.xp, 1, places=3)

        # calculation check 2 - lower limit
        test22 = SCNProperty(**{**kwargs, 'mw': 82, 'model': 'ra'})
        self.assertAlmostEqual(test22.sg, 0.690101, places=3)
        self.assertAlmostEqual(test22.Tb, 608.034744, places=3)
        self.assertAlmostEqual(test22.xp, 0.694298, places=3)
        self.assertAlmostEqual(test22.xn, 0.2924, places=3)
        self.assertAlmostEqual(test22.xa, 0.013302, places=3)
        self.assertAlmostEqual(test22.ri, 1.384475, places=3)
        self.assertAlmostEqual(test22.v100, 0.41346, places=3)
        self.assertAlmostEqual(test22.v210, 0.287399, places=3)
        self.assertAlmostEqual(test22.xa + test22.xn + test22.xp, 1, places=3)

        # ensure Error is raised when invalid kwargs are passed
        with self.assertRaises(ValueError):
            SCNProperty(**{**kwargs, 'subtract': '-----'})
        with self.assertRaises(ValueError):
            SCNProperty(**{**kwargs, 'init_guess': {'----': 1}})
        with self.assertRaises(TypeError):
            SCNProperty(**{**kwargs, 'ma': 95})

        # ensure no error is raised:
        SCNProperty(**{**kwargs, 'model': 'kf'})
        SCNProperty(**{**kwargs, 'model': 'ra'})
        SCNProperty(**{**kwargs, 'init_guess': {'mw': 90, 'sg': 100, 'Tb': 600}})
        SCNProperty(**{**kwargs, 'subtract': 'naphthenes'})
        SCNProperty(**{**kwargs, 'subtract': 'aromatics'})
        SCNProperty(**{**kwargs, 'subtract': 'paraffins'})
        SCNProperty(**{**kwargs, 'round': 5})
        SCNProperty(**{**kwargs, 'warning': False})
        SCNProperty(**{**kwargs, 'warning': True})

        # ensure errors are raised when input values are out of range
        filler = SCNProperty(mw=90)
        models = filler.kwargs_options['model']
        for model in models:
            test4 = SCNProperty(mw=90, model=model)
            working_range = test4._range_warnings
            for key in working_range:
                lower_lim = {key: working_range[key][0] - 0.01}
                upper_lim = {key: working_range[key][1] + 0.01}  # 0.01 is an arbitrary value so the input exceeds working range slightly
                with self.assertWarns(SCNPropertyWarning):
                    SCNProperty(**upper_lim, model=model)
                with self.assertWarns(SCNPropertyWarning):
                    SCNProperty(**lower_lim, model=model)

        # when input values are significantly out of range, non-linear solver should fail with warning msg
        with self.assertRaises(RuntimeError):
            with self.assertWarns(SCNPropertyWarning):
                SCNProperty(**{**kwargs, 'sg': 90, 'mw': None})
        with self.assertRaises(RuntimeError):
            with self.assertWarns(SCNPropertyWarning):
                SCNProperty(**{**kwargs, 'Tb': 2000, 'mw': None})

        # ensure different calculation models are applied for MW>200 vs. MW<=200
        test6 = SCNProperty(**{**kwargs, 'mw': 201})
        self.assertIsNone(test6.VGF)
        self.assertIsNotNone(test6.SUS_100)
        self.assertIsNotNone(test6.VGC)
        test7 = SCNProperty(**{**kwargs, 'mw': 200})
        self.assertIsNotNone(test7.VGF)
        self.assertIsNone(test7.SUS_100)
        self.assertIsNone(test7.VGC)



        kwargs_table = {
            'col': 'mw',
            'output_keys': ['sg', 'Tb', 'xp', 'xn', 'xa', 'ri', 'v100', 'v210'],
        }

        # ensure no errors are raised
        test_table1 = temp = SCNProperty.build_table([i for i in range(620, 627, 1)], **kwargs_table, **kwargs)
        test_table2 = SCNProperty.build_table([90], **kwargs_table, **kwargs)

        # ensure the table is order in the order specified by the user in output_keys
        output_keys = kwargs_table['output_keys'].copy()
        for _ in range(3):
            random.shuffle(output_keys)
            kwargs_table['output_keys'] = output_keys
            test_table = SCNProperty.build_table([90], **kwargs_table, **kwargs)
            expected_columns = [kwargs_table['col']] + output_keys
            self.assertEqual(list(test_table.columns), expected_columns)

        # ensure all cells are filled
        empty_cells = test_table1.isnull().any().any()
        self.assertFalse(empty_cells, "There should be no empty cells in the DataFrame")

    def test_PropertyTable(self):

        # default kwargs
        SCN_kwargs = {
            'model': 'kf',
            'init_guess': None,
            'subtract': 'naphthenes',
            'round': None,
            'warning': True
        }
        kwargs = {
            'summary': True,
            'SCNProperty_kwargs': None,
        }

        # error triggered when wrong input keys for SCN_kwargs are used
        with self.assertRaises(TypeError):
            PropertyTable(sample_1, summary=True, SCNProperty_kwargs={**SCN_kwargs, 'asdasd': 200})

        # Tb, mw, sg values should not be provided as SCNProperty_kwargs because there can be multiple pseudos
        # The values for the pseudos are to be provided in other ways
        with self.assertRaises(TypeError):
            PropertyTable(sample_1, summary=True, SCNProperty_kwargs={**SCN_kwargs, 'Tb': 200})

        # checking index of the plus fraction
        ptable1a = PropertyTable(sample_1a, **kwargs)
        self.assertEqual(ptable1a.names_fraction, ['Hexanes+'])
        self.assertEqual(len(ptable1a.names_fraction), 1)
        self.assertEqual(ptable1a.compound_indices_dict[ptable1a.names_fraction[0]], 8)

        ################# new test ###########################
        ptable1b = PropertyTable(sample_1, **kwargs)
        idx_total = max(ptable1b.compound_indices_dict.values()) + 1  # total row is the last row

        """------------------------- Basic calculated value checks -------------------------"""
        # GHV_liq input shouldn't calculate other columns, as it requires GHV_Gas and MW to solve for other properties
        ptable1b = PropertyTable(sample_1, **kwargs)
        ptable1b.update_property('Hexanes+', {'GHV_liq': 20408})
        self.assertTrue(pd.isna(ptable1b.table.loc[idx_total, 'GHV_liq']))
        self.assertTrue(pd.isna(ptable1b.table.loc[idx_total, 'Mass_Fraction']))
        self.assertTrue(pd.isna(ptable1b.table.loc[0, 'Mass_Fraction']))

        ptable1c = PropertyTable(sample_1, **kwargs)
        ptable1c.update_property('Hexanes+', {'sg_gas': 3.0180})
        self.assertAlmostEqual(ptable1c.table.loc[idx_total, 'MW'], 23.378916, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total, 'Mass_Fraction'], 1.000000, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total, 'GHV_gas'], 1380.181508, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total, 'GHV_liq'], 22403.289082, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total, 'SG_gas'], 0.807207, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total, 'SG_liq'], 0.350053, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total - 1, 'MW'], 87.408825, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total - 1, 'Mass_Fraction'], 0.016638, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total - 1, 'GHV_gas'], 4778.214334, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total - 1, 'GHV_liq'], 20744.548056, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total - 1, 'SG_gas'], 3.018000, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total - 1, 'SG_liq'], 0.697077, places=3)

        ptable1d = PropertyTable(sample_1, **kwargs)
        ptable1d.update_property('Hexanes+', {'SG liq': 0.6933})
        self.assertAlmostEqual(ptable1d.table.loc[idx_total, 'MW'], 23.373998, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total, 'Mass_Fraction'], 1.000000, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total, 'GHV_gas'], 1379.939882, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total, 'GHV_liq'], 22404.080005, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total, 'SG_gas'], 0.807037, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total, 'SG_liq'], 0.350036, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total - 1, 'MW'], 86.303649, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total - 1, 'Mass_Fraction'], 0.016431, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total - 1, 'GHV_gas'], 4723.916421, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total - 1, 'GHV_liq'], 20771.443629, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total - 1, 'SG_gas'], 2.979841, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total - 1, 'SG_liq'], 0.693300, places=3)

        ptable1e = PropertyTable(sample_1, **kwargs)
        ptable1e.update_property('Hexanes+', {'ghv gas': 4774.1})
        self.assertAlmostEqual(ptable1e.table.loc[idx_total, 'MW'], 23.378544, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total, 'Mass_Fraction'], 1.000000, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total, 'GHV_gas'], 1380.163199, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total, 'GHV_liq'], 22403.348666, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total, 'SG_gas'], 0.807194, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total, 'SG_liq'], 0.350051, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total - 1, 'MW'], 87.325161, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total - 1, 'Mass_Fraction'], 0.016622, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total - 1, 'GHV_gas'], 4774.100000, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total - 1, 'GHV_liq'], 20746.543528, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total - 1, 'SG_gas'], 3.015111, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total - 1, 'SG_liq'], 0.696796, places=3)

        ptable1f = PropertyTable(sample_1, **kwargs)
        ptable1f.update_property('Hexanes+', {'MW': 87.665})
        self.assertAlmostEqual(ptable1f.table.loc[idx_total, 'MW'], 23.380056, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total, 'Mass_Fraction'], 1.000000, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total, 'GHV_gas'], 1380.237584, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total, 'GHV_liq'], 22403.106907, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total, 'SG_gas'], 0.807246, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total, 'SG_liq'], 0.350056, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total - 1, 'MW'], 87.665000, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total - 1, 'Mass_Fraction'], 0.016686, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total - 1, 'GHV_gas'], 4790.815715, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total - 1, 'GHV_liq'], 20738.477101, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total - 1, 'SG_gas'], 3.026845, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total - 1, 'SG_liq'], 0.697933, places=3)

        ptable1g = PropertyTable(sample_1, **kwargs)
        ptable1g.update_property('Hexanes+', {'Mass_Fraction': 0.016686})
        self.assertTrue(pd.isna(ptable1g.table.loc[idx_total, 'GHV_liq']))
        self.assertTrue(pd.isna(ptable1g.table.loc[idx_total, 'Mass_Fraction']))
        self.assertTrue(pd.isna(ptable1g.table.loc[0, 'Mass_Fraction']))

        # sample 2 testing - has no fractions. Everything should solve without providing Plus fraction updates
        ptable2 = PropertyTable(sample_2, **kwargs)
        idx_total = max(ptable2.compound_indices_dict.values()) + 1
        self.assertFalse(ptable2.table.drop(columns=['CAS']).isnull().values.any())
        self.assertAlmostEqual(ptable2.table.loc[idx_total, 'SG_gas'], 1.404593, places=3)
        self.assertAlmostEqual(ptable2.table.loc[idx_total, 'SG_liq'], 0.478981, places=3)
        self.assertAlmostEqual(ptable2.table.loc[idx_total, 'MW'], 40.680797, places=3)

        """------------------------- Test calculation priority for MW -------------------------"""

        # the values are computed in the orders of
        # list specified in _internal_update_property.rules.correlations.required_columns
        mw = 87.655
        sg_gas = 3.026057
        sg_liq = 0.697899
        ghv_gas = 4790.323699

        ptable1h = PropertyTable(sample_1, **kwargs)
        ptable1h.update_property('Hexanes+', {'MW': mw})
        mw_from_mw = ptable1h.table.loc[9, 'MW']

        ptable1h = PropertyTable(sample_1, **kwargs)
        ptable1h.update_property('Hexanes+', {'sg_gas': sg_gas})
        mw_from_sg_gas = ptable1h.table.loc[9, 'MW']

        ptable1h = PropertyTable(sample_1, **kwargs)
        ptable1h.update_property('Hexanes+', {'SG_liq': sg_liq})
        mw_from_sg_liq = ptable1h.table.loc[9, 'MW']

        ptable1h = PropertyTable(sample_1, **kwargs)
        ptable1h.update_property('Hexanes+', {'GHV gas': ghv_gas})
        mw_from_ghv_gas = ptable1h.table.loc[9, 'MW']

        ptable1h = PropertyTable(sample_1, **kwargs)
        ptable1h.update_property('Hexanes+', {'MW': mw, 'SG_gas': sg_gas})
        mw_from_this = ptable1h.table.loc[9, 'MW']
        self.assertAlmostEqual(mw_from_this, mw_from_mw, places=3)

        ptable1h = PropertyTable(sample_1, **kwargs)
        ptable1h.update_property('Hexanes+', {'SG_gas': sg_gas, 'SG_liq': sg_liq})
        mw_from_this = ptable1h.table.loc[9, 'MW']
        self.assertAlmostEqual(mw_from_this, mw_from_sg_gas, places=3)

        ptable1h = PropertyTable(sample_1, **kwargs)
        ptable1h.update_property('Hexanes+', {'SG_gas': sg_gas, 'GHV_gas': ghv_gas})
        mw_from_this = ptable1h.table.loc[9, 'MW']
        self.assertAlmostEqual(mw_from_this, mw_from_sg_gas, places=3)

        """--------------------------- testing for 2 fractions scenario ---------------------------"""

        ptable1b = PropertyTable(sample_1b, **kwargs)
        idx_total = max(ptable1b.compound_indices_dict.values()) + 1

        self.assertEqual(ptable1b.names_fraction, ['Hexanes+', 'Heptanes+'])
        self.assertEqual(len(ptable1b.names_fraction), 2)
        self.assertEqual(ptable1b.compound_indices_dict[ptable1b.names_fraction[0]], 9)
        self.assertEqual(ptable1b.compound_indices_dict[ptable1b.names_fraction[1]], 10)

        # No Nones, instead I implemented np.nan (NaN)

        has_None = ptable1b.table.isin([None]).any().any()
        self.assertFalse(has_None, "DataFrame contains 'None' values, which is not allowed.")

        # since there are two fractions, providing values for only 1 fraction shouldn't calculate total properties
        ptable1b = PropertyTable(sample_1b, **kwargs)
        ptable1b.update_property('Hexanes+', {'MW': 87.665})
        c7_idx = ptable1b.compound_indices_dict['Heptanes+']
        self.assertTrue(pd.isna(ptable1b.table.loc[idx_total, 'GHV_liq']))
        self.assertTrue(pd.isna(ptable1b.table.loc[idx_total, 'MW']))
        self.assertTrue(pd.isna(ptable1b.table.loc[idx_total, 'Mass_Fraction']))
        self.assertTrue(pd.isna(ptable1b.table.loc[idx_total, 'GHV_gas']))
        self.assertTrue(pd.isna(ptable1b.table.loc[idx_total, 'SG_liq']))
        self.assertTrue(pd.isna(ptable1b.table.loc[idx_total, 'SG_gas']))
        self.assertTrue(pd.isna(ptable1b.table.loc[c7_idx, 'GHV_liq']))
        self.assertTrue(pd.isna(ptable1b.table.loc[c7_idx, 'MW']))
        self.assertTrue(pd.isna(ptable1b.table.loc[c7_idx, 'Mass_Fraction']))
        self.assertTrue(pd.isna(ptable1b.table.loc[c7_idx, 'GHV_gas']))
        self.assertTrue(pd.isna(ptable1b.table.loc[c7_idx, 'SG_liq']))
        self.assertTrue(pd.isna(ptable1b.table.loc[c7_idx, 'SG_gas']))

        ptable1b = PropertyTable(sample_1b, **kwargs)
        ptable1b.update_property('Hexanes+', {'MW': 87.665})
        ptable1b.update_property('Heptanes+', {'MW': 90})

        # new composition of bigger mole fraction of the plus fractions are tested to verify calculation
        # in multiple fractions scenario. The SG_gas and SG_liq values are cross-checked with promax outputs.
        test = dict([
            ('methane', 40),
            ('Hexanes+', 30),
            ('Heptanes+', 30),  # 0.743
        ])
        test_table = PropertyTable(test, **kwargs)
        test_table.update_property('Hexanes+', {'MW': 87.665})
        test_table.update_property('Heptanes+', {'MW': 90})
        idx_total = max(test_table.compound_indices_dict.values()) + 1
        self.assertAlmostEqual(test_table.table.loc[idx_total, 'SG_gas'], 2.061377, places=3)
        self.assertAlmostEqual(test_table.table.loc[idx_total, 'SG_liq'], 0.541002, places=3)

        """--------------------------- 6:3:1 Ternary mixture for Hexanes+ ---------------------------"""

        test = dict([
            ('n-hexane', 60),
            ('heptane', 30),
            ('n-octane', 10),
        ])
        test_table = PropertyTable(test, **kwargs)
        self.assertAlmostEqual(test_table.table.loc[idx_total, 'SG_gas'], 3.21755, places=3)
        self.assertAlmostEqual(test_table.table.loc[idx_total, 'SG_liq'], 0.67556, places=3)
        self.assertAlmostEqual(test_table.table.loc[idx_total, 'GHV_gas'], 5129.22, places=3)

        # Properties of ambient air MW = 28.9625. Composition originate from Table A.1 of GPA 2172-19
        # Demonstration to show that the library supports non-GPA 2145 components as well
        air_table = PropertyTable(air, **kwargs)
        idx_total = max(air_table.compound_indices_dict.values()) + 1
        self.assertAlmostEqual(air_table.table.loc[idx_total, 'MW'], 28.962562, places=3)

        """--------------------------- update_total tests with 1 property ---------------------------"""

        update_total_kwargs = {
            'target_compound': None,
            'recalc': True,
        }
        #for mw in [i for i in range(86, 600, 5)]:
        for mw in [i for i in range(199, 201, 1)]:  # Todo: later replace this with the above for actual implementation

            ptable3 = PropertyTable(sample_3, **kwargs)
            ptable3.update_property('Hexanes+', {'MW': mw})

            idx_plusFrac = ptable3.compound_indices_dict['Hexanes+']
            idx_total = max(ptable3.compound_indices_dict.values()) + 1

            mw_total = ptable3.table.loc[idx_total, 'MW']
            mw_from_update_property = ptable3.table.loc[idx_plusFrac, 'MW']
            ghv_gas_from_update_property = ptable3.table.loc[idx_plusFrac, 'GHV_gas']
            ghv_liq_from_update_property = ptable3.table.loc[idx_plusFrac, 'GHV_liq']
            sg_gas_from_update_property = ptable3.table.loc[idx_plusFrac, 'SG_gas']
            sg_liq_from_update_property = ptable3.table.loc[idx_plusFrac, 'SG_liq']

            ptable3a = PropertyTable(sample_3, **kwargs)
            ptable3a.update_total({'mw': mw_total}, **update_total_kwargs)

            self.assertAlmostEqual(ptable3a.table.loc[idx_plusFrac, 'GHV_gas'], ghv_gas_from_update_property, places=1)
            self.assertAlmostEqual(ptable3a.table.loc[idx_plusFrac, 'GHV_liq'], ghv_liq_from_update_property, places=1)
            self.assertAlmostEqual(ptable3a.table.loc[idx_plusFrac, 'SG_gas'], sg_gas_from_update_property, places=1)
            self.assertAlmostEqual(ptable3a.table.loc[idx_plusFrac, 'SG_liq'], sg_liq_from_update_property, places=1)
            self.assertAlmostEqual(ptable3a.table.loc[idx_plusFrac, 'MW'], mw_from_update_property, places=1)

        """--------------------------- consecutive update test for update_total ---------------------------"""

        test_comp = {
            'methane': 50,
            'Test+': 50,
        }
        #ptable5a = PropertyTable(test_comp, **{**kwargs, 'warning': False})
        ptable5a = PropertyTable(test_comp, **kwargs)
        idx_test_plus = ptable5a.compound_indices_dict['Test+']
        ptable5a.update_total({'mw': 23.396726}, warning=False)

        self.assertAlmostEqual(ptable5a.table.loc[idx_test_plus, 'MW'], 30.750952, places=3)
        self.assertAlmostEqual(ptable5a.table.loc[idx_test_plus, 'GHV_gas'], 1782.391064, places=3)

        ptable5b = PropertyTable(test_comp, **kwargs)
        ptable5b.update_total({'mw': 23.396726}, warning=False)
        ptable5b.update_total({'mw': 50}, warning=False)
        self.assertEqual(ptable5b.target_compound, 'Test+')
        self.assertAlmostEqual(ptable5b.table.loc[idx_test_plus, 'MW'], 83.9575, places=3)
        self.assertAlmostEqual(ptable5b.table.loc[idx_test_plus, 'GHV_gas'], 4608.985487, places=3)
        self.assertAlmostEqual(ptable5b.table.loc[idx_test_plus + 1, 'MW'], 50, places=3)

        """--------------------------- update_total tests with 2 properties ---------------------------"""

        ptable4 = PropertyTable(sample_1b, **kwargs)
        ptable4.update_property('Hexanes+', {'MW': 87.665})
        ptable4.update_property('Heptanes+', {'MW': 96})

        idx_hexanes = ptable4.compound_indices_dict['Hexanes+']
        idx_heptanes = ptable4.compound_indices_dict['Heptanes+']
        idx_total = max(ptable4.compound_indices_dict.values()) + 1

        mw_total = ptable4.table.loc[idx_total, 'MW']
        mw_hexanes_from_update_property = ptable4.table.loc[idx_hexanes, 'MW']
        mw_heptanes_from_update_property = ptable4.table.loc[idx_heptanes, 'MW']

        ghv_gas_hexanes_from_update_property = ptable4.table.loc[idx_hexanes, 'GHV_gas']
        ghv_liq_hexanes_from_update_property = ptable4.table.loc[idx_hexanes, 'GHV_liq']
        sg_gas_hexanes_from_update_property = ptable4.table.loc[idx_hexanes, 'SG_gas']
        sg_liq_hexanes_from_update_property = ptable4.table.loc[idx_hexanes, 'SG_liq']
        ghv_gas_heptanes_from_update_property = ptable4.table.loc[idx_heptanes, 'GHV_gas']
        ghv_liq_heptanes_from_update_property = ptable4.table.loc[idx_heptanes, 'GHV_liq']
        sg_gas_heptanes_from_update_property = ptable4.table.loc[idx_heptanes, 'SG_gas']
        sg_liq_heptanes_from_update_property = ptable4.table.loc[idx_heptanes, 'SG_liq']

        ptable4a = PropertyTable(sample_1b, **kwargs)
        ptable4a.update_property('Hexanes+', {'MW': 87.665})
        ptable4a.update_total({'mw': mw_total}, **{**update_total_kwargs})

        self.assertAlmostEqual(ptable4a.table.loc[idx_hexanes, 'GHV_gas'], ghv_gas_hexanes_from_update_property, places=1)
        self.assertAlmostEqual(ptable4a.table.loc[idx_hexanes, 'GHV_liq'], ghv_liq_hexanes_from_update_property, places=1)
        self.assertAlmostEqual(ptable4a.table.loc[idx_hexanes, 'SG_gas'], sg_gas_hexanes_from_update_property, places=1)
        self.assertAlmostEqual(ptable4a.table.loc[idx_hexanes, 'SG_liq'], sg_liq_hexanes_from_update_property, places=1)
        self.assertAlmostEqual(ptable4a.table.loc[idx_hexanes, 'MW'], mw_hexanes_from_update_property, places=1)
        self.assertAlmostEqual(ptable4a.table.loc[idx_heptanes, 'GHV_gas'], ghv_gas_heptanes_from_update_property, places=1)
        self.assertAlmostEqual(ptable4a.table.loc[idx_heptanes, 'GHV_liq'], ghv_liq_heptanes_from_update_property, places=1)
        self.assertAlmostEqual(ptable4a.table.loc[idx_heptanes, 'SG_gas'], sg_gas_heptanes_from_update_property, places=1)
        self.assertAlmostEqual(ptable4a.table.loc[idx_heptanes, 'SG_liq'], sg_liq_heptanes_from_update_property, places=1)
        self.assertAlmostEqual(ptable4a.table.loc[idx_heptanes, 'MW'], mw_heptanes_from_update_property, places=1)

        # test if Heptanes+ fraction is adjusted to match the target total mw=24
        ptable4b = PropertyTable(sample_1b, **kwargs)
        ptable4b.update_property('Hexanes+', {'MW': 87.665})
        ptable4b.update_property('Heptanes+', {'MW': 90})
        ptable4b.update_total({'mw': 24}, **{**update_total_kwargs, 'target_compound': 'Heptanes+'})
        idx_hexanes = ptable4b.compound_indices_dict['Hexanes+']
        idx_heptanes = ptable4b.compound_indices_dict['Heptanes+']
        self.assertAlmostEqual(ptable4b.table.loc[idx_hexanes, 'MW'], 87.665, places=3)
        self.assertAlmostEqual(ptable4b.table.loc[idx_heptanes, 'MW'], 397.636783, places=3)

        # test if Hexanes+ fraction is adjusted to match the target total mw=24
        ptable4c = PropertyTable(sample_1b, **kwargs)
        ptable4c.update_property('Hexanes+', {'MW': 87.665})
        ptable4c.update_property('Heptanes+', {'MW': 90})
        ptable4c.update_total({'mw': 24}, **{**update_total_kwargs, 'target_compound': 'Hexanes+'})
        idx_hexanes = ptable4c.compound_indices_dict['Hexanes+']
        idx_heptanes = ptable4c.compound_indices_dict['Heptanes+']
        self.assertAlmostEqual(ptable4c.table.loc[idx_hexanes, 'MW'], 338.797067, places=3)
        self.assertAlmostEqual(ptable4c.table.loc[idx_heptanes, 'MW'], 90.000000, places=3)

        # warning should be raised because target_compound is not provided
        with self.assertWarns(PropertyTableWarning):
            ptable4d = PropertyTable(sample_1b, **kwargs)
            ptable4d.update_property('Hexanes+', {'MW': 87.665})
            ptable4d.update_property('Heptanes+', {'MW': 90})
            ptable4d.update_total({'mw': mw_total}, **{**update_total_kwargs})

        # when target_compound is not provided, the first fraction is adjusted to match the target mw
        self.assertEqual(ptable4d.target_compound, ptable4d.names_fraction[0])

        # Ensure that error is triggered when there are more than 1 unkwowns.
        # In this case, Hexanes+ and Heptanes+ are not provided
        with self.assertRaises(ValueError):
            ptable4a = PropertyTable(sample_1b, **kwargs)
            ptable4a.update_total({'mw': mw_total}, **{**update_total_kwargs})

        """------------------- update_total tests with 3 properties and 2 consecutive updates -------------------"""

        # Todo: finish this section

        test = dict([
            ('methane', 40),
            ('Hexanes+', 30),
            ('Heptanes+', 20),
            ('Nonanes+', 10),
        ])

        # one unknown
        ptable5a = PropertyTable(test, **kwargs)
        ptable5a.update_property('Hexanes+', {'MW': 87.665})
        ptable5a.update_property('Heptanes+', {'MW': 90})
        idx_hexanes = ptable5a.compound_indices_dict['Hexanes+']
        idx_heptanes = ptable5a.compound_indices_dict['Heptanes+']
        idx_nonanes = ptable5a.compound_indices_dict['Nonanes+']

        self.assertEqual(ptable5a.target_compound, None)
        ptable5a.update_total({'mw': 100}, **{**update_total_kwargs})
        self.assertAlmostEqual(ptable5a.table.loc[idx_hexanes, 'MW'], 87.665, places=3)
        self.assertAlmostEqual(ptable5a.table.loc[idx_heptanes, 'MW'], 90, places=3)
        self.assertAlmostEqual(ptable5a.table.loc[idx_nonanes, 'MW'], 492.8350, places=3)
        # target compound should be automatically set as Nonanes+ since it's the only compound with unknowns
        self.assertEqual(ptable5a.target_compound, 'Nonanes+')

        ptable5a.update_total({'mw': 110}, **{**update_total_kwargs})
        self.assertEqual(ptable5a.target_compound, 'Nonanes+')
        self.assertAlmostEqual(ptable5a.table.loc[idx_nonanes, 'MW'], 592.8350, places=3)

        # two unknowns (2/3).
        with self.assertRaises(ValueError):
            ptable5b = PropertyTable(test, **kwargs)
            ptable5b.update_property('Hexanes+', {'MW': 87.665})
            ptable5b.update_total({'mw': 110}, **{**update_total_kwargs})

        # 0 unknown - scneario 1
        ptable5c = PropertyTable(test, **kwargs)
        ptable5c.update_property('Hexanes+', {'MW': 87.665})
        ptable5c.update_property('Heptanes+', {'MW': 90})
        idx_total = max(ptable5c.compound_indices_dict.values()) + 1

        # Total row should be NaN because 2/3 fractions are provided
        for col in [item for item in ptable5c.table.columns if item not in ['Name', 'CAS', 'Mole_Fraction']]:
            self.assertTrue(pd.isna(ptable5c.table.loc[idx_total, col]))

        # providing the last fraction should calculate the total row
        ptable5c.update_property('Nonanes+', {'MW': 95})
        for col in [item for item in ptable5c.table.columns if item not in ['Name', 'CAS', 'Mole_Fraction']]:
            self.assertFalse(pd.isna(ptable5c.table.loc[idx_total, col]))

        # ambiguous warning because target_compound is not specified yet.
        with self.assertWarns(PropertyTableWarning):
            ptable5c.update_total({'MW': 300})  # this line sets target_compound to Hexanes+, which is 0th index fraction
            idx_hexanes = ptable5c.compound_indices_dict['Hexanes+']
            self.assertAlmostEqual(ptable5c.table.loc[idx_hexanes, 'MW'], 886.943333, places=3)

        # 0 unknown - scneario 1, warning should be triggered
        ptable5d = PropertyTable(test, **kwargs)
        ptable5d.update_property('Hexanes+', {'MW': 87.665})
        ptable5d.update_property('Heptanes+', {'MW': 90})
        with self.assertWarns(SCNPropertyWarning):
            ptable5d.update_total({'MW': 300})

        # warning should not be triggered
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            ptable5d.update_total({'sg liq': 300}, warning=False)
            self.assertFalse(any(item.category == PropertyTableWarning for item in caught_warnings), "PropertyTableWarning was triggered but expected not to.")
            warnings.resetwarnings()

        # 0 unknown - scneario 2
        ptable5d = PropertyTable(test, **kwargs)
        ptable5d.update_property('Hexanes+', {'MW': 87.665})
        ptable5d.update_property('Nonanes+', {'MW': 90})
        ptable5d.update_total({'MW': 100})
        self.assertEqual(ptable5d.target_compound, 'Heptanes+')
        self.assertAlmostEqual(ptable5d.table.loc[idx_heptanes, 'MW'], 291.4175, places=3)
        ptable5d.update_total({'MW': 120})
        self.assertAlmostEqual(ptable5d.table.loc[idx_heptanes, 'MW'], 391.4175, places=3)

        """--------------------------- recalc testing ---------------------------"""
        ptable5e = PropertyTable(test, **{**kwargs})
        ptable5e.update_property('Hexanes+', {'MW': 87.665})
        ptable5e.update_property('Nonanes+', {'MW': 90})
        ptable5e.update_total({'MW': 100}, recalc=True)
        idx_total = max(ptable5e.compound_indices_dict.values()) + 1
        idx_heptanes = ptable5e.compound_indices_dict['Heptanes+']
        ghv_gas_total = ptable5e.table.loc[idx_total, 'GHV_gas']
        mw_heptanes = ptable5e.table.loc[idx_heptanes, 'MW']
        self.assertAlmostEqual(mw_heptanes, 291.4175, places=3)

        ptable5e.update_total({'MW': 200}, recalc=False, warning=False)
        ghv_gas_total_200_false = ptable5e.table.loc[idx_total, 'GHV_gas']
        mw_heptanes_false = ptable5e.table.loc[idx_heptanes, 'MW']
        self.assertAlmostEqual(ghv_gas_total, ghv_gas_total_200_false, places=3)
        self.assertNotEqual(mw_heptanes, mw_heptanes_false)
        self.assertAlmostEqual(mw_heptanes_false, 791.4175, places=3)

        ptable5e.update_total({'MW': 200}, recalc=True, warning=False)
        ghv_gas_total_200_true = ptable5e.table.loc[idx_total, 'GHV_gas']
        mw_heptanes_true = ptable5e.table.loc[idx_heptanes, 'MW']
        self.assertNotEqual(ghv_gas_total, ghv_gas_total_200_true)
        self.assertAlmostEqual(ghv_gas_total_200_true, 10777.928280, places=3)
        self.assertNotEqual(mw_heptanes, mw_heptanes_true)
        self.assertAlmostEqual(mw_heptanes_true, 791.4175, places=3)

        """--------------------------- warning testing ---------------------------"""

        ptable6a = PropertyTable(test, **kwargs)
        ptable6a.update_property('Hexanes+', {'MW': 87.665})
        ptable6a.update_property('Nonanes+', {'MW': 90})

        with self.assertWarns(SCNPropertyWarning):
            ptable6a.update_total({'MW': 300}, warning=True)

        # test for three executions of internal_property. Duplicate warnings should be suppressed
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            ptable6a.update_total({'MW': 300}, warning=True)  # each execution should show 3 warnings for MW, Tb, sg_liq
            ptable6a.update_total({'MW': 300}, warning=True)
            scn_property_warnings = [warning for warning in caught_warnings if issubclass(warning.category, SCNPropertyWarning)]
            self.assertEqual(len(scn_property_warnings), 6, f"Expected 6 warnings, got {len(scn_property_warnings)}")

        # warning kwargs in update_total should override the class level warning
        with self.assertWarns(SCNPropertyWarning):
            ptable6a.update_total({'MW': 300}, warning=True)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            ptable6a.update_total({'MW': 300}, warning=False)
            scn_warnings = [w for w in caught_warnings if issubclass(w.category, SCNPropertyWarning)]
            self.assertEqual(len(scn_warnings), 0, "Expected no SCNPropertyWarning, but some were raised.")

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            ptable6a.update_property('Hexanes+', {'MW': 90}, warning=False)
            scn_warnings = [w for w in caught_warnings if issubclass(w.category, SCNPropertyWarning)]
            self.assertEqual(len(scn_warnings), 0, "Expected no SCNPropertyWarning, but some were raised.")

        # duplicate warning check for PropertyTableWarning when runtime error occurs due to sg_liq exceeding ~1.1 limits
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            ptable6a = PropertyTable(test, **kwargs)
            ptable6a.update_property('Hexanes+', {'MW': 87.665})
            ptable6a.update_property('Nonanes+', {'MW': 90})
            ptable6a.update_property('Heptanes+', {'sg_liq': 95})
            scn_property_warnings = [warning for warning in caught_warnings if
                                     issubclass(warning.category, SCNPropertyWarning)]
            property_table_warnings = [warning for warning in caught_warnings if
                                     issubclass(warning.category, PropertyTableWarning)]
            self.assertEqual(len(scn_property_warnings), 1,
                             f"Expected 6 warnings, got {len(scn_property_warnings)}")
            self.assertEqual(len(property_table_warnings), 1,
                             f"Expected 6 warnings, got {len(property_table_warnings)}")

        # Testing for un-normalized composition warnings
        test_comp = {
            'methane': 50,
            'ethane': 10,
        }
        with self.assertWarns(PropertyTableWarning):
            PropertyTable(test_comp, **{**kwargs, 'warning': True})
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            PropertyTable(test_comp, **{**kwargs, 'warning': False})
            self.assertEqual(len(caught_warnings), 0, "Expected no warning, but some were raised.")

        # heneicosane and docosane are recognized in Thermo but are missing Hc values needed for GHV calc.
        test = {
            'heneicosane': 30,
            'docosane': 30,
            'heptadecane': 40
        }
        with self.assertWarns(ThermoMissingValueWarning):
            PropertyTable(test, **kwargs, warning=True)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            PropertyTable(test, **kwargs, warning=False)
            self.assertEqual(len(caught_warnings), 0, "Expected no warning, but some were raised.")

        test_comp = {
            'methane': 50,
            'n-C50': 50,
        }
        with self.assertWarns(SCNPropertyWarning):
            PropertyTable(test_comp, **{**kwargs, 'warning': True})
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            PropertyTable(test_comp, **{**kwargs, 'warning': False})
            self.assertEqual(len(caught_warnings), 0, "Expected no warning, but some were raised.")

        """--------------------------- normalization testing ---------------------------"""

        keys = [
            'hydrogen sulfide', 'nitrogen', 'carbon dioxide', 'methane', 'ethane',
            'propane', 'isobutane', 'n-butane', 'i-pentane', 'n-pentane', 'Hexanes+'
        ]
        samples = [{key: random.uniform(0, 9999) for key in keys} for _ in range(100)]
        for sample in samples:
            normalized_frac = list(PropertyTable(sample, warning=False).comp_dict.values())
            self.assertEqual(sum(normalized_frac), 1.0)

        """--------------------------- test for changing known compound properties ---------------------------"""

        test = {
            'methane': 50,
            'ethane': 10,
            'Hexanes+': 30,
            'Something+': 10,
        }

        # scenario 1: recalc = True
        ptable7 = PropertyTable(test, **kwargs)
        ptable7.update_property('Hexanes+', {'MW': 87.665})
        ptable7.update_property('Something+', {'MW': 600})
        idx_methane = ptable7.compound_indices_dict['methane']
        idx_total = max(ptable7.compound_indices_dict.values()) + 1
        self.assertAlmostEqual(ptable7.table.loc[idx_methane, 'MW'], 16.04245, places=3)
        self.assertAlmostEqual(ptable7.table.loc[idx_methane, 'GHV_gas'], 1010.0, places=3)
        self.assertAlmostEqual(ptable7.table.loc[idx_total, 'MW'], 97.32765, places=3)
        self.assertAlmostEqual(ptable7.table.loc[idx_total, 'GHV_gas'], 5322.388471, places=3)
        ptable7.update_property('methane', {'MW': 50}, warning=False)
        self.assertEqual(ptable7.table.loc[idx_methane, 'MW'], 50)
        self.assertAlmostEqual(ptable7.table.loc[idx_methane, 'GHV_gas'], 2900.394909, places=3)
        self.assertAlmostEqual(ptable7.table.loc[idx_total, 'MW'], 114.3064, places=3)
        self.assertAlmostEqual(ptable7.table.loc[idx_total, 'GHV_gas'], 6267.585926, places=3)

        # scenario 2: recalc = False, the columns for the compound doesn't update. The total for the column updates tho.
        ptable7 = PropertyTable(test, **kwargs)
        ptable7.update_property('Hexanes+', {'MW': 87.665})
        ptable7.update_property('Something+', {'MW': 600})
        ptable7.update_property('methane', {'MW': 50}, warning=False, recalc=False)
        self.assertAlmostEqual(ptable7.table.loc[idx_methane, 'GHV_gas'], 1010.0, places=3)
        self.assertAlmostEqual(ptable7.table.loc[idx_total, 'MW'], 114.3064, places=3)
        self.assertAlmostEqual(ptable7.table.loc[idx_total, 'GHV_gas'], 5322.388471, places=3)

        """--------------------------- test for identified but missing properties ---------------------------"""

        # Heneicosane and docosane don't have Hc values, therefore GHV_gas is approximated with correlation
        test = {
            'heneicosane': 30,
            'docosane': 30,
            'heptadecane': 40
        }
        ptable8 = PropertyTable(test, **{**kwargs, 'SCNProperty_kwargs': {'model': 'kf'}}, warning=False)
        heneicosane_idx = ptable8.compound_indices_dict['heneicosane']
        docosane_idx = ptable8.compound_indices_dict['docosane']
        heptadecane_idx = ptable8.compound_indices_dict['heptadecane']

        heneicosane_mw = ptable8.table.loc[heneicosane_idx, 'MW']
        docosane_mw = ptable8.table.loc[docosane_idx, 'MW']
        heptadecane_mw = ptable8.table.loc[heptadecane_idx, 'MW']

        xa_heneicosane = SCNProperty(mw=heneicosane_mw, model='kf').xa
        xa_docosane = SCNProperty(mw=docosane_mw, model='kf').xa
        xa_heptadecane = SCNProperty(mw=heptadecane_mw, model='kf').xa

        ghv_heneicosane = newton(lambda ghv: correlations.mw_ghv(heneicosane_mw, ghv, xa_heneicosane), x0=15000)
        ghv_docosane = newton(lambda ghv: correlations.mw_ghv(docosane_mw, ghv, xa_docosane), x0=15000)
        ghv_heptadecane = newton(lambda ghv: correlations.mw_ghv(heptadecane_mw, ghv, xa_heptadecane), x0=15000)

        self.assertAlmostEqual(ptable8.table.loc[heneicosane_idx, 'GHV_gas'], ghv_heneicosane, places=3)
        self.assertAlmostEqual(ptable8.table.loc[docosane_idx, 'GHV_gas'], ghv_docosane, places=3)
        self.assertNotAlmostEqual(ptable8.table.loc[heptadecane_idx, 'GHV_gas'], ghv_heptadecane, places=3)

        """--------------------------- test for unidentified compounds ---------------------------"""

        # refer to is_fraction() for detection conditions for plus fractions
        test = {
            'AA': 100
        }
        with self.assertRaises(ValueError):
            PropertyTable(test, **kwargs)
        test['AA+'] = test.pop('AA')
        PropertyTable(test, **kwargs)
        test['AAplus'] = test.pop('AA+')
        PropertyTable(test, **kwargs)
        test['AAPlus'] = test.pop('AAplus')
        PropertyTable(test, **kwargs)
        test['fraction'] = test.pop('AAPlus')
        PropertyTable(test, **kwargs)
        test['Fraction'] = test.pop('fraction')
        PropertyTable(test, **kwargs)
        test['fractions'] = test.pop('Fraction')
        ptable9 = PropertyTable(test, **kwargs)
        self.assertEqual(ptable9.names_fraction, ['fractions'])

        """-------------------------- Test for GHVs of GPA values vs. Thermo Hcs --------------------------"""

        # ensure that the GHV_gas calculated from Thermo's Hcs agree with GPA values within 3% tolerance
        ptable10 = PropertyTable(sample_2)
        ghvs_gpa = ptable10.table_['GHV_gas'].values

        components = list(sample_2.keys())
        for i, component in enumerate(components):
            constants = ChemicalConstantsPackage.constants_from_IDs([component])
            v_molar = utilities.ideal_gas_molar_volume()
            Hc = constants.Hcs[0]
            ghv_gas = Hc / v_molar
            ghv_gas = UREG('%.15f joule/m^3' % ghv_gas).to('Btu/ft^3')._magnitude * -1
            ghv_gas_gpa = ghvs_gpa[i]

            value1 = ghv_gas
            value2 = ghv_gas_gpa
            tolerance = 0.03

            self.assertTrue(value2 * (1 - tolerance) <= value1 <= value2 * (1 + tolerance))

        """--------------------------- test summary=False ---------------------------"""
        # Todo: lets remove summary=False and instead add documentation for self.table, self.table_, and self.summary

        """--------------------------- test for heavy components ---------------------------"""

        test = {
            'heneicosane': 30,
            'docosane': 30,
            'heptadecane': 40
        }
        ptable10 = PropertyTable(test, **{**kwargs, 'SCNProperty_kwargs': {'model': 'kf'}}, warning=False)

        print(ptable10.table.to_string())

        # Notes: The Tb values from Thermo seems to be reliable, agrees with Promax environtment
        # Notes: Pc from Thermo is unreliable for heavies
        # Notes: Promax Pc points for individual components seem to fluctuate up and down: C30~C33 confirmed

        # Todo: check accuracy of the Tc values in Thermo. Construct a dict from GPA components, and compare the
        #  Tcs side by side
        # Todo: implement liquid_density_method options: COSTALD, Thermo
        # Todo: think about how to implement GPA data override - Let's use GPA override only for GHV.

        # Todo: ptable10 = PropertyTable(sample_2, summary=False) this failed

        # Todo: Check Billingsley and labm paper to check how Pc and Tc are calculated.
        #  - This seems to be focused on mixture property and requires iterations.
        #  - Not suitable for verifying single compound sg_liq

        # Todo: check validity of sg_liq and sg_gas for heavy components. Perhaps sg_liq needs to be reverse-calculated
        #  with Pc, Tc and Tb correlations. For compounds
        #  The COSTALD rule works for individual components, agrees with GPA table
        #  But the current mol-frac average SG_liq doesn't agree with Promax for combined mixture

        # Todo: Investigate COSTALD molar volume methods with Promax for sg_liq60 inconsistencies in heavy components.

        # Todo: add custom chemical name for n-heptane

        # Todo: add Tb column on the returned DF, make the total Tb as NA because its essentially a bubble point

        # Todo: this column testing needs to be repeated for all test cases

        # ensure that all numeric columns have dtype=float64
        df2 = ptable2.table
        exclude_columns = ['Name', 'CAS']
        for column in df2.columns.difference(exclude_columns):
            self.assertTrue(df2[column].dtype == np.float64, f"Column '{column}' is not of dtype float64.")


        # Todo: testing section for GPA vs. non-GPA values


        #ptable.update_property('Hexanes+', {'mw': 95})



if __name__ == '__main__':
    unittest.main()


# Documents\GasCompressibiltiyFactor-py>python -m unittest tests.test_gascomp
# python -m unittest discover .
