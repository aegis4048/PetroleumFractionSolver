import unittest
import sys
import random
import numpy as np
import pandas as pd

sys.path.append('.')
from pfsolver import PropertyTable, SCNProperty
from pfsolver.customExceptions import SCNPropertyWarning


#shamrock
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
            'warning': True,
            'SCNProperty_kwargs': None,
        }

        # error triggered when wrong input keys for SCN_kwargs are used
        with self.assertRaises(TypeError):
            PropertyTable(sample_1, summary=True, warning=True, SCNProperty_kwargs={**SCN_kwargs, 'asdasd': 200})

        # Tb, mw, sg values should not be provided as SCNProperty_kwargs because there can be multiple pseudos
        # The values for the pseudos are to be provided in other ways
        with self.assertRaises(TypeError):
            PropertyTable(sample_1, summary=True, warning=True, SCNProperty_kwargs={**SCN_kwargs, 'Tb': 200})

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
            'reset': True,
        }

        ptable3 = PropertyTable(sample_3, **kwargs)
        ptable3.update_property('Hexanes+', {'MW': 90.161})

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

        """--------------------------- update_total tests with 2 properties ---------------------------"""

        ptable4 = PropertyTable(sample_1b, **kwargs)
        ptable4.update_property('Hexanes+', {'MW': 87.665})
        ptable4.update_property('Heptanes+', {'MW': 90})

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

        print(ptable4a.table.to_string())

        ptable4a.update_total({'mw': mw_total}, **{**update_total_kwargs})
        # ptable4a.update_total({'mw': mw_total}, **{**update_total_kwargs, 'target_compound': 'Hexanes+'})
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







        # Todo: make a test case for unidentified compound - description for + fraction detection condition (+, plus)


        # Todo: add custom chemical name for n-heptane


        # Todo: check again how GHV values of fractions are computed - GPA 2172 or ISO 6976 according to Promax

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
