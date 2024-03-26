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


        print('----------------------------------------------------------------------------------------')
        ptable1a = PropertyTable(sample_1a, **kwargs)
        self.assertEqual(ptable1a.names_fraction, ['Hexanes+'])
        self.assertEqual(len(ptable1a.names_fraction), 1)
        self.assertEqual(ptable1a.compound_indices_dict[ptable1a.names_fraction[0]], 8)

        ################# new test ###########################
        ptable1b = PropertyTable(sample_1, **kwargs)
        idx_total = max(ptable1b.compound_indices_dict.values()) + 1  # total row is the last row
        print(ptable1b.table.to_string())
        print('----------------------------------------------------------------------------------------')

        """------------------------- Basic calculated value checks -------------------------"""
        # GHV_liq input shouldn't calculate other columns, as it requires GHV_Gas and MW to solve for other properties
        ptable1b = PropertyTable(sample_1, **kwargs)
        ptable1b.update_property('Hexanes+', {'GHV_liq': 20408})
        self.assertTrue(pd.isna(ptable1b.table.loc[idx_total, 'GHV_liq']))
        self.assertTrue(pd.isna(ptable1b.table.loc[idx_total, 'Mass_Fraction']))
        self.assertTrue(pd.isna(ptable1b.table.loc[0, 'Mass_Fraction']))

        ptable1c = PropertyTable(sample_1, **kwargs)
        ptable1c.update_property('Hexanes+', {'sg_gas': 3.0180})
        self.assertAlmostEqual(ptable1c.table.loc[idx_total, 'MW'], 23.379, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total, 'Mass_Fraction'], 1, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total, 'GHV_gas'], 1380.186, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total, 'GHV_liq'], 22403.2729, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total, 'SG_gas'], 0.807207, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total, 'SG_liq'], 0.350053, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total - 1, 'MW'], 87.431460, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total - 1, 'Mass_Fraction'], 0.016642, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total - 1, 'GHV_gas'], 4779.327525, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total - 1, 'GHV_liq'], 20744.009182, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total - 1, 'SG_gas'], 3.018000, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total - 1, 'SG_liq'], 0.697153, places=3)

        ptable1d = PropertyTable(sample_1, **kwargs)
        ptable1d.update_property('Hexanes+', {'SG liq': 0.6933})
        self.assertAlmostEqual(ptable1d.table.loc[idx_total, 'MW'], 23.373998, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total, 'Mass_Fraction'], 1, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total, 'GHV_gas'], 1379.939882, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total, 'GHV_liq'], 22404.080005, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total, 'SG_gas'], 0.807034, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total, 'SG_liq'], 0.350036, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total - 1, 'MW'], 86.303649, places=3)
        self.assertAlmostEqual(ptable1c.table.loc[idx_total - 1, 'Mass_Fraction'], 0.016431, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total - 1, 'GHV_gas'], 4723.916421, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total - 1, 'GHV_liq'], 20771.443629, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total - 1, 'SG_gas'], 2.979070, places=3)
        self.assertAlmostEqual(ptable1d.table.loc[idx_total - 1, 'SG_liq'], 0.693300, places=3)

        ptable1e = PropertyTable(sample_1, **kwargs)
        ptable1e.update_property('Hexanes+', {'ghv gas': 4774.1})
        self.assertAlmostEqual(ptable1e.table.loc[idx_total, 'MW'], 23.378544, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total, 'Mass_Fraction'], 1, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total, 'GHV_gas'], 1380.163199, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total, 'GHV_liq'], 22403.348666, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total, 'SG_gas'], 0.807191, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total, 'SG_liq'], 0.350051, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total - 1, 'MW'], 87.325161, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total - 1, 'Mass_Fraction'], 0.016622, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total - 1, 'GHV_gas'], 4774.100000, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total - 1, 'GHV_liq'], 20746.543528, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total - 1, 'SG_gas'], 3.014331, places=3)
        self.assertAlmostEqual(ptable1e.table.loc[idx_total - 1, 'SG_liq'], 0.696796, places=3)

        ptable1f = PropertyTable(sample_1, **kwargs)
        ptable1f.update_property('Hexanes+', {'MW': 87.665})
        self.assertAlmostEqual(ptable1f.table.loc[idx_total, 'MW'], 23.380056, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total, 'Mass_Fraction'], 1, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total, 'GHV_gas'], 1380.237584, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total, 'GHV_liq'], 22403.106907, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total, 'SG_gas'], 0.807243, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total, 'SG_liq'], 0.350056, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total - 1, 'MW'], 87.665, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total - 1, 'Mass_Fraction'], 0.016686, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total - 1, 'GHV_gas'], 4790.815715, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total - 1, 'GHV_liq'], 20738.477101, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total - 1, 'SG_gas'], 3.026061, places=3)
        self.assertAlmostEqual(ptable1f.table.loc[idx_total - 1, 'SG_liq'], 0.697933, places=3)

        ptable1g = PropertyTable(sample_1, **kwargs)
        ptable1g.update_property('Hexanes+', {'Mass_Fraction': 0.016686})
        self.assertTrue(pd.isna(ptable1g.table.loc[idx_total, 'GHV_liq']))
        self.assertTrue(pd.isna(ptable1g.table.loc[idx_total, 'Mass_Fraction']))
        self.assertTrue(pd.isna(ptable1g.table.loc[0, 'Mass_Fraction']))

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

        """--------------------------------------------------------------------------------"""

        ptable2 = PropertyTable(sample_1b, **kwargs)
        self.assertEqual(ptable2.names_fraction, ['Hexanes+', 'Heptanes+'])
        self.assertEqual(len(ptable2.names_fraction), 2)
        self.assertEqual(ptable2.compound_indices_dict[ptable2.names_fraction[0]], 9)
        self.assertEqual(ptable2.compound_indices_dict[ptable2.names_fraction[1]], 10)

        # No Nones, instead I implemented np.nan (NaN)
        df2 = ptable2.table
        has_None = df2.isin([None]).any().any()
        self.assertFalse(has_None, "DataFrame contains 'None' values, which is not allowed.")

        print(df2.to_string())



        # Todo: add Tb column on the returned DF, make the total Tb as NA because its essentially a bubble point


        # Todo: this column testing needs to be repeated for all test cases
        # ensure that all numeric columns have dtype=float64
        exclude_columns = ['Name', 'CAS']
        for column in df2.columns.difference(exclude_columns):
            self.assertTrue(df2[column].dtype == np.float64, f"Column '{column}' is not of dtype float64.")


        # Todo: testing section for GPA vs. non-GPA values


        #ptable.update_property('Hexanes+', {'mw': 95})



if __name__ == '__main__':
    unittest.main()


# Documents\GasCompressibiltiyFactor-py>python -m unittest tests.test_gascomp
# python -m unittest discover .
