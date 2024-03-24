import pfsolver


sample_1 = dict([
    ('nitrogen', 0.86),
    ('carbon dioxide', 0.374),
    ('methane', 71.351),
    ('ethane', 12.287),
    ('propane', 9.147),
    ('isobutane', 0.985),
    ('n-butane', 3.180),
    ('i-pentane', 0.628),
    ('n-pentane', 0.743),  # 0.743
    ('Hexanes+', 0.445),
])

sample_2 = dict([
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
])

ptable = pfsolver.PropertyTable(shamrock, summary=True, warning=True, SCNProperty_kwargs={'warning': False, 'model': 'kf'})
print(ptable.table.to_string())


