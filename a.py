import pfsolver


shamrock = dict([
    ('nitrogen', 0.86),  # 0.86
    ('carbon dioxide', 0.374),
    ('methane', 71.351),
    ('ethane', 12.287),
    ('propane', 9.147),
    ('isobutane', 0.985),
    ('n-butane', 3.180),
    ('i-pentane', 0.628),
    ('n-pentane', 0.743),  # 0.743
    ('Hexanes+', 0.445),
    #('Heptanes+', 0.2),
    #('heneicosane', 0.01),
])

ptable = pfsolver.PropertyTable(shamrock, summary=True, warning=True, SCNProperty_kwargs={'warning': False, 'model': 'kf'})
print(ptable.table.to_string())


