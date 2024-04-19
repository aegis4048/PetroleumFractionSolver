import pandas as pd
import numpy as np
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CEOSGas, FlashVLN
import thermo
import chemicals
from chemicals.utils import Vm_to_rho
from chemicals.volume import COSTALD, COSTALD_mixture, COSTALD_compressed
import pint

import pfsolver
from pfsolver import config
from pfsolver import utilities

UREG = pint.UnitRegistry()


T = 288.7055555556
test = {'n-decane': 0.5, 'n-butane': 0.5}
names = list(test.keys())
constants = ChemicalConstantsPackage.constants_from_IDs(names)
MW_mix = np.sum(np.array(constants.MWs) * np.array(list(test.values())))
sg = Vm_to_rho(COSTALD_mixture(list(test.values()), T, constants.Tcs, constants.Vcs, constants.omegas), float(MW_mix))
print('-----------------------------------------------')
print('50:50 n-butane/n-decane mixture at %.1fK:' % T)
print('Density :', round(sg, 1), 'kg/m^3')

T = 288.7055555556
test = {'n-decane': 0.95, 'propane': 0.05}
names = list(test.keys())
constants = ChemicalConstantsPackage.constants_from_IDs(names)
MW_mix = np.sum(np.array(constants.MWs) * np.array(list(test.values())))
sg = Vm_to_rho(COSTALD_mixture(list(test.values()), T, constants.Tcs, constants.Vcs, constants.omegas), float(MW_mix))
print('-----------------------------------------------')
print('05:95 n-butane/n-decane mixture at %.1fK:' % T)
print('Density :', round(sg, 1), 'kg/m^3')

T = 126.53
test = {'n-decane': 0.9, 'methane': 0.1}
names = list(test.keys())
constants = ChemicalConstantsPackage.constants_from_IDs(names)
MW_mix = np.sum(np.array(constants.MWs) * np.array(list(test.values())))
sg = Vm_to_rho(COSTALD_mixture(list(test.values()), T, constants.Tcs, constants.Vcs, constants.omegas), float(MW_mix))
print('-----------------------------------------------')
print('1:9 methane/n-decane mixture at %.1fK:' % T)
print('Density :', round(sg, 1), 'kg/m^3')

T = 270.33
test = {
    'propane': 0.2,
    'n-butane': 0.2,
    'n-decane': 0.6,
}
names = list(test.keys())
constants = ChemicalConstantsPackage.constants_from_IDs(names)
MW_mix = np.sum(np.array(constants.MWs) * np.array(list(test.values())))
sg = Vm_to_rho(COSTALD_mixture(list(test.values()), T, constants.Tcs, constants.Vcs, constants.omegas), float(MW_mix))
print('-----------------------------------------------')
print('20:20:60 C3/C4/C10 mixture at %.1fK:' % T)
print('Density :', round(sg, 3), 'kg/m^3')


P_psi = 14.7
P = UREG('%.15f psi' % P_psi).to('pascal')._magnitude
T = 288.7055555556
test = {
    'propane': 0.1,
    'n-butane': 0.1,
    'n-decane': 0.8,
}
names = list(test.keys())
constants = ChemicalConstantsPackage.constants_from_IDs(names)
MW_mix = np.sum(np.array(constants.MWs) * np.array(list(test.values())))
#Vs_dense = COSTALD_compressed(T, P, Psat, constants.Tcs[0], constants.Pcs[0], constants.omegas[0], Vs)
#density_dense = Vm_to_rho(Vs_dense, constants.MWs[0])
sg = Vm_to_rho(COSTALD_mixture(list(test.values()), T, constants.Tcs, constants.Vcs, constants.omegas), float(MW_mix))
print('-----------------------------------------------')
print('10:10:80 C3/C4/C10 mixture at %.1fK:' % T)
print('Density :', round(sg, 3), 'kg/m^3')


names = ['methane', 'ethane', 'propane', 'n-butane', 'n-pentane', 'n-hexane', 'heptane', 'n-octane', 'n-nonane', 'n-decane']
#names = ['propane', 'n-butane', 'n-decane']

P_psi = 14.7
P = UREG('%.15f psi' % P_psi).to('pascal')._magnitude
for i, name in enumerate(names[1:]):

    constants = ChemicalConstantsPackage.constants_from_IDs([name])

    print(constants.names[0])
    T = 288.7
    Tr = T / constants.Tcs[0]
    print('Tr: ', round(Tr, 3))

    Vs = COSTALD_mixture([1], T, constants.Tcs, constants.Vcs, constants.omegas)
    density = Vm_to_rho(Vs, constants.MWs[0])
    sg = density / config.constants['RHO_WATER']
    Tc = UREG('%.15f kelvin' % constants.Tcs[0]).to('fahrenheit')._magnitude
    omega = constants.omegas[0]

    # Define critical volume in m^3/mol and molecular weight in g/mol using the provided variable names
    mw = constants.MWs[0]
    mw_kg_per_mol = mw * 0.001
    density_kg_per_m3 = mw_kg_per_mol / constants.Vcs[0]
    density_lb_per_ft3 = UREG(f'{density_kg_per_m3} kg/m^3').to('lb/ft^3').magnitude

    Psat = constants.Psat_298s[0]
    sg_dense = 1
    Vs_dense = COSTALD_compressed(T, P, Psat, constants.Tcs[0], constants.Pcs[0], constants.omegas[0], Vs)
    #density_dense = Vm_to_rho(Vs_dense, constants.MWs[0])
    #sg_dense = density_dense / config.constants['RHO_WATER']

    print('-----------------------------------------------')
    print('%s at %.1fK:' % (name, T))
    print('\tP:', round(P_psi, 5))
    print('\tsg:', round(sg, 5))
    print('\tsg_dense:', round(sg_dense, 5))
    print('\tTc:', round(Tc, 5))
    print('\tomega:', round(omega, 5))
    print('\trho_c:', round(density_lb_per_ft3, 5))

    temp = constants.Psat_298s[0]

shamrock = dict([
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
brazos_liquid = {
    'nitrogen': 0,
    'carbon dioxide': 0.007,
    'methane': 0.052,
    'ethane': 0.495,
    'propane': 3.985,
    'isobutane': 1.935,
    'n-butane': 11.854,
    'isopentane': 8.710,
    'n-pentane': 12.783,
    'Hexanes+': 60.179,
}
test = {
    'propane': 10,
    'n-butane': 10,
    'n-decane': 80,
}

ptable = pfsolver.PropertyTable(test, warning=False, extended=True)
#ptable.update_property('Hexanes+', {'MW': 87.665})
#ptable.update_property('Hexanes+', {'sg_liq': 0.6933})
#ptable.update_property('Hexanes+', {'MW': 93.189})
#ptable.update_property('Hexanes+', {'sg_liq': 0.6756}, recalc=False)
print(ptable.table.to_string())

# Todo:
#  1. Add temperature and pressure condition to PropertyTable. By default, read T and P from config.
#  2. Provide SG_liq values for known compounds with COSTALD method. Implement different models. Make an option to choose model
#  3. Implement sg_liq COSTALD method
#  4. Build test for mixture SG_liq without plus fractions. Like 50% decane and 50% butane mixtures. Get values of Promax
#  6. Find remedy for heavy compounds' unreliable properties
#  7. Find saturated liquid density at high pressure (COSTALD-HP model exists) - DONE: COSTALD_compressed works for pure components, need to test for mixtures
#  8. Investigate why my method of COSTALD sg_liq is always higher than Promax's for mixtures.
#  9. Also check if its higher for pure compounds too

# Todo: After lunch (Pandas), read the HBTD paper on compressed mixture



