import numpy as np
import pint
from thermo import ChemicalConstantsPackage
from matplotlib import pyplot as plt
import sys

from pfsolver import utilities
from pfsolver import config
from pfsolver import eos
from pfsolver.eos import calculatorPR


UREG = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

comp = {'n-butane': 0.1, 'n-hexane': 0.9}
#comp = {'n-butane': 1}
names = list(comp.keys())
zs = list(comp.values())

constants = ChemicalConstantsPackage.constants_from_IDs(names)
Pcs = np.array(constants.Pcs)
Tcs = np.array(constants.Tcs)
omegas = np.array(constants.omegas)
Tbs = np.array(constants.Tbs)
mws = np.array(constants.MWs)
mw = np.sum(np.array(constants.MWs) * np.array(list(comp.values())))

T = UREG('%.15f degF' % 400).to('kelvin')._magnitude  # Test T between 1 ~ 401 by 40 increment
P = UREG('%.15f psi' % 200).to('pascal')._magnitude   # Test P between 1 ~ 1001 by 100 increment

obj = eos.PR(T, P, Tcs[0], Pcs[0], omegas[0], mws[0])
print('V_liq:', obj.V_liq)
print('V_gas:', obj.V_gas)
print('rho_liq:', obj.rho_liq)
print('rho_gas:', obj.rho_gas)
print('roots:', obj.roots)

print('------------------------------------')

obj = eos.PR78(T, P, Tcs[0], Pcs[0], omegas[0], mws[0])
print('V_liq:', obj.V_liq)
print('V_gas:', obj.V_gas)
print('rho_liq:', obj.rho_liq)
print('rho_gas:', obj.rho_gas)
print('roots:', obj.roots)

obj.plot()
#plt.show()

print('--------------------------------------23424324')

obj = eos.PRMIX(T, P, Tcs, Pcs, omegas, zs, mws=mws)
print('V_liq:', obj.V_liq)
print('V_gas:', obj.V_gas)
print('rho_liq:', obj.rho_liq)
print('rho_gas:', obj.rho_gas)
print('roots:', obj.roots)

print('------------------------------------')

obj = eos.PRMIX78(T, P, Tcs, Pcs, omegas, zs, mws=mws, analytical=True)
print('V_liq:', obj.V_liq)
print('V_gas:', obj.V_gas)
print('rho_liq:', obj.rho_liq)
print('rho_gas:', obj.rho_gas)
print('roots:', obj.roots)




