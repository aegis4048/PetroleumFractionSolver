import numpy as np
import pint
from thermo import ChemicalConstantsPackage
from matplotlib import pyplot as plt

from pfsolver import utilities
from pfsolver import config
from pfsolver import eos


UREG = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)


comp = {'n-butane': 10, 'n-hexane': 90}
comp = {'n-hexane': 100}
comp = {'4-methyloctane': 100}
names = list(comp.keys())
zs = list(comp.values())

constants = ChemicalConstantsPackage.constants_from_IDs(names)
Pcs = np.array(constants.Pcs)
Tcs = np.array(constants.Tcs)
omegas = np.array(constants.omegas)
Tbs = np.array(constants.Tbs)

print(constants.CASs[0])

Tc = Tcs[0]
Pc = Pcs[0]
omega = omegas[0]
mw = constants.MWs[0]

#T = UREG('%.15f degF' % 450).to('kelvin')._magnitude
T = UREG('%.15f degF' % 400).to('kelvin')._magnitude
P = UREG('%.15f psi' % 200).to('pascal')._magnitude


print('T:', T, 'K')
print('P:', P, 'pascal')
print('Tc:', Tc)
print('Pc:', Pc)
print('omega:', omega)
print('Tr:', T / Tc)
omega = 0.5
obj = eos.PR(T, P, Tc, Pc, omega, mw=mw)
#fig, ax = obj.plot()
plt.show()
print('roots', obj.roots)
print('V_liq:', obj.V_liq, 'm^3/mol')
print('V_gas:', obj.V_gas, 'm^3/mol')
print('rho_liq:', obj.rho_liq, 'g/m^3')
print('rho_gas:', obj.rho_gas, 'g/m^3')

print('---------------------------------------')
print(omega)
obj = eos.PR78(T, P, Tc, Pc, omega, mw=mw)
print('roots', obj.roots)
print('V_liq:', obj.V_liq, 'm^3/mol')
print('V_gas:', obj.V_gas, 'm^3/mol')
print('rho_liq:', obj.rho_liq, 'g/m^3')
print('rho_gas:', obj.rho_gas, 'g/m^3')

fig, ax = obj.plot()


