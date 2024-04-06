import pandas as pd
import numpy as np
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CEOSGas, FlashVLN
import thermo
import chemicals
from chemicals.utils import Vm_to_rho
from chemicals.volume import COSTALD, COSTALD_mixture
from thermo.interaction_parameters import IPDB
import pint
import copy
import timeit

import pfsolver


UREG = pint.UnitRegistry()


comp = {
    'heneicosane': 30,
    'docosane': 30,
    'heptadecane': 40
}
names = list(comp.keys())

names = ['n-C6', 'n-C7', 'n-C8', 'n-C9', 'n-C10',
 'n-C11', 'n-C12', 'n-C13', 'n-C14', 'n-C15', 'n-C16', 'n-C17', 'n-C18', 'n-C19', 'n-C20',
 'n-C21', 'n-C22', 'n-C23', 'n-C24', 'n-C25', 'n-C26', 'n-C27', 'n-C28', 'n-C29', 'n-C30',
 'n-C31', 'n-C32', 'n-C33', 'n-C34', 'n-C35', 'n-C36', 'n-C37', 'n-C38', 'n-C39', 'n-C40',
 'n-C41', 'n-C42', 'n-C43', 'n-C44', 'n-C45', 'n-C46', 'n-C47', 'n-C48', 'n-C49', 'n-C50']


constants = ChemicalConstantsPackage.constants_from_IDs(names)

Tc = constants.Tcs[0]
Pc = constants.Pcs[0]
Vc = constants.Vcs[0]
omega = constants.omegas[0]


names = ['n-C6', 'n-C7', 'n-C8', 'n-C9', 'n-C10',
 'n-C11', 'n-C12', 'n-C13', 'n-C14', 'n-C15', 'n-C16', 'n-C17', 'n-C18', 'n-C19', 'n-C20',
 'n-C21', 'n-C22', 'n-C23', 'n-C24', 'n-C25', 'n-C26', 'n-C27', 'n-C28', 'n-C29', 'n-C30',
 'n-C31', 'n-C32', 'n-C33', 'n-C34', 'n-C35', 'n-C36', 'n-C37', 'n-C38', 'n-C39', 'n-C40',
 'n-C41', 'n-C42', 'n-C43', 'n-C44', 'n-C45', 'n-C46', 'n-C47', 'n-C48', 'n-C49', 'n-C50']

names = ['n-C16', 'n-C17', 'n-C18', 'n-C19']
names = ['n-C6', 'n-C7', 'n-C29', 'n-C30',
 'n-C31', 'n-C32', 'n-C49', 'n-C50']

# Since you want the sum of values to equal 1, each component will have an equal share of 1/total_components
value_per_comp = 1 / len(names)

# Create a dictionary with each compound name as a key and the value_per_comp as the value
comp = {name: value_per_comp for name in names}
ptable = pfsolver.PropertyTable(comp, warning=False, extended=True)
print(ptable.SCNProperty_kwargs)
print(ptable.table.to_string())

chemicals.acentric.omega('630-04-6', method=None)

names = ['n-C6', 'n-C7', 'n-C29', 'n-C30',
 'n-C31', 'n-C32', 'n-C49', 'n-C50']

"""
for i, name in enumerate(names):
    print(name, ':', constants.names[i])

    Tb_K = constants.Tbs[i]
    Tb_F = round(UREG('%.15f kelvin' % Tb_K).to('fahrenheit')._magnitude, 1)
    Tc_K = constants.Tcs[i]
    Tc_F = round(UREG('%.15f kelvin' % Tc_K).to('fahrenheit')._magnitude, 1)
    Pc = constants.Pcs[i]
    Pc_psia = round(UREG('%.15f pascal' % Pc).to('psi')._magnitude, 2)

    #print('\tTb [F]    :  ', Tb_F)
    #print('\tTc [F]    :  ', Tc_F)
    print('\tPc [psia] :  ', Pc_psia)
    print('-----------------------------------------------')
"""


names = ['n-C6', 'n-C7', 'n-C29', 'n-C30',
 'n-C31', 'n-C32', 'n-C49', 'n-C50']
constants = ChemicalConstantsPackage.constants_from_IDs(names)

for i, name in enumerate(names):
    print(name, ':', constants.names[i])
    print('\tTc:', constants.Tcs[i])
    print('\tPc:', constants.Pcs[i])
    print('\tVc:', constants.Vcs[i])
    print('\tomega:', constants.omegas[i])
    print('\trhol_60Fs_mass:', round(constants.rhol_60Fs_mass[i], 1))
    print('-----------------------------------------------')



"""
constants = ChemicalConstantsPackage.constants_from_IDs(['n-butane'])
T = 288.7
Tc = constants.Tcs[0]
Vc = constants.Vcs[0]
omega = constants.omegas[0]
mw = constants.MWs[0]
Vs = chemicals.volume.COSTALD(T, Tc, Vc, omega)
rho = Vm_to_rho(Vs, mw)

Tr = T/Tc

tau = 1.0 - Tr
tau_cbrt = (tau)**(1.0/3.)
V_delta = (-0.296123 + Tr*(Tr*(-0.0480645*Tr - 0.0427258) + 0.386914))/(Tr - 1.00001)
V_0 = tau_cbrt*(tau_cbrt*(tau_cbrt*(0.190454*tau_cbrt - 0.81446) + 1.43907) - 1.52816) + 1.0
V_s = Vc*V_0*(1.0 - omega*V_delta)
V_s

mw / V_s / 1000

"""












