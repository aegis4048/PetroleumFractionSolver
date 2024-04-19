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
from pfsolver import PropertyTable
from pfsolver import config


UREG = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)


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

names = ['methane', 'n-C6', 'n-C7', 'n-C29', 'n-C30',
 'n-C31', 'n-C32', 'n-C33', 'n-C49', 'n-C50']

print('-----------------------------------------------')
test = {'n-decane': 50, 'n-nonane': 50,}
test = {'n-decane': 50, 'n-hexane': 30, 'n-pentane': 10, 'n-butane': 5, 'methane': 5}

test = {'n-decane': 50, 'n-butane': 50}
names = list(test.keys())
constants = ChemicalConstantsPackage.constants_from_IDs(names)
Vs = COSTALD_mixture(list(test.values()), 288.7, constants.Tcs, constants.Vcs, constants.omegas)

MW_mix = np.sum(np.array(constants.MWs) * np.array(list(test.values())) / 100)
sg = Vm_to_rho(Vs, float(MW_mix))

print('Vs:', Vs)
print('MW_mix:', MW_mix)
print('sg:', sg)

test = {'methane': 5, 'ethane': 5, 'propane': 5, 'n-butane': 5,
        'n-pentane': 10, 'n-hexane': 30, 'n-decane': 40}

names = list(test.keys())
constants = ChemicalConstantsPackage.constants_from_IDs(names)
bubble_Ts = {'methane': 111.68, 'ethane': 184.68, 'propane': 231.21, 'n-butane': 272.74,
        'n-pentane': 288.7, 'n-hexane': 288.7, 'n-decane': 288.7}

for i, name in enumerate(names):
    print(name, '------------------------------')
    print('\tsg:', round(constants.rhol_60Fs_mass[i] / config.constants['RHO_WATER'], 3))

    sg_COSTALD_bubble_T = Vm_to_rho(COSTALD(list(bubble_Ts.values())[i], constants.Tcs[i], constants.Vcs[i], constants.omegas[i]), constants.MWs[i])
    sg_COSTALD_60F = Vm_to_rho(COSTALD(288.7, constants.Tcs[i], constants.Vcs[i], constants.omegas[i]), constants.MWs[i])

    print('\tsg-COSTALD @BubbleT:', round(sg_COSTALD_bubble_T, 1))
    print('\tsg-COSTALD @60F:', round(sg_COSTALD_60F, 1))

    #print('\tTr:', round(288.7 / constants.Tcs[i], 3))
    print('\tTb:', round(UREG('%.15f kelvin' % constants.Tbs[i]).to('fahrenheit')._magnitude, 1), 'F')
    print('\tTc:', round(UREG('%.15f kelvin' % constants.Tcs[i]).to('celsius')._magnitude, 1), 'C')
    print('\tPc:', round(UREG('%.15f pascal' % constants.Pcs[i]).to('atm')._magnitude, 1), 'atm')
    print('\tVc:', '%s' % float('%.3g' % constants.Vcs[i]), 'm^3/mol')
    print('\tVc:', '%s' % float('%.3g' % (constants.Vcs[i] / constants.MWs[i] / 1000)), 'm^3/kg')
    print('\tomega:', constants.omegas[i])
    #print('\trhol_60Fs_mass:', round(constants.rhol_60Fs_mass[i], 1))
    #print('\tdensity:', constants.rhol_60Fs_mass[i] / config.constants['RHO_WATER'])
    #print('-----------------------------------------------')


ptable10 = PropertyTable(test, warning=True, extended=True)
#ptable10.update_property('n-butane', {'SG_liq': 0.64587})

print(ptable10.table.to_string())

# methane
A = 0.1627
B = 0.29337
C = 190.56
n = 0.28571
T = 111.67

# C33 Tritriacontane
A = 0.2591
B = 0.24875
C = 946.27
n = 0.28571
T = 288.7

density = A * B ** -((1 - T/C) ** n)
print('methane density:', density)

# methane density in Promax = 0.4232 at -256.7 F.

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

constants = ChemicalConstantsPackage.constants_from_IDs(['n-butane'])
rhol_60F = constants.rhol_60Fs_mass[0]
sg = rhol_60F / config.constants['RHO_WATER']
print(sg)

# butane properties
name = 'tritriacontane'
constants = ChemicalConstantsPackage.constants_from_IDs([name])
T = 288.7
Tc = constants.Tcs[0]
Vc = constants.Vcs[0]
omega = constants.omegas[0]
Vs = chemicals.volume.COSTALD(T, Tc, Vc, omega)

print('-----------------------------------------------')
rho = chemicals.utils.Vm_to_rho(Vs, constants.MWs[0])
print('name:  ', name)
print('rho:   ', rho)
print('Tc:    ', Tc)
print('omega: ', omega)

# Todo: Test COSTALD density method for HCs lighter than n-C5 to test sg_liq.

# Notes: SG_liq values from GPA is not reliable. Promax seems to agree more with Pubchem values


# Todo: This code returns NaN for n-butane because I changed sg_liq values. This shouldn't happen. Fix it.
"""
test = {'n-decane': 50, 'n-butane': 50,}
ptable10 = PropertyTable(test, warning=True, extended=True)
ptable10.update_property('n-butane', {'SG_liq': 0.64587})
"""


# Todo: https://wiki.whitson.com/eos/bips/
#  Make BIPS tunable to match experimental data for reservoir sim.









