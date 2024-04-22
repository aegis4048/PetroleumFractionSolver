import pandas as pd
import numpy as np
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CEOSGas, FlashVLN
import thermo
from thermo.interaction_parameters import IPDB
import chemicals
from chemicals.utils import Vm_to_rho
from chemicals.volume import COSTALD, COSTALD_mixture, COSTALD_compressed
import pint


import pfsolver
from pfsolver import config
from pfsolver import utilities


UREG = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)



# Todo: test compressed mixture of methane and ethanes. They should fail? (I think) because at normal conditions
#  methane is in critical phase. Ethane is critical phase in T>90F


names = ['methane', 'ethane', 'propane', 'n-butane', 'n-pentane', 'n-hexane', 'heptane', 'n-octane', 'n-nonane', 'n-decane']
#names = ['propane', 'n-butane', 'n-decane']


T_F = 60
P_psi = 0
T = UREG('%.15f degF' % T_F).to('kelvin')._magnitude
P = UREG('%.15f psi' % P_psi).to('pascal')._magnitude
for i, name in enumerate(names[1:2]):

    # ethane at STP is vapor, so its ok for the sg_dense to fail.

    print('%s @%dF, %.1f psia -------------' % (name, T_F, P_psi))

    constants = ChemicalConstantsPackage.constants_from_IDs([name])
    Psat = constants.Psat_298s[0]
    Vs = COSTALD_mixture([1], T, constants.Tcs, constants.Vcs, constants.omegas)
    sg_STP = Vm_to_rho(Vs, constants.MWs[0])

    Tc = constants.Tcs[0]
    Pc = constants.Pcs[0]
    omega = constants.omegas[0]

    a = -9.070217
    b = 62.45326
    d = -135.1102
    f = 4.79594
    g = 0.250047
    h = 1.14188
    j = 0.0861488
    k = 0.0344483
    e = np.exp(f + omega*(g + h*omega))
    C = j + k*omega

    Tr = T/Tc
    if Tr > 1.0:
        print('Tr exceeds 1.0')
        #Tr = 0.999


    tau = 1.0 - Tr
    tau13 = tau**(1.0/3.0)
    B = Pc*(-1.0 + a*tau13 + b*tau13*tau13 + d*tau + e*tau*tau13)

    Vs_dense = Vs*(1.0 - C*np.log((B + P)/(B + Psat)))
    sg_dense = Vm_to_rho(Vs_dense, float(constants.MWs[0]))
    print('Tr          :', round(Tr, 2))
    print('(B + P)     :', round(B + P))
    print('(B + Psat)  :', round(B + Psat))
    print('(B + P)/(B + Psat):', (B + P)/(B + Psat))
    print('sg_dense    :', round(sg_dense, 1))
    print('sg_STP      :', round(sg_STP, 1))
    print('-------------------------------------------------------------')

    # Todo: If the phase is in critical phase, the Tau is negative and returns imaginary number.

    #Vs_dense2 = COSTALD_compressed(T, P, Psat, constants.Tcs[0], constants.Pcs[0], constants.omegas[0], Vs)


T_F = 60
P_psi = 14.7
T = UREG('%.15f degF' % T_F).to('kelvin')._magnitude
P = UREG('%.15f psi' % P_psi).to('pascal')._magnitude

comp = {'n-butane': 5, 'n-decane': 50}
comp = {'nitrogen': 50, 'methane': 50}
comp = {'ethane': 50}
comp_dict, _ = utilities.normalize_composition_dict(comp)
names = list(comp_dict.keys())
zs = list(comp_dict.values())
constants = ChemicalConstantsPackage.constants_from_IDs(['ethane'])
eos = thermo.eos_mix.PR78MIX(constants.Tcs, constants.Pcs, constants.omegas, zs=zs, T=T, P=P)




from thermo import *

comp = {'butane': 50, 'n-decane': 50}
comp_dict, _ = utilities.normalize_composition_dict(comp)
names = list(comp_dict.keys())
zs = list(comp_dict.values())

pure_constants = ChemicalConstantsPackage.constants_from_IDs(names)
constants = pure_constants
properties = PropertyCorrelationsPackage(constants=constants)

T_F = 60
P_psi = 14.7
T = UREG('%.15f degF' % T_F).to('kelvin')._magnitude
P = UREG('%.15f psi' % P_psi).to('pascal')._magnitude

eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)

gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
liq2 = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
phase_list = [gas, liq, liq]

flashN = FlashVLN(constants, properties, liquids=[liq, liq2], gas=gas)
res = flashN.flash(T=T, P=P, zs=zs)

kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', constants.CASs, 'kij')

# Todo: implement the PR78MIX from scratch. Make a simple function. Evolve to class if needed later
#