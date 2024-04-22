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
import sys

UREG = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

def calc_Vs_sg(comp_dict, T):
    comp_dict = utilities.normalize_composition_dict(comp_dict)
    names = list(comp_dict.keys())
    constants = ChemicalConstantsPackage.constants_from_IDs(names)
    MW_mix = np.sum(np.array(constants.MWs) * np.array(list(comp_dict.values())))
    if len(names) <= 1:
        Vs = COSTALD(T, constants.Tcs[0], constants.Vcs[0], constants.omegas[0])
    else:
        Vs = COSTALD_mixture(list(comp_dict.values()), T, constants.Tcs, constants.Vcs, constants.omegas)

    sg = Vm_to_rho(Vs, float(MW_mix))
    return Vs, sg


def calc_Vs_sg_compressed(comp_dict, T, P):
    comp_dict = utilities.normalize_composition_dict(comp_dict)
    names = list(comp_dict.keys())
    zs = list(comp_dict.values())
    constants = ChemicalConstantsPackage.constants_from_IDs(names)
    MW_mix = np.sum(np.array(constants.MWs) * np.array(list(comp_dict.values())))

    if len(names) <= 1:
        Vs = COSTALD(T, constants.Tcs[0], constants.Vcs[0], constants.omegas[0])
        Vs_dense = COSTALD_compressed(T, P, constants.Psat_298s[0], constants.Tcs[0], constants.Pcs[0],
                                      constants.omegas[0], Vs)
        sg_dense = Vm_to_rho(Vs_dense, float(MW_mix))
    else:
        omega_m = np.sum(np.array(constants.omegas) * np.array(zs))
        Zcm = 0.291 - 0.080 * omega_m
        N = len(zs)
        sum1, sum2, sum3 = 0.0, 0.0, 0.0
        for i in range(N):
            sum1 += zs[i] * constants.Vcs[i]
            p = constants.Vcs[i] ** (1.0 / 3.)
            v = zs[i] * p
            sum2 += v
            sum3 += v * p

        Vm = 0.25 * (sum1 + 3.0 * sum2 * sum3)
        Vm_inv_root = np.sqrt(2) * (Vm) ** -0.5
        vec = [0.0] * N
        for i in range(N):
            vec[i] = (constants.Tcs[i] * constants.Vcs[i]) ** 0.5 * zs[i] * Vm_inv_root

        Tcm = 0.0
        for i in range(N):
            for j in range(i):
                Tcm += vec[i] * vec[j]
            Tcm += 0.5 * vec[i] * vec[i]
        Trm = T / Tcm

        alpha = 35 - 36 / Trm - 96.736 * np.log10(Trm) + Trm**6
        beta = np.log10(Trm) + 0.03721754 * alpha

        Prm_0 = 5.8031817 * np.log10(Trm) + 0.07608141 * alpha
        Prm_1 = 4.86601 * beta

        Prm = 10 ** (Prm_0 + Prm_1 * omega_m)
        Pcm = Zcm * config.constants['R'] * Tcm / Vm
        Psm = Prm * Pcm

        print('Tcm:', round(Tcm, 1), 'K')

        Vs = COSTALD_mixture(list(comp_dict.values()), T, constants.Tcs, constants.Vcs, constants.omegas)
        Vs_dense = COSTALD_compressed(T, P, Psm, Tcm, Pcm, omega_m, Vs)
        sg_dense = Vm_to_rho(Vs_dense, float(MW_mix))

        # note: the point of this process is to compute the Ps of the mixture

    return Vs_dense, sg_dense

P_psi = 14.7
P_Mpa = 0.1
#P = UREG('%.15f psi' % P_psi).to('pascal')._magnitude
P = UREG('%.15f MPa' % P_Mpa).to('pascal')._magnitude
#calc_Vs_sg_compressed({'n-butane': 50, 'n-decane': 50}, 288.7, P, mode='mixture')
#calc_Vs_sg_compressed({'n-propane': 5, 'n-decane': 95}, 288.7, P, mode='mixture')

density_x1 = 0.3968
density_x2 = 1 - density_x1
calc_Vs_sg_compressed({'n-hexane': density_x1, 'n-decane': density_x2}, 288.7, P)


print('---------------------------------------')
T = UREG('%.15f degF' % 60).to('kelvin')._magnitude
P_psi = 20000
P = UREG('%.15f psi' % P_psi).to('pascal')._magnitude
comp = {'n-butane': 50, 'n-decane': 50}
Vs, sg = calc_Vs_sg(comp, T=T)
Vs_dense, sg_dense = calc_Vs_sg_compressed(comp, T=T, P=P)
print('50:50 n-butane/n-decane mixture at %.1fK:' % T)
print('Pressure:', P_psi, 'psi')
print('LP Density:', round(sg, 1), 'kg/m^3')
print('HP Density:', round(sg_dense, 1), 'kg/m^3')

print('---------------------------------------')
T = UREG('%.15f degF' % 60).to('kelvin')._magnitude
P_psi = 20000
P = UREG('%.15f psi' % P_psi).to('pascal')._magnitude
comp = {'n-butane': 1}
Vs, sg = calc_Vs_sg(comp, T=T)
Vs_dense, sg_dense = calc_Vs_sg_compressed(comp, T=T, P=P)
print('%s at %.1fK:' % (list(comp.keys())[0], T))
print('Pressure:', P_psi, 'psi')
print('LP Density:', round(sg, 1), 'kg/m^3')                           # 582.6
print('HP Density:', round(sg_dense, 1), 'kg/m^3 @ %.1f psi' % P_psi)  # 692.6


import preos
from thermo import ChemicalConstantsPackage
import pint

UREG = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

T = UREG('%.15f degF' % 300).to('kelvin')._magnitude
T = 394.15
P = UREG('%.15f psi' % 14.7).to('bar')._magnitude
P = UREG('%.15f MPa' % 0.3).to('bar')._magnitude

comp = {'n-butane': 0.5, 'n-decane': 0.5}
comp = {'n-butane': 1}
comp = {'ethane': 1}
comp = {'n-hexane': 1}
comp = {'heptane': 0.5, 'CO2': 0.5}
names = list(comp.keys())
zs = list(comp.values())
constants = ChemicalConstantsPackage.constants_from_IDs(names)

try:
    m1 = preos.Molecule(names[0], constants.Tcs[0], constants.Pcs[0], constants.omegas[0])
    m2 = preos.Molecule(names[1], constants.Tcs[1], constants.Pcs[1], constants.omegas[1])
    bips = 0

    props = preos.preos_mixture(m1, m2, bips, T, P, zs, plotcubic=True, printresults=True)
except:
    m1 = preos.Molecule("methane", constants.Tcs[0], constants.Pcs[0], constants.omegas[0])
    props = preos.preos(m1, T, P, plotcubic=True, printresults=True)

density = props['density(mol/m3)'] * constants.MWs[0]
print('Density:', density, 'g/m^3')
print(props)









