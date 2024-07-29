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
from pfsolver.eos import PRMIX78, PR78
import sys


UREG = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)


def calc_Vs_sg(comp_dict, T):
    comp_dict, total_comp = utilities.normalize_composition_dict(comp_dict)
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
    comp_dict, total_comp = utilities.normalize_composition_dict(comp_dict)
    print(comp_dict, total_comp)
    names = list(comp_dict.keys())
    zs = list(comp_dict.values())
    constants = ChemicalConstantsPackage.constants_from_IDs(names)
    MW_mix = np.sum(np.array(constants.MWs) * np.array(list(comp_dict.values())))

    if len(names) <= 1:
        Tc = constants.Tcs[0]
        Vc = constants.Vcs[0]
        omega = constants.omegas[0]
        Psat = constants.Psat_298s[0]
        Pc = constants.Pcs[0]

        Vs = COSTALD(T, Tc, Vc, omega)
        rho_STP = Vm_to_rho(Vs, float(MW_mix)) * 1000

        # Vs_dense = COSTALD_compressed(T, P, Psat, Tc, Pc, omega, Vs)
        a = -9.070217
        b = 62.45326
        d = -135.1102
        f = 4.79594
        g = 0.250047
        h = 1.14188
        j = 0.0861488
        k = 0.0344483
        e = np.exp(f + omega * (g + h * omega))
        C = j + k * omega

        Tr = T / Tc
        if Tr > 0.93:
            print('Tr exceeds 0.93:', Tr)
            # Tr = 0.999

        # decane density 10F, 200 psia = 756651 g/m3

        tau = 1.0 - Tr
        tau13 = tau ** (1.0 / 3.0)
        B = Pc * (-1.0 + a * tau13 + b * tau13 * tau13 + d * tau + e * tau * tau13)
        Vs_dense = Vs * (1.0 - C * np.log((B + P) / (B + Psat)))
        rho_dense = Vm_to_rho(Vs_dense, float(MW_mix)) * 1000

        print('Tr          :', round(Tr, 4))
        print('(B + P)     :', round(B + P))
        print('(B + Psat)  :', round(B + Psat))
        print('(B + P)/(B + Psat):', (B + P) / (B + Psat))
        print('sg_dense    :', round(rho_dense, 1))
        print('sg_STP      :', round(rho_STP, 1))
        print('-------------------------------------------------------------')
        print('T           :', T)
        print('P           :', P)
        print('Tc          :', Tc)
        print('Pc          :', UREG('%.15f pascal' % Pc).to('psi')._magnitude)
        print('omega       :', omega)
        print('MW_mix      :', MW_mix)
        print('-------------------------------------------------------------')
        obj = PR78(T, P, Tc, Pc, omega, MW_mix, analytical=False)
        print('rho_liq     :', obj.rho_liq)
        print('rho_gas     :', obj.rho_gas)
        print('-------------------------------------------------------------')
        obj_mix = PRMIX78(T, P, constants.Tcs, constants.Pcs, constants.omegas, zs, mws=constants.MWs, analytical=False)
        print('rho_liq     :', obj_mix.rho_liq)
        print('rho_gas     :', obj_mix.rho_gas)
        print('-------------------------------------------------------------')

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

        Vs = COSTALD_mixture(list(comp_dict.values()), T, constants.Tcs, constants.Vcs, constants.omegas)
        #Vs_dense = COSTALD_compressed(T, P, Psm, Tcm, Pcm, omega_m, Vs)
        #sg_dense = Vm_to_rho(Vs_dense, float(MW_mix))

        Tc = Tcm
        Pc = Pcm
        omega = omega_m
        Psat = Psm

        a = -9.070217
        b = 62.45326
        d = -135.1102
        f = 4.79594
        g = 0.250047
        h = 1.14188
        j = 0.0861488
        k = 0.0344483
        e = np.exp(f + omega * (g + h * omega))
        C = j + k * omega

        Tr = T / Tc
        if Tr > 0.93:
            print('Tr exceeds 0.93:', Tr)
            # Tr = 0.999

        # decane density 10F, 200psia = 756651 g/m3

        tau = 1.0 - Tr
        tau13 = tau ** (1.0 / 3.0)
        B = Pc * (-1.0 + a * tau13 + b * tau13 * tau13 + d * tau + e * tau * tau13)

        print('tau         :', tau)
        print('tau13       :', tau13)

        Vs_dense = Vs * (1.0 - C * np.log((B + P) / (B + Psat)))
        rho_dense = Vm_to_rho(Vs_dense, float(constants.MWs[0])) * 1000
        rho_STP = Vm_to_rho(Vs, constants.MWs[0]) * 1000

        print('Tcm         :', round(Tcm, 1), 'K')
        print('Tr          :', round(Tr, 4))
        print('B           :', B)
        print('(B + P)     :', round(B + P))
        print('(B + Psat)  :', round(B + Psat))
        print('(B + P)/(B + Psat):', (B + P) / (B + Psat))
        print('Psat        :', Psat)
        print('Psat decane :', constants.Psat_298s[1])
        print('sg_dense    :', round(rho_dense, 1))
        print('sg_STP      :', round(rho_STP, 1))
        print('Vm          :', Vm)
        print('Vs          :', Vs)
        print('Vs_dense    :', Vs_dense)
        print('Zcm         :', Zcm)
        print('-------------------------------------------------------------')

        print('T           :', T)
        print('P           :', P)
        print('Tc          :', Tc)
        print('Pc          :', UREG('%.15f pascal' % Pc).to('psi')._magnitude)
        print('omega       :', omega)
        print('MW_mix      :', MW_mix)
        print('-------------------------------------------------------------')
        obj = PR78(T, P, Tc, Pc, omega, MW_mix, analytical=False)
        print('rho_liq     :', obj.rho_liq)
        print('rho_gas     :', obj.rho_gas)
        print('-------------------------------------------------------------')
        obj_mix = PRMIX78(T, P, constants.Tcs, constants.Pcs, constants.omegas, zs, mws=constants.MWs, analytical=False)
        print('Z_liq       :', obj_mix.Z_liq)
        print('Z_gas       :', obj_mix.Z_gas)
        print('rho_liq     :', obj_mix.rho_liq)
        print('rho_gas     :', obj_mix.rho_gas)
        print('V_liq       :', obj_mix.V_liq)
        print('V_gas       :', obj_mix.V_gas)
        print('-------------------------------------------------------------')

    return Vs_dense,  rho_dense


# Todo: Promax seems to be switching to EOS with some smoothing tech applied.
# Todo: investigate negative density for decane
# Todo: 60F 1000psi 99% decane/1% ethane gives negative root

T_f = -27.67
T = UREG('%.15f degF' % T_f).to('kelvin')._magnitude
P_psi = 60
P = UREG('%.15f psi' % P_psi).to('pascal')._magnitude

# T=600, P=1000 (1 real root):  440, 29 (3 real roots)

molfrac_1 = 0.99
molfrac_2 = 1 - molfrac_1
#calc_Vs_sg_compressed({'ethane': molfrac_1, 'n-decane': molfrac_2}, T, P)
#calc_Vs_sg_compressed({'ethane': 1}, T, P)

# decane
constants = ChemicalConstantsPackage.constants_from_IDs(['n-decane'])
Tr = T / constants.Tcs[0]
Pr = P / constants.Pcs[0]
if Tr < 1 and Pr < 1:
    phase = 'Liquid'
elif Tr > 1 and Pr < 1:
    phase = 'Vapor'
elif Tr <  1 and Pr > 1:
    phase = 'Liquid'
else:
    phase = 'Supercritical'

obj = PR78(T, P, constants.Tcs[0], constants.Pcs[0], constants.omegas[0], constants.MWs[0], analytical=False)
print('-----')
print('T      :', UREG('%.15f degF' % T_f).to('kelvin')._magnitude)
print('P      :', UREG('%.15f pascal' % P).to('pascal')._magnitude)

print('Tc     :', constants.Tcs[0])
print('Pc     :', UREG('%.15f pascal' % constants.Pcs[0]).to('bar')._magnitude)
print('Tr     :', Tr)
print('Pr     :', Pr)
print('Phase  :', phase)
print('omega  :', constants.omegas[0])
print('-----')


print("obj.rho_liq :", obj.rho_liq)
print("obj.rho_gas :", obj.rho_gas)
print("obj.roots   :", obj.roots)
print("obj.roots_real   :", obj.roots_real)


a = PR78(Tc=507.6, Pc=3025000.0, omega=0.2975, T=400., P=1E6)
print(a.roots * 8.14 * 507.6 / 3025000.0)

#sys.exit()
# ----------------------------------------------------------------------------------------------------


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

Zs = []
a = np.array([-0.0734714 -3.21781229j, -0.0734714 +3.21781229j,
        0.60120786+0.j        ])
for item in a:
    real = item.real
    imag = item.imag
    Z_abs = np.sqrt(real**2 + imag**2)
    Zs.append(Z_abs)
Zs = np.array(Zs)