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

T_f = 500
T = UREG('%.15f degF' % T_f).to('kelvin')._magnitude
P_psi = 20
P = UREG('%.15f psi' % P_psi).to('pascal')._magnitude

molfrac_1 = 0.2              # ethane
molfrac_2 = 1 - molfrac_1    # decane
comp_dict = {'ethane': molfrac_1, 'n-decane': molfrac_2}

comp_dict, total_comp = utilities.normalize_composition_dict(comp_dict)
comp_vals = np.array(list(comp_dict.values()))
names = list(comp_dict.keys())
constants = ChemicalConstantsPackage.constants_from_IDs(names)
MW_mix = np.sum(np.array(constants.MWs) * np.array(list(comp_dict.values())))


obj = PRMIX78(T, P, constants.Tcs, constants.Pcs, constants.omegas, zs=comp_vals, mws=constants.MWs, analytical=False)
print("obj.roots      :", obj.roots)
print("obj.roots_real :", obj.roots_real)
print("obj.V_liq      :", obj.V_liq)
print("obj.rho_liq    :", obj.rho_liq)
print("obj.Z_liq      :", obj.Z_liq)
print("obj.V_gas      :", obj.V_gas)
print("obj.Z_gas      :", obj.Z_gas)
print("obj.rho_gas    :", obj.rho_gas)
print("obj.phase      :", obj.phase)





