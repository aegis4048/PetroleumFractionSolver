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


n_paraffins = ['n-C6', 'n-C7', 'n-C8', 'n-C9', 'n-C10',
 'n-C11', 'n-C12', 'n-C13', 'n-C14', 'n-C15', 'n-C16', 'n-C17', 'n-C18', 'n-C19', 'n-C20',
 'n-C21', 'n-C22', 'n-C23', 'n-C24', 'n-C25', 'n-C26', 'n-C27', 'n-C28', 'n-C29', 'n-C30',
 'n-C31', 'n-C32', 'n-C33', 'n-C34', 'n-C35', 'n-C36', 'n-C37', 'n-C38', 'n-C39', 'n-C40',
 'n-C41', 'n-C42', 'n-C43', 'n-C44', 'n-C45', 'n-C46', 'n-C47', 'n-C48', 'n-C49', 'n-C50']

constants = ChemicalConstantsPackage.constants_from_IDs(n_paraffins)

constants = ChemicalConstantsPackage.constants_from_IDs(['n-C30'])

constants = ChemicalConstantsPackage.constants_from_IDs(['docosane'])
T = 288.7
Tc = constants.Tcs[0]
Vc = constants.Vcs[0]
omega = constants.omegas[0]
mw = constants.MWs[0]
Vs = chemicals.volume.COSTALD(T, Tc, Vc, omega)
rho = Vm_to_rho(Vs, mw)

# 60F = 288.7K

ghv_ideal_gas = Hc / V_molar
ghv_ideal_gas = UREG('%.15f joule/m^3' % ghv_ideal_gas).to('Btu/ft^3')._magnitude * -1

CONSTANTS = {
    "T_STANDARD": 288.7056,  # Temperature in Kelvin
    "P_STANDARD": 101325.0,    # Pressure in Pascal
    "R": 8.31446261815324,
    "MW_AIR": 28.9625,  # molecular weight of air at standard conditions, g/mol, GPA 2172-19
    "RHO_WATER": 999.0170125317171,  # density of water @60F, 1atm (kg/m^3) according to IAPWS-95 standard. Calculate rho at different conditions by:  chemicals.iapws95_rho(288.706, 101325) (K, pascal)
}


comp = {
    'n-butane': 50,
    'n-decane': 50
}
ptable = PropertyTable(comp)

def ideal_gas_molar_volume():
 """
 PV=nRT, where number of moles n=1. Rearranging -> V=RT/P
 R = 8.31446261815324 ((m^3-Pa)/(mol-K))
 T = 288.7056 K, 60F, standard temperature
 P = 101325 Pa, 1 atm, standard pressure
 :return: ideal gas molar volume in a standard condition (m^3/mol) = 0.023690421108823113
 """
 return CONSTANTS['R'] * CONSTANTS['T_STANDARD'] / CONSTANTS['P_STANDARD']


