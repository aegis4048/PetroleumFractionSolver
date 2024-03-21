import numpy as np
import config

import pint
UREG = pint.UnitRegistry()


def mw_scn_model_SCN(mw, scn):
    """
    source: [1] eq-7.
    working range: SCN 6-50, MW 82-698, sg_liq 0.690-0.947, Tb 337-851K, 606-1542R, 146.9-1072F
    """
    return -mw + 14 * scn - 4


def Tb_scn_model_SCN(Tb_R, scn):
    """
    source: [1] eq-3.
    working range: SCN 6-50, MW 82-698, sg_liq 0.690-0.947, Tb 337-851K, 606-1542R, 146.9-1072F
    """
    Tb_K = UREG('%.15f rankine' % Tb_R).to('kelvin').magnitude
    return -Tb_K + 1090 - np.exp(6.9955 - 0.11193 * scn**(2/3))


def sg_liq_scn_model_SCN(sg_liq, scn):
    """
    source: [1] eq-4, and my article
    working range: SCN 6-50, MW 82-698, sg_liq 0.690-0.947, Tb 337-851K, 606-1542R, 146.9-1072F
    """
    return -sg_liq + 1.07 - np.exp(3.65097 - 3.8864 * scn**0.1)


def sg_mw_model_SCN_KF(sg, mw):
    """
    source: my article (eq 1a and 1b)
    working range: 84 < MW < 626 | 0.685 < sg < 0.937 | 606.7 < Tb < 1486.7
    """
    if mw < 136:
        return -sg -3.16160517e-01 + 2.32199514e-02 * mw - 1.71830840e-04 * mw ** 2 + 4.43897650e-07 * mw ** 3
    else:
        return -sg + 1.103 - np.exp(2.934 - 2.485 * mw**0.1)


def sg_mw_model_SCN_RA(sg, mw):
    """
    source: my article (eq 1a and 1b)
    working range: 82 < MW < 698 | 0.690 < sg < 0.947 | 606.6 < Tb < 1531.8
    """

    if mw < 136:
        return -sg -6.94728285e-02 + 1.71874520e-02 * mw - 1.21389818e-04 * mw ** 2 + 3.01844635e-07 * mw ** 3
    else:
        return -sg + 1.078 - np.exp(3.403 - 2.824 * mw**0.1)


def Tb_mw_model_SCN_KF(Tb_R, mw):
    """
    source: my article (eq A)
    unit: Tb in R
    working range: 84 < MW < 626 | 0.685 < sg < 0.937 | 606.7 < Tb < 1486.7
    """
    return -Tb_R + 1.94302832e+03 - np.exp(7.56781597 - 1.96435739e-02 * mw**(2/3))


def Tb_mw_model_SCN_RA(Tb_R, mw):
    """
    source: my article (eq A)
    unit: Tb in R
    working range: 82 < MW < 698 | 0.690 < sg < 0.947 | 606.6 < Tb < 1531.8
    """
    return -Tb_R + 1.94547935e+03 - np.exp(7.56885268 - 1.96209827e-02 * mw**(2/3))


def mw_ghv_paraffinic(mw, ghv):
    """
    source: my article - update this later
    unit: ghv in Btu/scf
    notes: linear correlation for paraffinic plus fractions
    """
    return -mw + 0.0188 * ghv - 2.758


def mw_ghv_aromatic(mw, ghv):
    """
    source: my article - update this later
    unit: ghv in Btu/scf
    notes: linear correlation for aromatic plus fractions
    """
    return -mw + 0.0186 * ghv + 10.326


def mw_ghv(mw, ghv, aromatic_fraction):
    """
    source: my article - update this later
    unit: ghv in Btu/scf
    notes: linear correlation for plus fractions
    """
    coef = 0.0188 * (1 - aromatic_fraction) + 0.0186 * aromatic_fraction
    intercept = -2.758 * (1 - aromatic_fraction) + 10.326 * aromatic_fraction
    return -mw + coef * ghv + intercept


def mw_GHV_gas_xa(mw, GHV_gas, xa):

    # xa = aromatic fraction
    # two different linear regression coefficients.
    # Naphthenes are not considered because they are nearly identical to paraffins for MW vs. GHV correlation
    coef_paraffin = 0.0188
    intercept_paraffin = -2.758
    coef_aromatic = 0.0186
    intercept_aromatic = 10.326

    return -mw + coef_paraffin * GHV_gas * (1 - xa) + intercept_paraffin * (1 - xa) + coef_aromatic * GHV_gas * xa + intercept_aromatic * xa



def ideal_gas_molar_volume():
    """
    PV=nRT, where number of moles n=1. Rearranging -> V=RT/P
    R = 8.31446261815324 ((m^3-Pa)/(mol-K))
    T = 288.7056 K, 60F, standard temperature
    P = 101325 Pa, 1 atm, standard pressure
    :return: ideal gas molar volume in a standard condition (m^3/mol)
    """
    return config.constants['R'] * config.constants['T_STANDARD'] / config.constants['P_STANDARD']


def ghv_liq_ghv_gas_mw(ghv_liq, ghv_gas, mw):
    """
    source: None, this is a simple unit conversion
    unit: ghv_liq in Btu/lbm ghv_gas in Btu/scf, mw in lbm/lbmol
    """
    V_molar = ideal_gas_molar_volume()  # fixed 0.0236 m^3/mol (379.48 ft^3/mol) at standard conditions for all compounds
    V_molar = UREG('%.15f m^3/g' % V_molar).to('ft^3/lb').magnitude  # unit conversion
    return -ghv_liq + ghv_gas * V_molar / mw


def ghv_liq_sg_liq(ghv_liq, sg_liq):
    """
    source: [3] (eq 7.73)
    working range: intended for crude oil, no specific range given
    notes: the original is ghv_liq = 51.9 - 8.8 * sg_liq**2 in metric, but I converted it to british units
    unit: ghv_liq in Btu/lb
    """
    return -ghv_liq + 22312.8 - 3783.3 * sg_liq**2


def mw_sg_gas(mw, sg_gas):
    """
    source: [1] (eq 2.6)
    notes: valid for all gas
    """
    sg_air = config.constants['MW_AIR']
    return mw / sg_air - sg_gas


def kelvin_to_rankine(K):

    R = K * 9/5
    return R


def calc_RI(Tb, sg_liq):
    """
    units: Tb in R
    notes: refractive index at 68F & 1 atm. Reference T @68F for RI is what's used by the PNA correlation.
    source: [2] Procedure 2B5.1
    working range: RI 1.35-1.55, sg_liq 0.63-0.97, Tb 310.9-783K, 559.6-1409.6R, 100-950F,
    Todo: https://www.mdpi.com/2227-9717/11/8/2328 - this paper covers models that work for wider working range for heavier fractions (ex: bitumens that exceed sg>1)
    """
    I = 2.266e-2 * np.exp(3.905e-4 * Tb + 2.468*sg_liq - 5.704e-4 * Tb * sg_liq) * Tb**0.0572 * sg_liq**(-0.720)

    return ((1 + 2*I)/(1 - I))**0.5


def calc_RI_intercept(sg_liq, RI):
    """
    notes: sg_liq is density at 68F, but density at 60F is fine too because liquid is incompressible at near standard conditions
    source: [2] Procedure 2B4.1 (eq 2B4.1-4)
    working range: RI 1.35-1.55, sg_liq 0.63-0.97, Tb 310.9-783K, 559.6-1409.6R, 100-950F,
    """
    return RI - sg_liq / 2


def calc_v100_v210(Tb, sg_liq):
    """
    source: [4] (eq-3 to eq-16)
    notes: default method by Promax, but 10% error compared to promax. I confirmed that I did this by comparing with the example calculation in the original paper, Promax seems to have implemented wrong.
    working range: -30 to 110 API, 600 to 1800R Tb
    errors:
    """
    # Tb in Rankine
    # v100 = kinematic viscosity at 100F, cSt
    # v210 = kinematic viscosity at 210F, cSt

    Tco = Tb * (
                0.533272 + 0.191017e-3 * Tb + 0.779681e-7 * Tb ** 2 - 0.284376e-10 * Tb ** 3 + 0.959468e28 / Tb ** 13) ** -1
    a = 1 - Tb / Tco
    sg_o = 0.843593 - 0.128624 * a - 3.36159 * a ** 3 - 13749.5 * a ** 12
    delta_sg = sg_liq - sg_o

    v210o = np.exp(4.73227 - 27.0975 * a + 49.4491 * a ** 2 - 50.4706 * a ** 4) - 1.5
    v100o = np.exp(0.801621 + 1.37179 * np.log(v210o))

    x = np.abs(1.99873 - 56.7394 / Tb ** 0.5)

    f1 = 1.33932 * x * delta_sg - 21.1141 * delta_sg ** 2 / Tb ** 0.5
    f2 = x * delta_sg - 21.1141 * delta_sg ** 2 / Tb ** 0.5

    v100 = np.exp(np.log(v100o + 450 / Tb) * ((1 + 2 * f1) / (1 - 2 * f1)) ** 2) - 450 / Tb
    v210 = np.exp(np.log(v210o + 450 / Tb) * ((1 + 2 * f2) / (1 - 2 * f2)) ** 2) - 450 / Tb

    return v100, v210


def calc_SUS_100(v100):
    """
    source: [3] eq 1.17
    working range: works for all range
    notes: conversion of kinematic viscosity at 100F to Saybolt Universal Seconds (SUS) in seconds
    """
    return 4.6324 * v100 + (1 + 0.03264*v100)/((3930.2 + 262.7*v100 + 23.97*v100**2 + 1.646*v100**3)* 10**-5)


def calc_VGC(sg_liq, SUS_100):
    """
    source: [2] Procedure 2B4.1 (eq 2B4.1-5.1)
    working range: intended to be used for MW 200-600 for PNA composition calculation
    """
    if SUS_100 < 39:
        SUS_100 = 39
    return (10*sg_liq - 1.0752 * np.log10(SUS_100 - 38)) / (10 - np.log10(SUS_100 - 38))


def calc_VGF(sg_liq, v100):
    """
    source: [2] Procedure 2B4.1 (eq 2B4.1-6.1)
    working range: intended to be used for MW 200-600 for PNA composition calculation
    """
    return -1.816 + 3.484 * sg_liq - 0.1156 * np.log(v100)


def calc_PNA_comp(MW, VG, RI_intercept):
    """
    source: [2] Procedure 2B4.1 (eq 2B4.1-1 to 2B4.1-3)
    working range: MW 70 to 600
    """

    if MW > 200:
        a = 2.5737
        b = 1.0133
        c = -3.573
        d = 2.464
        e = -3.6701
        f = 1.96312
        g = -4.0377
        h = 2.6568
        i = 1.60988

    else:
        a = -13.359
        b = 14.4591
        c = -1.41344
        d = 23.9825
        e = -23.333
        f = 0.81517
        g = -9.6235
        h = 8.8739
        i = 0.59827

    xp = a + b * RI_intercept + c * VG
    xn = d + e * RI_intercept + f * VG
    xa = g + h * RI_intercept + i * VG

    return xp, xn, xa




"""
.. [1] Riazi, M.R., and Al-Sahhaf, T.A.: "Physical Properties of Heavy Petroleum Fractions and Crude Oils" (1996), Fluid Phase Equilibria 117
.. [2] API Technical Databook 1997
.. [3] Riazi, M. R.: "Characterization and Properties of Petroleum Fractions," first edition (1985), West Conshohocken, Pennsylvania: ASTM International`
.. [4] Twu, C.H.: "An internally consistent correlation for predicting the critical properties and molecular weights of petroleum and coal-tar liquids," Fluid Phase Equilibria 16 (1985) 137-150
.. [5] Kim, Eric.: "aegis4048.github.io"

.. [3] Nourozieh, H., Kariznovi,  M., and Abedi, J.: "Measurement and Modeling of Solubility and Saturated - Liquid Density and Viscosity for Methane / Athabasca - Bitumen Mixtures," paper SPE-174558-PA (2016). `(link) <https://onepetro.org/SJ/article/21/01/180/205922/Measurement-and-Modeling-of-Solubility-and>`__
.. [4] API Technical Databook GPA Publication 2145-82
.. [5] Maxwell's Databook on Hydrocarbons
"""