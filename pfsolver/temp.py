import numpy as np
from scipy.optimize import newton
import correlations


ghv_liq = 4450
sg_liq = newton(lambda sg_liq: correlations.ghv_liq_sg_liq(ghv_liq, sg_liq), x0=0.65, maxiter=50)


sg_liq = newton(lambda ghv_liq: correlations.ghv_liq_sg_liq(ghv_liq, 0.62514), x0=0.65, maxiter=50)
print(sg_liq)
