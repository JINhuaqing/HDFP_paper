# this file contains fns to generate beta 
# based on the settings from sinica paper
# "Hypothesis Testing in Large-scale Functional Linear Regression"


import numpy as np
from numbers import Number
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler() # for console. 
ch.setLevel(logging.WARNING)
ch.setFormatter(formatter)

logger.addHandler(ch)

def fourier_basis_fn(x):
    """The function to regresion fourier basis at x. 
       The setting is based on paper 
           "Hypothesis Testing in Large-scale Functional Linear Regression"
       args:
           x: The locs to evaluate the basis
       return:
           fourier_basis: matrix of len(x) x 50
    """

    if isinstance(x, Number):
        xs = np.array([x])
    else:
        xs = np.array(x)
    
    fourier_basis = []
    for l in range(1, 51):
        if l == 1:
            fourier_basis.append(np.ones(len(xs)))
        elif l % 2 == 0:
            fourier_basis.append(np.sqrt(2)*np.cos(l*np.pi*(2*xs-1)))
        elif l % 2 == 1:
            fourier_basis.append(np.sqrt(2)*np.sin((l-1)*np.pi*(2*xs-1)))
    fourier_basis = np.array(fourier_basis).T
    return fourier_basis


def coef_fn(const=0.2):
    """The function to return the coefficients from the paper when const=0.2
        Note that when const = 0.2, it follows the sinica paper
        To get the exact sinica betas, c_j * coef_fn(0.2) (on Sep 4, 2023)
    """
    part1 = 1.2 - const * np.array([1, 2, 3, 4])
    part2 = 0.4 * (1/(np.arange(5, 51)-3))**(4)
    coefs = np.concatenate([part1, part2])
    return coefs

def gen_sini_Xthetas(srho, n, d, N_gen_basis=50):
    """ Generate thetas for X, X = thetas @ basis
        args:
            srho # the corr for X in sinica paper, default 0.3
            n: sample size
            d: num of ROIs, pn in paper
            N_gen_basis: the basis to generate the X and beta, default is 50
        
    """
    logger.debug(f"Corr is {srho:.2f}, sample size is {n:.0f}, "
                 f"num of ROIs is {d:.0f}, num of basis is {N_gen_basis:.0f}.")
    # generate thetaijk_tilde ~ N(0, k^(-2))
    thetas_tilde = np.random.randn(n, d, N_gen_basis);
    stds = 1/np.arange(1, N_gen_basis+1).reshape(1, 1, -1);
    thetas_tilde = thetas_tilde * stds;
    adj_fcts = srho ** np.abs(np.arange(1, d+1) - np.arange(1, d+1).reshape(-1, 1));
    thetas = np.matmul(adj_fcts, thetas_tilde);
    return thetas