import torch
import numpy as np

def GIC_fn(res, fct="BIC"):
    """
    This function calculates the generalized information criterion (GIC) based on the results from _run_fn().
    
    Args:
        res: The results obtained from the main function.
            est_Gam: estimation of Gam
            est_alp: estimation of alp
            _paras.n: sample size
            est_sigma2: estimation of variance of error
        fct: The function used to adjust the penalty. 
             If fct=2, GIC is equivalent to AIC.
             If fct=log(n), GIC is equivalent to BIC.
    
    Returns:
        The calculated GIC value.
    """
    if isinstance(fct, str):
        if fct.lower().startswith("bic"):
            fct = np.log(res._paras.n)
        elif fct.lower().startswith("aic"):
            fct = 2
    # 1 is the variance
    DoF = torch.sum(res.est_Gam.sum(axis=0)!=0) * res.est_Gam.shape[0] + len(res.est_alp) + 1
    return np.log(res.est_sigma2) + fct*DoF/res._paras.n


def GCV_fn(res):
    """
    This function calculates the generalized crossvalidation (GCV) based on the results from _run_fn().
    
    Args:
        res: The results obtained from the main function.
            est_Gam: estimation of Gam
            est_alp: estimation of alp
            _paras.n: sample size
            est_sigma2: estimation of variance of error
        
    Returns:
        The calculated GCV value.
    """
    # 1 is the variance
    DoF = torch.sum(res.est_Gam.sum(axis=0)!=0) * res.est_Gam.shape[0] + len(res.est_alp) + 1
    den = (1-(DoF/res._paras.n))**2
    return res.est_sigma2/den