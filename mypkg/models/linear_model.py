import torch
import numpy as np
from .base_model import BaseModel


class LinearModel(BaseModel):
    """Linear model
       To calculate test stat or estimate the parameters, 
       the value of sigma2 is not important.
       Here, I just keep sigma2hat as an option
    """
    
    def __init__(self, Y, Z, X, basis_mat, ws=None, sigma2=1):
        """
        args:
               Y: response values: n
               Z: matrix or vector of other covariates, (n) x q
               X: freq of data: (n) x d x npts
               basis_mat: Basis matrix of B-spline evaluated at some pts: npts x N
               ws: the weights used for approximating the integration: npts. 
               sigma2: Variance of the data Y, note that it is only for optimization
        """
        super().__init__(Y=Y, Z=Z, X=X, basis_mat=basis_mat, ws=ws)
        self.sigma2 = sigma2
        
    def log_lik(self, alp, Gam, sigma2hat=None):
        """Up to a constant
        args:
            sigma2hat: estimate of sigma2
        """
        Os = self._obt_lin_tm(alp, Gam)
        tm1 = -(self.Y - Os)**2
        if sigma2hat is None:
            rev = torch.mean(tm1/2/self.sigma2)
        else:
            rev = torch.mean(tm1/2/sigma2hat)
        return rev
    
    def log_lik_der1(self, alp, Gam, sigma2hat=None):
        """First dervative of log_likelihood w.r.t theta = [alp^T, N^{-1/2}*col_vec(Gam)^T]^T, 
            i.e., -loss/theta
        args:
            sigma2hat: estimate of sigma2
        """
        Os = self._obt_lin_tm(alp, Gam) # linear term
        
        
        if self.lin_tm_der is None:
            self._linear_term_der()
        tm2 = self.lin_tm_der #n x (q+dxN)
        
        if sigma2hat is None:
            tm1 = -(Os-self.Y)/self.sigma2 # n
        else:
            tm1 = -(Os-self.Y)/sigma2hat # n
        
        log_lik_der1_vs = tm1.unsqueeze(-1) * tm2 #n x (q+dxN)
        self.log_lik_der1_vs = log_lik_der1_vs
        log_lik_der1_v = log_lik_der1_vs.mean(axis=0) # (q+dxN)
        return log_lik_der1_v
    
    def log_lik_der2(self, alp, Gam, sigma2hat=None):
        """Second dervative of log_likelihood w.r.t theta = [alp^T, N^{-1/2}*col_vec(Gam)^T]^T
        args:
            sigma2hat: estimate of sigma2
        """
        if self.lin_tm_der is None:
            self._linear_term_der()
        tm1 = self.lin_tm_der #n x (q+dxN)
        
        if sigma2hat is None:
            log_lik_der2_vs = - tm1.unsqueeze(1) * tm1.unsqueeze(2)/self.sigma2
        else:
            log_lik_der2_vs = - tm1.unsqueeze(1) * tm1.unsqueeze(2)/sigma2hat
        log_lik_der2_v = log_lik_der2_vs.mean(axis=0) # (q+dxN) x (q+dxN)
        return log_lik_der2_v