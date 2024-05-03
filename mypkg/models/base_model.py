import torch
import numpy as np
from utils.matrix import col_vec_fn


class BaseModel():
    """Base model
    """
    def __init__(self, Y, Z, X, basis_mat, ws=None):
        """
        args:
               Y: response values: n
               Z: matrix or vector of other covariates, (n) x q
               X: freq of data: (n) x d x npts
               basis_mat: Basis matrix of B-spline evaluated at some pts: npts x N
               ws: the weights used for approximating the integration: npts. 
        """
        if ws is None:
            ws = torch.ones(basis_mat.shape[0])/basis_mat.shape[0]
        self.ws = ws
        self.basis_mat = basis_mat
        self.X = X
        self.Z = Z
        self.Y = Y
        self.lin_tm_der = None
        self.log_lik_der1_vs = None # the First dervative of log_likelihood for each obs
        self.ints_der1 = None # the first dev of ints part w.r.t Gam
    
    
    def _obt_int_der1(self):
        """
        obt the first dev of ints part w.r.t Gam
        """
        basis_mat_trans = self.basis_mat.unsqueeze(0).unsqueeze(-1) # 1 x ncpts x N x 1
        X_trans = self.X.permute((0, 2, 1)).unsqueeze(2) # n x ncpts x 1 x d
        vec_part2_raw = basis_mat_trans*X_trans
        vec_part2_raw = vec_part2_raw.permute((0, 1, 3, 2)).flatten(2)
        vec_part2 = vec_part2_raw*self.ws.unsqueeze(0).unsqueeze(-1)
        self.ints_der1 = vec_part2.sum(axis=1)
        
    def _obt_lin_tm(self, alp, Gam):
        """Give the linear terms of likelihood fn
           args: 
               alp: parameters for Z: q
               Gam: parameters of B-spline: N x d
            return:
               lin_tm: the linear terms: scalar or vector of n
        """
        if self.ints_der1 is None:
            self._obt_int_der1()
        cov_tm = self.Z.matmul(alp)
        
        Gam_vec = col_vec_fn(Gam)
        inte_tm = self.ints_der1 @ Gam_vec
        
        lin_tm = cov_tm + inte_tm
        return lin_tm
    
    
    def _linear_term_der(self):
        """
        # derivative of linear term w.r.t (alp, N^{-1/2}*Gam)
        It is a constant
        """
        if self.ints_der1 is None:
            self._obt_int_der1()
        vec_part2 = self.ints_der1*np.sqrt(self.basis_mat.shape[1])
        lin_tm_der = torch.concat([self.Z, vec_part2], axis=1) #n x (q+dxN)
        self.lin_tm_der = lin_tm_der