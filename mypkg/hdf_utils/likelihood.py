# this file contains fns for likelihood
import torch

def obt_lin_tm(Z, X, alp, Gam, basis_mat, ws=None):
    """Give the linear terms of likelihood fn
       args: 
           Z: matrix or vector of other covariates, (n) x q
           X: freq of data: (n) x d x npts
           alp: parameters for Z: q
           Gam: parameters of B-spline: N x d
           basis_mat: Basis matrix of B-spline evaluated at some pts: npts x N
           ws: the weights used for approximating the integration: npts. 
        return:
           lin_tm: the linear terms: scalar or vector of n
    """
    assert X.dim() == Z.dim() + 1
    if X.dim() == 2:
        X = torch.unsqueeze(X, 0)
        Z = torch.unsqueeze(Z, 0)
    if ws is None:
        ws = torch.ones(basis_mat.shape[0])/basis_mat.shape[0]
        
    inte_tms = basis_mat.matmul(Gam).T * X
    inte_tm = torch.sum(inte_tms.sum(axis=1) *  ws, axis=1)
    
    cov_tm = Z.matmul(alp)
    
    lin_tm = cov_tm + inte_tm
    return lin_tm