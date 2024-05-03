import numpy as np
import torch

def cholesky_inv(A):
    """
    Use it only when A is large (>1000)
    This function calculates the inverse of a matrix using the Cholesky decomposition.
    
    Parameters:
    A (torch.Tensor): The input matrix to be inverted, a symmetric PD matrix.
    
    Returns:
    torch.Tensor: The inverted matrix.
    """
    U = torch.linalg.cholesky(A)
    Ainv = torch.cholesky_inverse(U)
    return Ainv


def conju_grad(A, vec, maxIter=1000, eps=1e-9):
        """ 
        A should be PD matrix
        solve x for  Ax = vec
        """
        xk = torch.zeros_like(vec)
        rk = vec - A @ xk 
        pk = rk
        if torch.norm(rk) <= eps:
            return xk
        
        for k in range(maxIter):
            alpk = torch.sum(rk**2) / torch.sum(pk * (A@pk))
            xk = xk + alpk * pk 
            
            rk_1 = rk
            rk = rk - alpk * (A@pk)
            
            if torch.norm(rk) <= eps:
                break 
                
            betk = torch.sum(rk**2)/torch.sum(rk_1**2)
            pk = rk + betk * pk
            
        if k == (maxIter-1):
            print(f"Conjugate gradient may not converge. The norm of rk is {torch.norm(rk)}")
        return xk

def Dmat_opt(v, q):
    """args:
            v: vector of q+dN
      return Dmat x v
       Dmat = [0_dNxq, I_dNxdN]
    """
    return v[q:]

def transDmat_opt(v, q):
    """args:
            v: vector of dN
      return Dmat^T x v
       Dmat = [0_dNxq, I_dNxdN]
    """
    zero_pool = torch.zeros(q)
    v_expand = torch.cat([zero_pool, v])
    return v_expand

def gen_Dmat(d, N, q):
    """Generate D matrix
       Dmat = [0_dNxq, I_dNxdN]
    """
    Dp1 = torch.zeros(d*N, q)
    Dp2 = torch.eye(d*N)
    D = torch.concatenate([Dp1, Dp2], dim=1)
    return D


def svd_inverse(mat, eps=0.999):
    """This fn is to caculate inverse for mat with SVD decom
    """
    res = torch.svd(mat)
    
    # to avoid non-inveritble matrix
    sing_v_ratios = torch.cumsum(res.S, dim=0)/torch.sum(res.S)
    idx = torch.where(sing_v_ratios >= eps)[0][0] + 1
    
    inv_mat = res.V[:, :idx] @ torch.diag(1/res.S[:idx]) @ res.U.T[:idx, :]
    return inv_mat

# eig decomp with sorted results by the mode of eigvals
def eig_sorted(mat):
    eigVals, eigVecs = np.linalg.eig(mat)
        # sort the eigvs and eigvecs
    sidx = np.argsort(-np.abs(eigVals))
    eigVals = eigVals[sidx]
    eigVecs = eigVecs[:, sidx]
    return eigVals, eigVecs


# vecterize a matrix by cols
# mat = [v1, v2, \ldots, vd]: vec(mat) = [v1T, v2T, \ldots, vdT]T
col_vec_fn = lambda x: x.T.flatten()

# mat = col_vec2mat_fn(col_vec_fn(mat))
def col_vec2mat_fn(v, nrow):
    """Transform a vec to a matrix by col
        mat=[c1, c2, , cn] v=[c1^T, \ldots, cn^T]^T
    """
    mat = v.reshape(-1, nrow).T
    return mat



def vecidx2matidx(num, ncol=None, nrow=None):
    """Return the idx in a matrix from a idx from vector
       both Indix from 0
    """
    num = num+1
    assert not ((ncol is None) and (nrow is None))
    if (ncol is not None) and (nrow is not None):
        print(f"Only need either, so ignore nrow={nrow}.")
    
    v = ncol if ncol is not None else nrow
    loc1 = num // v
    loc2 = num % v - 1
    if loc2 == -1:
        loc2 = v - 1
        loc1 = loc1 -1
    loc1, loc2 = int(loc1), int(loc2)
    if ncol is not None:
        return loc1, loc2
    else:
        return loc2, loc1