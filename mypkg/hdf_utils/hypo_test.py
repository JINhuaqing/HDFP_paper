import torch
import numpy as np
from easydict import EasyDict as edict
from utils.matrix import col_vec_fn
import logging
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
ch = logging.StreamHandler() # for console. 
ch.setLevel(logging.WARNING)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

def get_Amat(k, Cmat, paras):
    """Get A matrix for hypothesis test
        k: Num of elements in S
        Cmat: Hypothesis matrix
        paras: parameters
                required: N, m, q
    """
    _paras = edict(paras.copy())
    
    m = Cmat.shape[-1]
    part1 = np.kron(Cmat, np.eye(_paras.N))
    part2 = np.zeros((m*_paras.N, _paras.q+(m+k)*_paras.N))
    part2[:, _paras.q:(_paras.q+m*_paras.N)] = np.eye(m*_paras.N)
    A = part1 @ part2
    return A

def MS2idxs(q, N, MS_unions):
    """This fn is to return the idxs to keep in mat Q, Sigma and vec Theta
    """
    idxs_all = [np.arange(0, q)]
    for cur_idx in MS_unions:
        idxs_all.append(np.arange(q+cur_idx*N, q+(cur_idx+1)*N))
    idxs_all = np.concatenate(idxs_all)
    return idxs_all

def obt_test_stat(model, est_alp, est_Gam, Cmat, paras):
    """
    Obtain the test statistics via the estimator.

    Args:
        model: The model you used, LinearModel or LogisticModel
        est_alp: Estimated alp parameters
        est_Gam: Estimated Gam parameters
        Cmat: Hypothesis mat, r x m
        paras: Dictionary containing parameters for the test
            - N: Number of samples
            - M_idxs: Indices of the M set
            - q: Number of ROIs
            - svdinv_eps_Q: Threshold for inverse of Q matrix
            - svdinv_eps_Psi: Threshold for inverse of Psi matrix
            - n: Number of observations

    Returns:
        T_v: Test statistic value
    """
    _paras = edict(paras.copy())
    est_theta = torch.cat([est_alp, col_vec_fn(est_Gam)/np.sqrt(_paras.N)])
    Q_mat = -model.log_lik_der2(est_alp, est_Gam)
    model.log_lik_der1(est_alp, est_Gam);
    Sig_mat = (model.log_lik_der1_vs.unsqueeze(-1) * model.log_lik_der1_vs.unsqueeze(1)).mean(axis=0) 
    # minus sign canceled
    
    
    # obtain the idxs to keep for test
    nonzero_idxs = torch.nonzero(torch.norm(est_Gam, dim=0)).reshape(-1).numpy()
    MS_unions = np.sort(np.union1d(_paras.M_idxs, nonzero_idxs))
    keep_idxs_test = MS2idxs(_paras.q, _paras.N, MS_unions)
    
    # A mat
    k = len(np.setdiff1d(nonzero_idxs, _paras.M_idxs))
    Amat = torch.Tensor(get_Amat(k, Cmat, _paras))
    
    # calculate Test stats
    Q_mat_part = Q_mat[keep_idxs_test][:, keep_idxs_test]
    Q_mat_part_inv = torch.linalg.pinv(Q_mat_part, hermitian=True, rtol=_paras.svdinv_eps_Q)
    
    Sig_mat_part = Sig_mat[keep_idxs_test][:, keep_idxs_test]
    Psi = Amat @ Q_mat_part_inv @ Sig_mat_part @ Q_mat_part_inv @ Amat.T
    
    T_p1 = Amat @ est_theta[keep_idxs_test]
    Psi_inv = torch.linalg.pinv(Psi, hermitian=True, rtol=_paras.svdinv_eps_Psi)
    T_v = T_p1 @ Psi_inv @ T_p1 * _paras.n 
    return T_v

def obt_test_stat_simple(Q_mat, Sig_mat, est_alp, est_Gam, Cmat, paras):
    """
    Obtain the test statistics via the estimator, a simpler version, not need model

    Args:
        Q_mat: Second der of model
        Sig_mat: first der 
        est_alp: Estimated alp parameters
        est_Gam: Estimated Gam parameters
        Cmat: Hypothesis mat, r x m
        paras: Dictionary containing parameters for the test
            - N: Number of samples
            - M_idxs: Indices of the M set
            - q: Number of ROIs
            - svdinv_eps_Q: Threshold for inverse of Q matrix
            - svdinv_eps_Psi: Threshold for inverse of Psi matrix
            - n: Number of observations

    Returns:
        T_v: Test statistic value
    """
    _paras = edict(paras.copy())
    est_theta = torch.cat([est_alp, col_vec_fn(est_Gam)/np.sqrt(_paras.N)])
    # minus sign canceled
    
    
    # obtain the idxs to keep for test
    nonzero_idxs = torch.nonzero(torch.norm(est_Gam, dim=0)).reshape(-1).numpy()
    MS_unions = np.sort(np.union1d(_paras.M_idxs, nonzero_idxs))
    keep_idxs_test = MS2idxs(_paras.q, _paras.N, MS_unions)
    
    # A mat
    k = len(np.setdiff1d(nonzero_idxs, _paras.M_idxs))
    Amat = torch.Tensor(get_Amat(k, Cmat, _paras))
    
    # calculate Test stats
    Q_mat_part = Q_mat[keep_idxs_test][:, keep_idxs_test]
    Q_mat_part_inv = torch.linalg.pinv(Q_mat_part, hermitian=True, rtol=_paras.svdinv_eps_Q)
    
    Sig_mat_part = Sig_mat[keep_idxs_test][:, keep_idxs_test]
    Psi = Amat @ Q_mat_part_inv @ Sig_mat_part @ Q_mat_part_inv @ Amat.T
    
    T_p1 = Amat @ est_theta[keep_idxs_test]
    Psi_inv = torch.linalg.pinv(Psi, hermitian=True, rtol=_paras.svdinv_eps_Psi)
    T_v = T_p1 @ Psi_inv @ T_p1 * _paras.n 
    return T_v

def obt_test_stat_simple2(Q_mat_part, Sig_mat_part, est_alp, est_Gam, Cmat, paras):
    """
    Obtain the test statistics via the estimator, even simpler than simple 

    Args:
        Q_mat_part: Second der of model
        Sig_mat_part: first der 
        est_alp: Estimated alp parameters
        est_Gam: Estimated Gam parameters
        Cmat: Hypothesis mat, r x m
        paras: Dictionary containing parameters for the test
            - N: Number of samples
            - M_idxs: Indices of the M set
            - q: Number of ROIs
            - svdinv_eps_Q: Threshold for inverse of Q matrix
            - svdinv_eps_Psi: Threshold for inverse of Psi matrix
            - n: Number of observations

    Returns:
        T_v: Test statistic value
    """
    _paras = edict(paras.copy())
    est_theta = torch.cat([est_alp, col_vec_fn(est_Gam)/np.sqrt(_paras.N)])
    
    # obtain the idxs to keep for test
    nonzero_idxs = torch.nonzero(torch.norm(est_Gam, dim=0)).reshape(-1).numpy()
    MS_unions = np.sort(np.union1d(_paras.M_idxs, nonzero_idxs))
    keep_idxs_test = MS2idxs(_paras.q, _paras.N, MS_unions)
    
    # A mat
    k = len(np.setdiff1d(nonzero_idxs, _paras.M_idxs))
    Amat = torch.Tensor(get_Amat(k, Cmat, _paras))
    
    # calculate Test stats
    Q_mat_part_inv = torch.linalg.pinv(Q_mat_part, hermitian=True, rtol=_paras.svdinv_eps_Q)
    Psi = Amat @ Q_mat_part_inv @ Sig_mat_part @ Q_mat_part_inv @ Amat.T
    Psi_inv = torch.linalg.pinv(Psi, hermitian=True, rtol=_paras.svdinv_eps_Psi)
    #pdb.set_trace()
    
    T_p1 = Amat @ est_theta[keep_idxs_test]
    T_v = T_p1 @ Psi_inv @ T_p1 * _paras.n 
    return T_v

def obt_test_stat_simple3(est_sigma2, Q_mat_part, est_alp, est_Gam, Cmat, paras):
    """
    Obtain the test statistics via the estimator, even simpler than simple2
    Note that it is only valid for linear

    Args:
        est_sigma2: estimation of variance of error
        Q_mat_part: Second der of model
        est_alp: Estimated alp parameters
        est_Gam: Estimated Gam parameters
        Cmat: Hypothesis mat, r x m
        paras: Dictionary containing parameters for the test
            - N: Number of samples
            - M_idxs: Indices of the M set
            - q: Number of ROIs
            - svdinv_eps_Q: Threshold for inverse of Q matrix
            - svdinv_eps_Psi: Threshold for inverse of Psi matrix
            - n: Number of observations

    Returns:
        T_v: Test statistic value
    """
    _paras = edict(paras.copy())
    est_theta = torch.cat([est_alp, col_vec_fn(est_Gam)/np.sqrt(_paras.N)])
    
    # obtain the idxs to keep for test
    nonzero_idxs = torch.nonzero(torch.norm(est_Gam, dim=0)).reshape(-1).numpy()
    MS_unions = np.sort(np.union1d(_paras.M_idxs, nonzero_idxs))
    keep_idxs_test = MS2idxs(_paras.q, _paras.N, MS_unions)
    
    # A mat
    k = len(np.setdiff1d(nonzero_idxs, _paras.M_idxs))
    Amat = torch.Tensor(get_Amat(k, Cmat, _paras))
    
    # calculate Test stats
    Q_mat_part_inv = torch.linalg.pinv(Q_mat_part, hermitian=True, rtol=_paras.svdinv_eps_Q)
    Psi = est_sigma2*Amat @ Q_mat_part_inv @ Amat.T/_paras.norminal_sigma2
    Psi_inv = torch.linalg.pinv(Psi, hermitian=True, rtol=_paras.svdinv_eps_Psi)
    
    T_p1 = Amat @ est_theta[keep_idxs_test]
    T_v = T_p1 @ Psi_inv @ T_p1 * _paras.n 
    return T_v