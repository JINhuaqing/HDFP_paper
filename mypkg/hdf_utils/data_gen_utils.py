import numpy as np
from utils.misc import load_pkl
from constants import DATA_ROOT
from scipy.ndimage import zoom


sc_org = load_pkl(DATA_ROOT/"sc_distmat/sc68_org.pkl", verbose=False)
dist_org = load_pkl(DATA_ROOT/"sc_distmat/dist68_org.pkl", verbose=False)
 # remove the small one
sc_org[sc_org<sc_org.max()*0.05] = 0


def get_sc(target_n_node):
    sc_new = zoom(sc_org, target_n_node/68, order=0)
    return sc_new

def get_sc_my(target_n_node):
    sc_new = gen_new_sc(target_n_node, sc_org)
    return sc_new
    
def get_dist(target_n_node):
    dist_new = zoom(dist_org, target_n_node/68, order=0)
    return dist_new


# may helpful, not I prefer zoom
def gen_new_sc(target_n_node, mat):
    """
    Generate a new symmetric matrix with the specified number of nodes.
    
    Parameters:
    target_n_node (int): The desired number of nodes in the output matrix.
    mat (numpy.ndarray): The input symmetric matrix.
    
    Returns:
    new_mat (numpy.ndarray): The generated symmetric matrix with target_n_node nodes.
    """
    # for sc, it has many symmetric properties, 
    # we have three parts
    # 1. upper tril of diag_bl
    # 2. upper tril of off_diag_bl
    # 3. diag of off_diag_bl
    
    n_node = mat.shape[0]
    hl_len = int(n_node/2)
    hl_len_target = int(target_n_node/2)
    
    diag_bl1 = mat[:hl_len, :hl_len]
    off_diag_bl1 = mat[:hl_len, hl_len:]
    
    
    up_tri_idxs = np.triu_indices(diag_bl1.shape[0], k=1);
    p1 = diag_bl1[up_tri_idxs]
    p2 = off_diag_bl1[up_tri_idxs]
    p3 = np.diag(off_diag_bl1)
    
    np1 = np.random.choice(p1, size=int(hl_len_target*(hl_len_target-1)/2))
    np2 = np.random.choice(p2, size=int(hl_len_target*(hl_len_target-1)/2))
    np3 = np.random.choice(p3, size=hl_len_target)
    
    new_mat_diag = np.zeros((hl_len_target, hl_len_target));
    new_mat_diag[np.triu_indices(hl_len_target, k=1)] = np1
    new_mat_diag = new_mat_diag + new_mat_diag.T
    
    new_mat_offdiag = np.zeros((hl_len_target, hl_len_target));
    new_mat_offdiag[np.triu_indices(hl_len_target, k=1)] = np2
    new_mat_offdiag = new_mat_offdiag + new_mat_offdiag.T + np.diag(np3) 
    
    new_mat = np.zeros((target_n_node, target_n_node));
    new_mat[:hl_len_target, :hl_len_target] = new_mat_diag
    new_mat[hl_len_target:, hl_len_target:] = new_mat_diag
    new_mat[hl_len_target:, :hl_len_target] = new_mat_offdiag
    new_mat[:hl_len_target, hl_len_target:] = new_mat_offdiag
    return new_mat