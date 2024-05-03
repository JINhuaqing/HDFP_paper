# this file contains fns for generating data for simulation
import torch
import numpy as np
from numbers import Number
from pathlib import Path
from tqdm import trange
from joblib import Parallel, delayed
from easydict import EasyDict as edict
from copy import deepcopy

from constants import DATA_ROOT, MIDRES_ROOT
from utils.misc import  _set_verbose_level, _update_params, load_pkl, save_pkl
from utils.functions import logit_fn
from hdf_utils.utils import integration_fn
from hdf_utils.fns_sinica import  fourier_basis_fn
from splines import obt_bsp_obasis_Rfn, obt_bsp_basis_Rfn_wrapper
from .data_gen_utils import get_dist, get_sc_my
from .sgm import SGM
from .fns_sinica import gen_sini_Xthetas

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 

#AD_ts = load_pkl(DATA_ROOT/"AD_vs_Ctrl_ts/AD88_all.pkl")
#Ctrl_ts = load_pkl(DATA_ROOT/"AD_vs_Ctrl_ts/Ctrl92_all.pkl")
#ts_data = np.concatenate([AD_ts, Ctrl_ts], axis=0)
#std_ts_data = (ts_data - ts_data.mean(axis=2)[:, :, np.newaxis])/ts_data.std(axis=2)[:, :, np.newaxis]
#_cur_dir = Path(__file__).parent
#_database_dir = Path(_cur_dir/"../../data/fooof_data_ADvsCtrl")

def gen_simu_psd(n, d, freqs, prior_sd=10, n_jobs=1, is_prog=False, is_std=True):
    """
    Generate simulated power spectral density (PSD) data.

    Args:
        n (int): Number of samples to generate.
        d (int): Number of regions of interest (ROIs).
        freqs (array-like): A vector of frequencies, in Hz.
        prior_sd (float, optional): The prior standard deviation of the SGM parameters. Default is 10.
        n_jobs (int, optional): Number of jobs to generate the data. Default is 1.
        is_prog (bool, optional): Whether to show progress bar or not. Default is False.
        is_std (bool, optional): Whether to std the psd across freq axis or not. Default is True.

    Returns:
        psds (numpy.ndarray): An n x d x len(freqs) array of simulated PSD data.
    """
    C = get_sc_my(d);
    D = get_dist(d);
    sgmmodel = SGM(C, D, freqs=freqs);
    cur_parass = np.random.randn(n, 7)*prior_sd
    def fun(ix):
        cur_psd = sgmmodel(cur_parass[ix], is_std)
        return cur_psd
    if is_prog:
        pbar = trange(n)
    else:
        pbar = range(n)
    with Parallel(n_jobs=n_jobs) as parallel:
        psds = parallel(delayed(fun)(i) for i in pbar)
    #psds = Parallel(n_jobs=n_jobs)(delayed(fun)(i) for i in pbar);
    psds = np.array(psds)
    return psds

def obt_maskmat(sel_seqs):
    """Return the mask matrix for generating new MEG data
    """
    num_sel, num_pt = sel_seqs.shape
    seg_len = int(num_pt/num_sel)
    mask_mat = np.zeros_like(sel_seqs)
    for ix in range(num_sel):
        low, up = ix*seg_len, (ix+1)*seg_len
        if ix == (num_sel-1):
            up = num_pt
        mask_mat[ix, low:up] = 1
    return mask_mat


def gen_ts_single(base_data, num_sel, sub_idx):
    """
    This function is to generate one single MEG seq in time domain. 
        args:
            base_data: The dataset used for generaing
            num_sel: Num of seqs used for jointing
            sub_idx: The data idx used for generating
    """
    tmp_stds = np.zeros(10)
    # some MEG dataset is very bad, to avoid zero-std
    while np.min(tmp_stds) <= 1e-5:
        sel_roi_idx = np.sort(np.random.choice(base_data.shape[1], num_sel, False))
        sel_seqs = base_data[sub_idx, sel_roi_idx, :]
        maskmat = obt_maskmat(sel_seqs)
        sel_seqs_masked = sel_seqs * maskmat
        
        # make each segment standardized
        sel_seqs_masked[maskmat==0] = np.NAN
        tmp_stds = np.nanstd(sel_seqs_masked, axis=1)
        tmp_means = np.nanmean(sel_seqs_masked, axis=1)
    
    simu_seq = np.nansum((sel_seqs_masked - tmp_means[:, np.newaxis])/tmp_stds[:, np.newaxis], axis=0)
    
    simu_seq = (simu_seq - simu_seq.mean())/simu_seq.std()
    
    return simu_seq

def gen_simu_ts(n, d, num_sel, base_data=None, decimate_rate=10, verbose=False):
    """generate time_series data based on AD vs ctrl
        args:
            n: Num of sps to generate
            d: Num of ROIs
            num_sel: Num of selected curve to get the shape
            base_data: The dataset used for generaing
        return:
            simu_ts: n x d x len array
    """
    if base_data is None:
        base_data = std_ts_data[:, :, ::decimate_rate]
    simu_tss = []
    if verbose:
        pbar = trange(n)
    else:
        pbar = range(n)
    for ix in pbar:
        sel_sub_idx = np.random.choice(base_data.shape[0], 1)
        cur_ts = []
        for iy in range(d):
            cur_ts.append(gen_ts_single(base_data, num_sel, sel_sub_idx))
        cur_ts = np.array(cur_ts)
        simu_tss.append(cur_ts)
    return np.array(simu_tss)



def gen_covs(n, types_, std_con=True):
    """Generate the covariates for simulated datasets
        args:
            n: The num of obs to generate
            types_: A list of types for each col
                    num: Discrete with num classes
                    "c": Continuous type
                    "int" intercept
    """
    covs = []
    for type_ in types_:
        if isinstance(type_, Number):
            cur_cov = np.random.choice(int(type_), n)
        else:
            type_ = type_.lower()
            if type_.startswith("c"):
                cur_cov = np.random.randn(n)
                if std_con: 
                    cur_cov = (cur_cov-cur_cov.mean())/cur_cov.std()
            elif type_.startswith("int"):
                cur_cov = np.ones(n)
            else:
                pass
        covs.append(cur_cov)
    covs = np.array(covs).T
    return covs



def _is_exists(tmp_paras):
    """
    Check if a file with the given parameters exists.

    Args:
    tmp_paras:
        d (int): The value of d in the file name.
        n (int): The value of n in the file name.
        npts:
        is_std
        seed (int): The seed value in the file name.

    Returns:
    bool or Path: Returns the file path if the file exists, otherwise returns False.
    """
    _get_n = lambda fil: int(fil.stem.split("_")[2].split("-")[-1])
    fils = MIDRES_ROOT.glob(f"PSD_d-{tmp_paras.d}_n-*npts-{tmp_paras.npts}_is_std-{tmp_paras.is_std}")
    # We do not need fil with n as we know the data with corresponding seed does not exist
    fils = [fil for fil in fils if _get_n(fil) !=tmp_paras.n]
    if len(fils) == 0:
        return False
    else:
        fils = sorted(fils, key=_get_n)
        ns = np.array([_get_n(fil) for fil in fils])
        idxs = np.where(tmp_paras.n <= ns)[0]
        if len(idxs) == 0:
            return False
        else:
            fil =fils[idxs[0]]
            path = MIDRES_ROOT/fil/f"seed_{tmp_paras.seed}.pkl"
            return path if path.exists() else False
def _get_filename(params, npts=None):
    keys = ["d", "n", "npts", "is_std"]
    if npts is not None:
        params = deepcopy(params)
        params["npts"] = npts
    folder_name = 'PSD_'+'_'.join(f"{k}-{params[k]}" for k in keys)
    return folder_name + f'/seed_{params.seed}.pkl'


def gen_simu_psd_dataset(n, d, q, types_, gt_alp, gt_beta, freqs, 
                 data_type="linear", data_params={}, 
                 seed=0, is_std=False, verbose=2, is_gen=False):
    """
    Generate simulated data for all parameters.

    Args:
        seed (int): Seed for random number generator.
        n (int): Number of samples.
        d (int): Number of dimensions.
        q (int): Number of covariates.
        types_ (list): List of types for generating covariates.
        gt_alp (list): List of ground truth alpha values.
        gt_beta (list): List of ground truth beta values.
        freqs (list): List of frequencies for generating simulated PSD.
        is_std (bool): Whether std psd or not, if not, center it 
        data_params (dict): Dict of other data params
            - sigma2 (float): Variance of the noise for linear model
            - err_dist (str): Distribution of the noise for linear model, norm or t
            - psd_noise_sd (float): the level of noise added to PSD
        verbose(bool): 0-3
        is_gen(bool): Only for generating or not. If True, only checking or generating X, not return anything.

    Returns:
        all_data (dict): Dictionary containing the following simulated data:
            - X (torch.Tensor): Tensor of shape (n, d, npts) containing the simulated PSD.
            - Y (torch.Tensor): Tensor of shape (n,) containing the response variable.
            - Z (torch.Tensor): Tensor of shape (n, q) containing the covariates.
    """
    np.random.seed(seed)
    _set_verbose_level(verbose, logger)
    data_type = data_type.lower()
    if data_type.startswith("linear"):
        data_params_def = {
            "sigma2": 1, 
            "err_dist": "norm", 
            "psd_noise_sd": 1 if is_std else 10,
        }
    elif data_type.startswith("logi"):
        data_params_def = {
            "psd_noise_sd": 1 if is_std else 10,
        }
    else:
        raise ValueError(f"{data_type} is not supported now.")
    data_params = _update_params(data_params, data_params_def, logger)
    
    # simulated PSD
    assert len(types_) == q
    assert len(gt_alp) == q
    tmp_paras = edict()
    tmp_paras.seed = seed 
    tmp_paras.n = n
    tmp_paras.d = d
    tmp_paras.npts = len(freqs)
    tmp_paras.is_std = is_std
    con_idxs = [typ =="c" for typ in types_]
    
    # get simu_curvs for Y
    #freqs0 = np.linspace(freqs[0], freqs[-1], 101)
    #file_path = MIDRES_ROOT/_get_filename(tmp_paras, npts=101)
    #if file_path.exists():
    #    simu_curvs0 = load_pkl(file_path, verbose=verbose>=2)
    #else:
    #    ofil =  _is_exists(tmp_paras)
    #    if ofil:
    #        simu_curvs0 = load_pkl(ofil, verbose=verbose>=2)
    #    else:
    #        simu_curvs0 = gen_simu_psd(n, d, freqs0, prior_sd=10, n_jobs=28, is_prog=verbose>=2, is_std=is_std)
    #        if not is_std:
    #            simu_curvs0 = simu_curvs0 - simu_curvs0.mean(axis=-1, keepdims=True); # not std, but center it
    #        save_pkl(file_path, simu_curvs0, verbose=verbose>=2)
    #simu_curvs0 = simu_curvs0[:n]
    ##simu_curvs0 = (simu_curvs0 + np.random.randn(*simu_curvs0.shape)*data_params.psd_noise_sd)
    
    # get simu_curvs for X, 
    file_path = MIDRES_ROOT/_get_filename(tmp_paras)
    if file_path.exists():
        if is_gen:
            return None
        simu_curvs = load_pkl(file_path, verbose=verbose>=2)
    else:
        ofil =  _is_exists(tmp_paras)
        if ofil:
            if is_gen:
                return None
            simu_curvs = load_pkl(ofil, verbose=verbose>=2)
        else:
            simu_curvs = gen_simu_psd(n, d, freqs, prior_sd=10, n_jobs=28, is_prog=verbose>=2, is_std=is_std)
            if not is_std:
                simu_curvs = simu_curvs - simu_curvs.mean(axis=-1, keepdims=True); # not std, but center it
            save_pkl(file_path, simu_curvs, verbose=verbose>=2)
    if is_gen:
        return None
    simu_curvs = simu_curvs[:n]
    simu_curvs = (simu_curvs + np.random.randn(*simu_curvs.shape)*data_params.psd_noise_sd)*1 # larger
    simu_covs = gen_covs(n, types_)
    
    # linear term and Y
    fs = np.sum(gt_beta.T * simu_curvs, axis=1) # n x npts
    int_part = integration_fn(fs.T, "sim").numpy()
    cov_part = simu_covs @ gt_alp 
    
    # linear term
    lin_term = cov_part + int_part
    
    # Y 
    if data_type.startswith("linear"):
        if data_params["err_dist"].lower().startswith("t"):
            errs_raw = np.random.standard_t(df=3, size=n)
            errs = np.sqrt(data_params["sigma2"])*(errs_raw - errs_raw.mean())/errs_raw.std()
        elif data_params["err_dist"].lower().startswith("norm"):
            errs = np.random.randn(n)*np.sqrt(data_params["sigma2"])
        Y = lin_term + errs
    elif data_type.startswith("logi"):
        probs = logit_fn(lin_term)
        Y = np.random.binomial(1, probs, size=len(probs))
    
    # To torch
    X = torch.Tensor(simu_curvs) # n x d x npts
    Z = torch.Tensor(simu_covs) # n x q
    Y = torch.Tensor(Y)
    
    all_data = edict()
    all_data.X = X
    all_data.Y = Y
    all_data.Z = Z
    return all_data



def gen_simu_sinica_dataset(n, d, q, types_, gt_alp, gt_beta, x, 
                            data_type="linear",
                            data_params={}, seed=0, verbose=2):
    """
    Generate simulated data for all parameters under sinica 

    Args:
        seed (int): Seed for random number generator.
        n (int): Number of samples.
        d (int): Number of dimensions.
        q (int): Number of covariates.
        types_ (list): List of types for generating covariates.
        gt_alp (list): List of ground truth alpha values.
        gt_beta (list): List of ground truth beta values.
        x (array): The points for X, npts vec
        fourier_basis(np.ndarray): The fourier basis used to generate X, npts0 x 50
        data_params (dict): Dict of other data params
            - sigma2 (float): Variance of the noise for linear model
            - err_dist (str): Distribution of the noise for linear model, norm or t
            - srho (float): The corr between Xs
        verbose(bool): 0-3

    Returns:
        all_data (dict): Dictionary containing the following simulated data:
            - X (torch.Tensor): Tensor of shape (n, d, npts) containing the simulated PSD.
            - Y (torch.Tensor): Tensor of shape (n,) containing the response variable.
            - Z (torch.Tensor): Tensor of shape (n, q) containing the covariates.
    """

    np.random.seed(seed)
    _set_verbose_level(verbose, logger)
    data_type = data_type.lower()
    if data_type.startswith("linear"):
        data_params_def = {
            "sigma2": 1, 
            "err_dist": "norm", 
            "srho": 0.3,
            "basis_type": "fourier", 
        }
    elif data_type.startswith("logi"):
        data_params_def = {
            "srho": 0.3,
            "basis_type": "fourier", 
        }
    else:
        raise ValueError(f"{data_type} is not supported now.")
    data_params = _update_params(data_params, data_params_def, logger)
    
    # simulated PSD
    assert len(types_) == q
    assert len(gt_alp) == q

    # this is for the integartion,not for the output X
    x0 = np.linspace(x[0], x[-1], gt_beta.shape[0])
   
    if data_params["basis_type"].lower().startswith("bsp"):
        basis_vs = obt_bsp_basis_Rfn_wrapper(x, N=10, bsp_ord=4)
        basis_vs0 = obt_bsp_basis_Rfn_wrapper(x0, N=10, bsp_ord=4)
        thetas = np.random.randn(n, d, 50)*5
        errsX = np.random.randn(n, d, len(x)) * 0.5

    elif data_params["basis_type"].lower().startswith("four"):
        basis_vs = fourier_basis_fn(x)
        basis_vs0 = fourier_basis_fn(x0)
        thetas = gen_sini_Xthetas(data_params.srho, n, d);
        errsX = np.random.randn(n, d, len(x)) * 0.0

    simu_curvs = thetas[:, :, :basis_vs.shape[1]] @ basis_vs.T; # n x d x npts
    simu_curvs = simu_curvs + errsX
    #simu_curvs = np.random.randn(n, d, npts) * 5
    simu_covs = gen_covs(n, types_)
    
    # linear term and Y
    simu_curvs0 = thetas[:, :, :basis_vs0.shape[1]] @ basis_vs0.T; # n x d x npts0
    simu_curvs0 = simu_curvs0 + errsX
    fs = np.sum(gt_beta.T* simu_curvs0[:, :, :], axis=1) # n x npts0
    int_part = integration_fn(fs.T, "sim").numpy()
    cov_part = simu_covs @ gt_alp
    
    # linear term
    lin_term = cov_part + int_part
    
    # Y 
    if data_type.startswith("linear"):
        if data_params["err_dist"].lower().startswith("t"):
            errs_raw = np.random.standard_t(df=3, size=n)
            errs = np.sqrt(data_params["sigma2"])*(errs_raw - errs_raw.mean())/errs_raw.std()
        elif data_params["err_dist"].lower().startswith("norm"):
            errs = np.random.randn(n)*np.sqrt(data_params["sigma2"])
        Y = lin_term + errs
    elif data_type.startswith("logi"):
        probs = logit_fn(lin_term)
        Y = np.random.binomial(1, probs, size=len(probs))
    
    # To torch
    X = torch.Tensor(simu_curvs) # n x d x npts
    Z = torch.Tensor(simu_covs) # n x q
    Y = torch.Tensor(Y)
    
    all_data = edict()
    all_data.X = X
    all_data.Y = Y
    all_data.Z = Z
    return all_data

def gen_simu_meg_dataset(n, q, types_, gt_alp, gt_beta, npts, 
                            base_data, 
                            data_type="linear",
                            data_params={}, seed=0, verbose=2):
    """
    Generate simulated data for all parameters under meg data
    d is 68

    Args:
        seed (int): Seed for random number generator.
        n (int): Number of samples.
        q (int): Number of covariates.
        types_ (list): List of types for generating covariates.
        gt_alp (list): List of ground truth alpha values.
        gt_beta (list): List of ground truth beta values.
        npts (int): The number of points
        data_params (dict): Dict of other data params
            - sigma2 (float): Variance of the noise for linear model
            - err_dist (str): Distribution of the noise for linear model, norm or t
        verbose(bool): 0-3

    Returns:
        all_data (dict): Dictionary containing the following simulated data:
            - X (torch.Tensor): Tensor of shape (n, d, npts) containing the simulated PSD.
            - Y (torch.Tensor): Tensor of shape (n,) containing the response variable.
            - Z (torch.Tensor): Tensor of shape (n, q) containing the covariates.
    """

    np.random.seed(seed)
    _set_verbose_level(verbose, logger)
    data_type = data_type.lower()
    if data_type.startswith("linear"):
        data_params_def = {
            "sigma2": 1, 
            "err_dist": "norm", 
        }
    elif data_type.startswith("logi"):
        data_params_def = {
        }
    else:
        raise ValueError(f"{data_type} is not supported now.")
    data_params = _update_params(data_params, data_params_def, logger)
    
    # simulated PSD
    assert len(types_) == q
    assert len(gt_alp) == q
   
    simu_curvs = get_meg_curvs(n, npts, base_data, move_step=20)
    #simu_curvs = np.random.randn(n, d, npts) * 5
    simu_covs = gen_covs(n, types_)
    
    # linear term and Y
    int_part = np.sum(gt_beta.T* simu_curvs[:, :, :], axis=1).mean(axis=1)
    cov_part = simu_covs @ gt_alp
    
    # linear term
    lin_term = cov_part + int_part
    
    # Y 
    if data_type.startswith("linear"):
        if data_params["err_dist"].lower().startswith("t"):
            errs_raw = np.random.standard_t(df=3, size=n)
            errs = np.sqrt(data_params["sigma2"])*(errs_raw - errs_raw.mean())/errs_raw.std()
        elif data_params["err_dist"].lower().startswith("norm"):
            errs = np.random.randn(n)*np.sqrt(data_params["sigma2"])
        Y = lin_term + errs
    elif data_type.startswith("logi"):
        probs = logit_fn(lin_term)
        Y = np.random.binomial(1, probs, size=len(probs))
    
    # To torch
    X = torch.Tensor(simu_curvs) # n x d x npts
    Z = torch.Tensor(simu_covs) # n x q
    Y = torch.Tensor(Y)
    
    all_data = edict()
    all_data.X = X
    all_data.Y = Y
    all_data.Z = Z
    return all_data

def get_meg_curvs(n, npts, base_data, move_step=20):
    """Get 68 x npts MEG based on basedata
    args:
        n: Num of data you want to get
        npts: The length of the seq
        base_data: The base data to generate MEG, num_sub x 68 x total_seq
    return:
        curvs: n x 68 x npts
    """
    num_subs, d, total_seq = base_data.shape
    init_pts = np.arange(0, total_seq-npts, move_step)
    sel_sub_idx = np.random.choice(num_subs, size=n);
    sel_init_idx = np.random.choice(init_pts, size=n);
    
    curvs = []
    for sub_ix, init_ix in zip(sel_sub_idx, sel_init_idx):
        curv = base_data[sub_ix, :, init_ix:(init_ix+npts)]
        #curv = 10*np.random.randn(68, npts)
        curv = (curv - curv.mean(axis=1, keepdims=1))/curv.std(axis=1, keepdims=1)*5 + 1*np.random.randn(68, npts)
        curvs.append(curv)
    curvs = np.array(curvs);
    return curvs
