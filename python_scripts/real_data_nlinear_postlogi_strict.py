#!/usr/bin/env python
# coding: utf-8

# This file contains python code to check the hypothesis testing

# In[24]:


RUN_PYTHON_SCRIPT = False
OUTLIER_IDXS = dict(AD=[49], ctrl=[14, 19, 30, 38])
SAVED_FOLDER = "real_data_nlinear_nostd"
DATA = ["AD88_matlab_1-45.pkl", "Ctrl92_matlab_1-45.pkl"]
num_cv_fold_out = 2


# In[25]:


import sys
sys.path.append("../mypkg")


# In[26]:


import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from numbers import Number
import itertools

from easydict import EasyDict as edict
from tqdm import trange, tqdm
from scipy.io import loadmat
from pprint import pprint
from IPython.display import display
from joblib import Parallel, delayed


from constants import DATA_ROOT, RES_ROOT, FIG_ROOT, MIDRES_ROOT
from hdf_utils.data_gen import gen_simu_sinica_dataset
from hdf_utils.SIS import SIS_GLIM
from utils.matrix import col_vec_fn, col_vec2mat_fn, conju_grad, svd_inverse, cholesky_inv
from utils.functions import logit_fn
from utils.misc import save_pkl, load_pkl
from splines import obt_bsp_obasis_Rfn, obt_bsp_basis_Rfn_wrapper
from projection import euclidean_proj_l1ball
from optimization.opt import HDHTOpt
from hdf_utils.fns_sinica import  fourier_basis_fn

from joblib import Parallel, delayed


import argparse
parser = argparse.ArgumentParser(description='run')
parser.add_argument('--out_ix', type=int, help='the cv index, from 0') 
args = parser.parse_args()


torch.set_default_tensor_type(torch.DoubleTensor)
def_dtype = torch.get_default_dtype()




# # Load  data and prepare

# In[31]:


data_root = DATA_ROOT/"AD_vs_Ctrl_PSD/";
AD_PSD = load_pkl(data_root/DATA[0]);
ctrl_PSD = load_pkl(data_root/DATA[1]);
df0= pd.read_csv(data_root/"AllDataBaselineOrdered_r_ncpt.csv");
df1= pd.read_csv(data_root/"AllDataBaselineOrdered_r_ncpt_more.csv");
df1 = df1.set_index("RID")
df0 = df0.set_index("RID");
df1 = df1.reindex(df0.index)
baseline = df1
baseline["Gender_binary"] = baseline["Gender"].apply(lambda x: 0 if x=="female" else 1);
baseline["Grp_binary"] = baseline["Grp"].apply(lambda x: 1 if x=="AD" else 0);


# In[32]:


# The outlier idxs to rm
outlier_idxs = np.concatenate([OUTLIER_IDXS["AD"], len(AD_PSD.PSDs)+np.array(OUTLIER_IDXS["ctrl"])])
outlier_idxs = outlier_idxs.astype(int)

# make PSD in dB and std 
raw_X = np.concatenate([AD_PSD.PSDs, ctrl_PSD.PSDs]); #n x d x npts
X_dB = 10*np.log10(raw_X);
outlier_idxs2 = np.where(X_dB.mean(axis=(1, 2))<0)
X = X_dB

Y = np.array(baseline["MMSE"])[:X.shape[0]];
# if logi
Yb = np.array(baseline["Grp_binary"])[:X.shape[0]];

#sel_cov = ["Education"]
sel_cov = ["Gender_binary", "MEG_Age","Education"]
Z_raw = np.array(baseline[sel_cov])[:X.shape[0]];

grp_idxs = np.array(baseline["Grp"])[:X.shape[0]];


outlier_idxs = np.sort(np.union1d(outlier_idxs, outlier_idxs2))


# remove outliers
X = np.delete(X, outlier_idxs, axis=0)
Y = np.delete(Y, outlier_idxs, axis=0)
Yb = np.delete(Yb, outlier_idxs, axis=0)
Z_raw = np.delete(Z_raw, outlier_idxs, axis=0)
grp_idxs = np.delete(grp_idxs, outlier_idxs, axis=0)


#remove nan
keep_idx = ~np.bitwise_or(np.isnan(Y), np.isnan(Z_raw.sum(axis=1)));
X = X[keep_idx];
Y = Y[keep_idx];
Yb = Yb[keep_idx]
Z_raw = Z_raw[keep_idx]
grp_idxs = grp_idxs[keep_idx]

Z = np.concatenate([np.ones((Z_raw.shape[0], 1)), Z_raw], axis=1); # add intercept


freqs = AD_PSD.freqs;
# only take PSD between [2, 35] freqs of interest
X = X[:, :, np.bitwise_and(freqs>=2, freqs<=35)]
X = X/X.mean()


print(X.shape, Y.shape, Z.shape)

all_data = edict()
if SAVED_FOLDER.endswith("X1err"):
    print("add noise to PSD")
    all_data.X = torch.tensor(X+np.random.randn(*X.shape)*0.1)
else:
    all_data.X = torch.tensor(X)
all_data.Y = torch.tensor(Yb)
all_data.Z = torch.tensor(Z)


# In[33]:


# atlas
rois = np.loadtxt(DATA_ROOT/"dk68_utils/ROI_order_DK68.txt", dtype=str);


# In[ ]:





# # Param and fns

# ## Params

# In[101]:


from easydict import EasyDict as edict
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn
from copy import deepcopy
from scenarios.base_params import get_base_params

base_params = get_base_params("logi") 
base_params.data_params = edict()
base_params.data_params.n = all_data.X.shape[0]
base_params.data_params.npts = all_data.X.shape[-1]
base_params.data_params.freqs = AD_PSD.freqs

base_params.can_Ns = [4, 6, 8, 10, 12]
base_params.SIS_params = edict({"SIS_pen": 0.02, "SIS_basis_N":8, "SIS_ws":"simpson"})
base_params.opt_params.beta = 1 
base_params.can_lams = [0.001, 0.005, 0.01, 0.02, 0.03,  0.04, 0.05, 0.06, 0.07, 0.1]
base_params.is_shuffle_cv = True
base_params.num_cv_fold = 10 


setting = edict(deepcopy(base_params))
add_params = edict({})
add_params.setting = "real_data_linear_postlogi"
add_params.SIS_ratio = 1
setting.update(add_params)


# In[35]:


save_dir = MIDRES_ROOT/SAVED_FOLDER
if not save_dir.exists():
    save_dir.mkdir()

seed = 0
np.random.seed(seed)
full_idx = load_pkl((RES_ROOT/SAVED_FOLDER)/f"shuffled_seed{seed}_full_idx_postlogi.pkl")

# In[37]:


def _run_main_fn(out_ix, sig_roi_idxs, lam, N, setting,  prefix, is_save=False, is_cv=False, verbose=2):
    torch.set_default_dtype(torch.double)
        
    Z = all_data.Z.clone()
    X = all_data.X[:, sig_roi_idxs].clone()
    Y = all_data.Y.float().clone()
    n = len(Y)
    n_perfold = int(n/num_cv_fold_out)
    test_idx = full_idx[(out_ix*n_perfold):(out_ix*n_perfold+n_perfold)]
    if out_ix == num_cv_fold_out-1:
        test_idx = full_idx[(out_ix*n_perfold):]
    train_idx = np.setdiff1d(full_idx, test_idx)
        
    Y = Y[train_idx]
    X = X[train_idx]
    Z = Z[train_idx]
    
    
    _setting = edict(setting.copy())
    _setting.lam = lam
    _setting.N = N
    _setting.data_params.d = X.shape[1]
    _setting.sel_idx = np.arange(_setting.data_params.d)
    
    
    f_name = f"{prefix}_cv{out_ix+1}-{num_cv_fold_out}_postlogi-lam_{lam*1000:.0f}-N_{N:.0f}_fit.pkl"
    
    
    if not (save_dir/f_name).exists():
        hdf_fit = HDHTOpt(lam=_setting.lam, 
                         sel_idx=_setting.sel_idx, 
                         model_type=_setting.model_type,
                         verbose=verbose, 
                         SIS_ratio=_setting.SIS_ratio, 
                         N=_setting.N,
                         is_std_data=True, 
                         cov_types=None, 
                         inits=None,
                         model_params = _setting.model_params, 
                         SIS_params = _setting.SIS_params, 
                         opt_params = _setting.opt_params,
                         bsp_params = _setting.bsp_params, 
                         pen_params = _setting.pen_params
               );
        hdf_fit.add_data(X, Y, Z)
        opt_res = hdf_fit.fit()
        
        if is_cv:
            hdf_fit.get_cv_est(_setting.num_cv_fold, _setting.is_shuffle_cv)
        if is_save:
            hdf_fit.save(save_dir/f_name, is_compact=False, is_force=True)
    else:
        hdf_fit = load_pkl(save_dir/f_name, verbose>=2);
        
    return None

sig_roi_idxs = load_pkl((RES_ROOT/SAVED_FOLDER)/f"sig_roi_idxs.pkl");

setting.opt_params.max_iter = 5000
all_coms = itertools.product(setting.can_lams, setting.can_Ns)
with Parallel(n_jobs=35) as parallel:
    ress = parallel(delayed(_run_main_fn)(args.out_ix, 
                                          sig_roi_idxs, lam=lam, N=N, setting=setting, 
    prefix="hdf", is_save=True, is_cv=True, verbose=1) 
    for  lam, N in tqdm(all_coms, total=len(setting.can_lams)*len(setting.can_Ns)))

