#!/usr/bin/env python
# coding: utf-8

# This file contains python code to check the hypothesis testing

# In[1]:


RUN_PYTHON_SCRIPT = False
#OUTLIER_IDXS = dict(AD=[], ctrl=[])
OUTLIER_IDXS = dict(AD=[49], ctrl=[14, 19, 30, 38])
SAVED_FOLDER = "real_data_nlinear_nostd"
#SAVED_FOLDER = "real_data_nlinear_nostd_X1err"
DATA = ["AD88_matlab_1-45.pkl", "Ctrl92_matlab_1-45.pkl"]


# In[2]:


import sys
sys.path.append("../mypkg")


# In[3]:


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
from utils.misc import save_pkl, load_pkl
from optimization.opt import HDFOpt

from joblib import Parallel, delayed

import argparse
parser = argparse.ArgumentParser(description='run')
parser.add_argument('--N', type=int, help='Bspline basis') 
args = parser.parse_args()

# In[6]:




# In[7]:


torch.set_default_tensor_type(torch.DoubleTensor)
def_dtype = torch.get_default_dtype()


# In[ ]:





# # Load  data and prepare

# In[8]:


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


# In[23]:


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
#Yb = np.array(baseline["Grp_binary"])[:X.shape[0]];

#sel_cov = [ "MEG_Age","Gender_binary"]
sel_cov = ["Gender_binary", "MEG_Age","Education"]
Z_raw = np.array(baseline[sel_cov])[:X.shape[0]];

grp_idxs = np.array(baseline["Grp"])[:X.shape[0]];

outlier_idxs = np.sort(np.union1d(outlier_idxs, outlier_idxs2))
# remove outliers
X = np.delete(X, outlier_idxs, axis=0)
Y = np.delete(Y, outlier_idxs, axis=0)
Z_raw = np.delete(Z_raw, outlier_idxs, axis=0)
grp_idxs = np.delete(grp_idxs, outlier_idxs, axis=0)


#remove nan
keep_idx = ~np.bitwise_or(np.isnan(Y), np.isnan(Z_raw.sum(axis=1)));
X = X[keep_idx];
Y = Y[keep_idx]
Z_raw = Z_raw[keep_idx]
grp_idxs = grp_idxs[keep_idx]

Z = np.concatenate([np.ones((Z_raw.shape[0], 1)), Z_raw], axis=1); # add intercept



freqs = AD_PSD.freqs;
# only take PSD between [2, 35] freqs of interest
X = X[:, :, np.bitwise_and(freqs>=2, freqs<=35)]
X = X/X.mean()

print(X.shape, Y.shape, Z.shape)

all_data = edict()
all_data.X = torch.tensor(X)
#all_data.X = torch.tensor(X+np.random.randn(*X.shape)*0.1)
all_data.Y = torch.tensor(Y)
all_data.Z = torch.tensor(Z)


from easydict import EasyDict as edict
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn
from copy import deepcopy
from scenarios.base_params import get_base_params

base_params = get_base_params("linear") 
base_params.data_params = edict()
base_params.data_params.d = all_data.X.shape[1]
base_params.data_params.n = all_data.X.shape[0]
base_params.data_params.npts = all_data.X.shape[-1]
base_params.data_params.freqs = AD_PSD.freqs

base_params.can_Ns = [4, 6, 8, 10, 12]
base_params.SIS_params = edict({"SIS_pen": 0.02, "SIS_basis_N":8, "SIS_ws":"simpson"})
base_params.opt_params.beta = 1
base_params.bsp_params.is_orth_basis = True
base_params.can_lams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
#base_params.can_lams = [0.01, 0.1, 0.4, 0.8, 1.2, 1.6, 3.2]


setting = edict(deepcopy(base_params))
add_params = edict({})
add_params.setting = "real_data_linear"
add_params.SIS_ratio = 1.0
setting.update(add_params)


# In[26]:


save_dir = RES_ROOT/SAVED_FOLDER
if not save_dir.exists():
    save_dir.mkdir(exist_ok=True)


def _run_main_fn(roi_idx, lam, N, setting, is_save=False, is_cv=False, verbose=2):
    torch.set_default_dtype(torch.double)
        
    _setting = edict(setting.copy())
    _setting.lam = lam
    _setting.N = N
    _setting.sel_idx = np.delete(np.arange(setting.data_params.d), [roi_idx])
    
    
    f_name = f"roi_{roi_idx:.0f}-lam_{lam*1000:.0f}-N_{N:.0f}_fit.pkl"
    
    
    if not (save_dir/f_name).exists():
        hdf_fit = HDFOpt(lam=_setting.lam, 
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
        hdf_fit.add_data(all_data.X, all_data.Y, all_data.Z)
        opt_res = hdf_fit.fit()
        
        if is_cv:
            hdf_fit.get_cv_est(_setting.num_cv_fold)
        if is_save:
            hdf_fit.save(save_dir/f_name, is_compact=False, is_force=True)
    else:
        hdf_fit = load_pkl(save_dir/f_name, verbose>=2);
        
    return None




setting.opt_params.max_iter = 5000
all_coms = itertools.product(range(0, setting.data_params.d), setting.can_lams)
with Parallel(n_jobs=35) as parallel:
    ress = parallel(delayed(_run_main_fn)(roi_idx=roi_idx, lam=lam, N=args.N, setting=setting, is_save=True, is_cv=True, verbose=1) for roi_idx, lam
                    in tqdm(all_coms, total=len(setting.can_lams)*setting.data_params.d))


