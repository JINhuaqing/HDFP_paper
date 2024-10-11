#!/usr/bin/env python
# coding: utf-8

# This file contains python code to mimic real setting
# 
# It is under the logi setting
# 
# Now, I use the same beta from the paper but the PSD as X

# In[1]:


import sys
sys.path.append("../mypkg")

import numpy as np
import torch
import itertools
from easydict import EasyDict as edict
from tqdm import tqdm
from pprint import pprint
from joblib import Parallel, delayed


from constants import DATA_ROOT, RES_ROOT, FIG_ROOT, MIDRES_ROOT
from hdf_utils.data_gen import gen_simu_psd_dataset
from utils.misc import save_pkl, load_pkl, bcross_entropy_loss
from optimization.opt import HDHTOpt
from scenarios.real_simu_logi import settings

import argparse
parser = argparse.ArgumentParser(description='run')
parser.add_argument('-c', '--cs', type=float, help='cs value') 
parser.add_argument('-s', '--setting', type=str, help='Setting') 
args = parser.parse_args()

torch.set_default_dtype(torch.double)


np.random.seed(0)
c = args.cs

setting = settings[args.setting]
data_gen_params = setting.data_gen_params
data_gen_params.cs = data_gen_params.cs_fn(c)
data_gen_params.gt_beta = data_gen_params.beta_fn(data_gen_params.cs)
setting.model_params = {"ws":"naive"}

num_rep = 1000
n_jobs = 30
Cmat = np.eye(data_gen_params.d - len(setting.sel_idx))
save_dir = RES_ROOT/f"simu_logi_setting{setting.setting}_{c*1000:.0f}"
if not save_dir.exists():
    save_dir.mkdir()


# In[13]:


def _get_logi_int(data_gen_params, n_jobs=30, num_rep=100):
    ress = []
    for inte in tqdm(data_gen_params.intercept_cans):
        gt_alp = np.concatenate([[inte], data_gen_params.gt_alp0])
        def _tmp_fn(seed, data_gen_params=data_gen_params):
            data = gen_simu_psd_dataset(n=data_gen_params.n, 
                                        d=data_gen_params.d, 
                                        q=data_gen_params.q, 
                                        types_=data_gen_params.types_, 
                                        gt_alp=gt_alp, 
                                        gt_beta=data_gen_params.gt_beta, 
                                        freqs=data_gen_params.freqs, 
                                        data_type=data_gen_params.data_type, 
                                        data_params=data_gen_params.data_params, 
                                        seed=seed, 
                                        is_std=data_gen_params.is_std, 
                                        verbose=1, 
                                        is_gen=False);
            return data.Y.numpy()
        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(_tmp_fn)(seed) for seed in range(num_rep))
        ress.append(np.array(res))


    # get the intercept
    Yms = np.array([res.mean() for res in ress])
    intercept = data_gen_params.intercept_cans[np.argmin(np.abs(Yms-0.5))]
    print(f"The mean of Y is {Yms[np.argmin(np.abs(Yms-0.5))]:.3f} under intercept {intercept:.3f}.")
    gt_alp = np.concatenate([[intercept], data_gen_params.gt_alp0])
    return gt_alp


# In[14]:


data_gen_params.gt_alp = _get_logi_int(data_gen_params, n_jobs=n_jobs);

pprint(setting)
print(f"Save at {save_dir}")

# # Simu

def _main_run_fn(seed, lam, N, setting, is_save=False, is_cv=False, verbose=2):
    """Now (on Aug 25, 2023), if we keep seed the same, the cur_data is the same. 
       If you want to make any changes, make sure this. 
    """
    torch.set_default_dtype(torch.double)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
    _setting = edict(setting.copy())
    _setting.seed = seed
    _setting.lam = lam
    _setting.N = N
    
    data_gen_params = setting.data_gen_params
    
    f_name = f"seed_{seed:.0f}-lam_{lam*1000:.0f}-N_{N:.0f}_fit.pkl"
    
    
    if not (save_dir/f_name).exists():
        cur_data = gen_simu_psd_dataset(n=data_gen_params.n, 
                            d=data_gen_params.d, 
                            q=data_gen_params.q, 
                            types_=data_gen_params.types_, 
                            gt_alp=data_gen_params.gt_alp, 
                            gt_beta=data_gen_params.gt_beta, 
                            freqs=data_gen_params.freqs, 
                            data_type=data_gen_params.data_type, 
                            data_params=data_gen_params.data_params, 
                            seed=seed, 
                            is_std=data_gen_params.is_std, 
                            verbose=verbose, 
                            is_gen=False);
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
        hdf_fit.add_data(cur_data.X, cur_data.Y, cur_data.Z)
        opt_res = hdf_fit.fit()
        
        if is_cv:
            hdf_fit.get_cv_est(_setting.num_cv_fold)
        if is_save:
            hdf_fit.save(save_dir/f_name, is_compact=True, is_force=True)
    else:
        hdf_fit = load_pkl(save_dir/f_name, verbose>=2);
        
    return None




all_coms = itertools.product(range(0, num_rep), setting.can_lams, setting.can_Ns)
with Parallel(n_jobs=n_jobs) as parallel:
    ress = parallel(delayed(_main_run_fn)(seed, lam=lam, N=N, setting=setting, is_save=True, is_cv=True, verbose=1) 
                    for seed, lam, N 
                    in tqdm(all_coms, total=len(setting.can_Ns)*len(setting.can_lams)*num_rep))






def _get_valset_metric_fn(res):
    valsel_metrics = edict()
    valsel_metrics.entropy_loss = bcross_entropy_loss(res.cv_Y_est, res.Y.numpy());
    valsel_metrics.mse_loss = np.mean((res.cv_Y_est- res.Y.numpy())**2);
    valsel_metrics.mae_loss = np.mean(np.abs(res.cv_Y_est-res.Y.numpy()));
    valsel_metrics.cv_probs = res.cv_Y_est
    valsel_metrics.tY = res.Y.numpy()
    return valsel_metrics
def _run_fn_extract(seed, N, lam, c):
    f_name = f"seed_{seed:.0f}-lam_{lam*1000:.0f}-N_{N:.0f}_fit.pkl"
    res = load_pkl(save_dir/f_name, verbose=0)
    return (seed, N, lam), _get_valset_metric_fn(res)

all_coms = itertools.product(range(0, num_rep), setting.can_lams, setting.can_Ns)
with Parallel(n_jobs=n_jobs) as parallel:
    all_cv_errs_list = parallel(delayed(_run_fn_extract)(cur_seed, cur_N, cur_lam, c=c)  for cur_seed, cur_lam, cur_N in tqdm(all_coms, total=num_rep*len(setting.can_Ns)*len(setting.can_lams), desc=f"c: {c}"))
all_cv_errs = {res[0]:res[1] for res in all_cv_errs_list};
save_pkl(save_dir/f"all-valsel-metrics.pkl", all_cv_errs, is_force=1)
