#!/usr/bin/env python
# coding: utf-8

# generate data for sinica method simulation 
# 
# X is PSD

# In[1]:


import sys
sys.path.append("../mypkg")


# In[2]:


import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import trange, tqdm
from pprint import pprint
from scipy.io import savemat



from constants import DATA_ROOT, MIDRES_ROOT
from hdf_utils.data_gen import gen_simu_psd_dataset
from utils.misc import save_pkl, load_pkl
from scenarios.simu_linear_psd import settings

from joblib import Parallel, delayed


# In[9]:

import argparse
parser = argparse.ArgumentParser(description='run')
parser.add_argument('-c', '--cs', type=float, help='cs value') 
parser.add_argument('-s', '--setting', type=str, help='Setting') 
args = parser.parse_args()

torch.set_default_dtype(torch.double)


num_rep = 1000
c = args.cs

np.random.seed(0)

setting = settings[args.setting]
data_gen_params = setting.data_gen_params
data_gen_params.cs = data_gen_params.cs_fn(c)
data_gen_params.gt_beta = data_gen_params.beta_fn(data_gen_params.cs)

save_dir = MIDRES_ROOT/f"matlab_simu_data/simu_setting{setting.setting}"
if not save_dir.exists():
    save_dir.mkdir(exist_ok=True)


# In[ ]:


pprint(data_gen_params)
print(f"Save to {save_dir}")



# In[16]:


def _run_fn(seed, verbose=0):
    torch.set_default_dtype(torch.double)
    np.random.seed(seed)
    torch.manual_seed(seed)
    f_name = f'c1_{c*1000:.0f}_seed_{seed}.mat'
    
    
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
        X = cur_data.X
        Y = cur_data.Y
        X_centered = X - X.mean(axis=0, keepdims=True)
        Y_centered = Y - Y.mean(axis=0, keepdims=True)
        sinica_data = {'Y_centered':Y_centered.numpy(), 
                       'X_centered':X_centered.numpy()}
        savemat(save_dir/f_name, sinica_data)
    else:
        print(f"File {save_dir/f_name} exists!")
    return None


with Parallel(n_jobs=35) as parallel:
    ress = parallel(delayed(_run_fn)(seed) 
                    for seed
                    in tqdm(range(num_rep), total=num_rep))




