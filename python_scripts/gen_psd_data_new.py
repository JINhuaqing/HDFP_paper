#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("../mypkg")


# In[6]:


import numpy as np
from tqdm import trange
from hdf_utils.data_gen import gen_simu_psd_dataset
from scenarios.simu_linear_psd import settings
from joblib import Parallel, delayed


import argparse
parser = argparse.ArgumentParser(description='gen PSD')
parser.add_argument('--start', type=int, default=0, help='starting seed') 
parser.add_argument('--interval', type=int, default=250, help='interval') 
args = parser.parse_args()

parser.add_argument('--is_std', action='store_true', help='Std PSD across freq or not, if not, only center, no --is_std=False, --is_std=True') 
np.random.seed(0)
c = 0.0

setting = settings.cmpn1b
data_gen_params = setting.data_gen_params
data_gen_params.cs = data_gen_params.cs_fn(c)
data_gen_params.gt_beta = data_gen_params.beta_fn(data_gen_params.cs)



for seed in trange(args.start, args.start+args.interval):
    gen_simu_psd_dataset(n=data_gen_params.n, 
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
                            verbose=1, 
                            is_gen=True);
