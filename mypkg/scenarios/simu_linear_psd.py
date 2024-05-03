import numpy as np
from easydict import EasyDict as edict
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn
from copy import deepcopy
from .base_params import get_base_params
from constants import RES_ROOT

base_params = get_base_params("linear") 
base_params.data_gen_params = edict()
base_params.data_gen_params.d = 200 # num of ROIs
base_params.data_gen_params.q = 1 # num of other covariates
base_params.data_gen_params.npts = 100 # num of pts to evaluate X(s)
base_params.data_gen_params.freqs = np.linspace(2, 45, base_params.data_gen_params.npts) # freqs
base_params.data_gen_params.types_ = ["int"]
base_params.data_gen_params.is_std = False
base_params.data_gen_params.gt_alp = np.array([0]) # we will determine intercept later
base_params.data_gen_params.data_type = base_params.model_type
base_params.data_gen_params.data_params={"sigma2":1}
base_params.can_Ns = [4, 6, 8, 10, 12]
base_params.can_lams = [0.5, 0.60,  0.70, 0.80, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
#base_params.can_lams = [0.001, 0.5, 0.60,  0.70, 0.80, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0]
def _get_gt_beta(cs, d, fct=1):
    x = np.linspace(0, 1, 100)
    fourier_basis = fourier_basis_fn(x)
    fourier_basis_coefs = ([cs[0]*coef_fn(0.2), cs[1]*coef_fn(0.2), cs[2]*coef_fn(0.2)] + 
                                 [np.zeros(50)] * (d-3-1) +
                                 [coef_fn(0.2)]
                                 )
    fourier_basis_coefs = np.array(fourier_basis_coefs).T 
    gt_beta = fourier_basis @ fourier_basis_coefs * fct
    return gt_beta



##---settings------------------------------
#========================================================================================================
settingcmpn1 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 100 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.data_params["psd_noise_sd"] = 10
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs,
                                                add_params.data_gen_params.d,
                                                fct=1) 
add_params.setting = "cmpn1"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingcmpn1.update(add_params)

##---- 
settingcmpn2 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 100 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, 0]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.data_params["psd_noise_sd"] = 10
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs,
                                                add_params.data_gen_params.d,
                                                fct=1) 
add_params.setting = "cmpn2"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingcmpn2.update(add_params)

## -------
settingcmpn3 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 100 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, c]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.data_params["psd_noise_sd"] = 10
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs,
                                                add_params.data_gen_params.d,
                                                fct=1) 
add_params.setting = "cmpn3"
add_params.sel_idx =  np.arange(3, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingcmpn3.update(add_params)


#========================================================================================================
### t(3) error
settingcmpn1b = edict(deepcopy(settingcmpn1))
settingcmpn1b.setting = "cmpn1b"
settingcmpn1b.data_gen_params.data_params["err_dist"] = "t"

settingcmpn2b = edict(deepcopy(settingcmpn2))
settingcmpn2b.setting = "cmpn2b"
settingcmpn2b.data_gen_params.data_params["err_dist"] = "t"

settingcmpn3b = edict(deepcopy(settingcmpn3))
settingcmpn3b.setting = "cmpn3b"
settingcmpn3b.data_gen_params.data_params["err_dist"] = "t"


#========================================================================================================

settings = edict()
settings.cmpn1 = settingcmpn1
settings.cmpn2 = settingcmpn2
settings.cmpn3 = settingcmpn3
settings.cmpn1b = settingcmpn1b
settings.cmpn2b = settingcmpn2b
settings.cmpn3b = settingcmpn3b
