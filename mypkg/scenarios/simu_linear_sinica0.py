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
base_params.data_gen_params.types_ = ["int"]
base_params.data_gen_params.gt_alp = np.array([0]) # we will determine intercept later
base_params.data_gen_params.data_type = base_params.model_type
base_params.data_gen_params.data_params={"sigma2":1, "srho":0.3, "basis_type":"bsp"}
base_params.SIS_params = edict({"SIS_pen": 0.02, "SIS_basis_N":8, "SIS_ws":"simpson"})
base_params.opt_params.beta = 1 
base_params.can_Ns = [4, 6, 8, 10, 12]
base_params.can_lams = [0.5, 0.60,  0.70, 0.80, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
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
settingcmpn0s1 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 100 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs,
                                                add_params.data_gen_params.d,
                                                fct=1) 
add_params.setting = "cmpn0s1"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingcmpn0s1.update(add_params)

##---- 
settingcmpn0s2 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 100 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, 0]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs,
                                                add_params.data_gen_params.d,
                                                fct=1) 
add_params.setting = "cmpn0s2"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingcmpn0s2.update(add_params)

## -------
settingcmpn0s3 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 100 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, c]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs,
                                                add_params.data_gen_params.d,
                                                fct=1) 
add_params.setting = "cmpn0s3"
add_params.sel_idx =  np.arange(3, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingcmpn0s3.update(add_params)


#========================================================================================================
### t(3) error
settingcmpn0s1b = edict(deepcopy(settingcmpn0s1))
settingcmpn0s1b.setting = "cmpn0s1b"
settingcmpn0s1b.data_gen_params.data_params["err_dist"] = "t"

settingcmpn0s2b = edict(deepcopy(settingcmpn0s2))
settingcmpn0s2b.setting = "cmpn0s2b"
settingcmpn0s2b.data_gen_params.data_params["err_dist"] = "t"

settingcmpn0s3b = edict(deepcopy(settingcmpn0s3))
settingcmpn0s3b.setting = "cmpn0s3b"
settingcmpn0s3b.data_gen_params.data_params["err_dist"] = "t"


#========================================================================================================

settings = edict()
settings.cmpn0s1 = settingcmpn0s1
settings.cmpn0s2 = settingcmpn0s2
settings.cmpn0s3 = settingcmpn0s3
settings.cmpn0s1b = settingcmpn0s1b
settings.cmpn0s2b = settingcmpn0s2b
settings.cmpn0s3b = settingcmpn0s3b
