import numpy as np
from easydict import EasyDict as edict
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn
from copy import deepcopy
from .base_params import get_base_params
from constants import RES_ROOT

base_params = get_base_params("linear") 
base_params.data_gen_params = edict()
base_params.data_gen_params.d = 68 # num of ROIs
base_params.data_gen_params.q = 3 # num of other covariates
base_params.data_gen_params.npts = 100 # num of pts to evaluate X(s)
base_params.data_gen_params.freqs = np.linspace(2, 45, base_params.data_gen_params.npts) # freqs
base_params.data_gen_params.types_ = ["int", "c", 2]
base_params.data_gen_params.is_std = False
base_params.data_gen_params.gt_alp = np.array([5, -1, 2]) # we will determine intercept later
base_params.data_gen_params.data_params={"psd_noise_sd":10, "sigma2":1}
base_params.data_gen_params.data_type = base_params.model_type
base_params.can_Ns = [4, 6, 8, 10, 12]
def _get_gt_beta(cs, d, npts, fct=2):
    x = np.linspace(0, 1, npts)
    fourier_basis = fourier_basis_fn(x)
    fourier_basis_coefs = ([cs[0]*coef_fn(0.2), cs[1]*coef_fn(0.2), cs[2]*coef_fn(0.2)] + 
                                 [np.zeros(50)] * (d-3-1) +
                                 [coef_fn(0.2)]
                                 )
    fourier_basis_coefs = np.array(fourier_basis_coefs).T 
    gt_beta = fourier_basis @ fourier_basis_coefs * fct
    return gt_beta



#---settings------------------------------
#========================================================================================================
settingn1 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "n1"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 2, 8]
#add_params.can_lams = [0.001, 0.2, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1, 1.1, 1.2, 2, 8]
add_params.can_Ns = [4, 6, 8, 10, 12]
add_params.SIS_ratio = 0.2
settingn1.update(add_params)

## -------

settingn1b = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.data_params["err_dist"] = "t"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "n1b"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.6, 0.7, 0.8,  0.9, 1, 1.1, 1.2, 2, 8]
#add_params.can_lams = [0.001, 0.1, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 2, 8]
add_params.can_Ns = [4, 6, 8, 10, 12]
add_params.SIS_ratio = 0.2
settingn1b.update(add_params)


#========================================================================================================

settingn2 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "n2"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.5, 0.6, 0.7, 0.8,  0.9,  1, 1.1, 1.2, 2, 8] 
#add_params.can_lams = [0.001, 0.1, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 2, 8] 
add_params.can_Ns = [4, 6, 8, 10, 12]
add_params.SIS_ratio = 0.2
settingn2.update(add_params)


## -------

settingn2b = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.data_params["err_dist"] = "t"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "n2b"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.6, 0.7, 0.8,  0.9, 1, 1.1, 1.2, 2, 8]
#add_params.can_lams = [0.001, 0.1, 0.6, 0.7, 0.8,  0.9, 0.95, 1, 1.05, 1.1, 1.2, 2, 8]
add_params.can_Ns = [4, 6, 8, 10, 12]
add_params.SIS_ratio = 0.2
settingn2b.update(add_params)

#========================================================================================================

settingn3 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, c]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "n3"
add_params.sel_idx =  np.arange(3, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.4, 0.5, 0.6,  0.7, 0.8,  0.9, 1, 2, 8]
#add_params.can_lams = [0.001, 0.1, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1, 2, 8]
add_params.can_Ns = [4, 6, 8, 10, 12]
add_params.SIS_ratio = 0.2
settingn3.update(add_params)

## -------

settingn3b = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, c]
add_params.data_gen_params.data_params["err_dist"] = "t"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)
add_params.setting = "n3b"
add_params.sel_idx =  np.arange(3, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.4, 0.5, 0.6, 0.7,  0.8, 0.9, 1, 2, 8]
#add_params.can_lams = [0.001, 0.1, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1, 2, 8]
add_params.can_Ns = [4, 6, 8, 10, 12]
add_params.SIS_ratio = 0.2
settingn3b.update(add_params)



#========================================================================================================

#### SIS_ratio = 1
settingn1c = edict(deepcopy(settingn1))
settingn1c.setting = "n1c"
settingn1c.SIS_ratio = 1
settingn1d = edict(deepcopy(settingn1b))
settingn1d.setting = "n1d"
settingn1d.SIS_ratio = 1

settingn2c = edict(deepcopy(settingn2))
settingn2c.setting = "n2c"
settingn2c.SIS_ratio = 1
settingn2d = edict(deepcopy(settingn2b))
settingn2d.setting = "n2d"
settingn2d.SIS_ratio = 1

settingn3c = edict(deepcopy(settingn3))
settingn3c.setting = "n3c"
settingn3c.SIS_ratio = 1
settingn3d = edict(deepcopy(settingn3b))
settingn3d.setting = "n3d"
settingn3d.SIS_ratio = 1

#### n = 500
settingn1a = edict(deepcopy(settingn1))
settingn1a.setting = "n1a"
settingn1a.can_lams = [0.001, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 8]
settingn1a.data_gen_params.n = 500
settingn1e = edict(deepcopy(settingn1b))
settingn1e.setting = "n1e"
settingn1e.can_lams = [0.001, 0.1, 0.4, 0.5, 0.6, 0.7,  0.8,  0.9,  1, 1.1, 1.2, 2, 8]
#settingn1e.can_lams = [0.001, 0.1, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 2, 8]
settingn1e.data_gen_params.n = 500

settingn2a = edict(deepcopy(settingn2))
settingn2a.setting = "n2a"
settingn2a.data_gen_params.n = 500
settingn2e = edict(deepcopy(settingn2b))
settingn2e.setting = "n2e"
settingn2e.can_lams = [0.001, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8,  0.9, 1,  1.1, 1.2, 2, 8]
#settingn2e.can_lams = [0.001, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8,  0.9, 0.95, 1, 1.05, 1.1, 1.2, 2, 8]
settingn2e.data_gen_params.n = 500

settingn3a = edict(deepcopy(settingn3))
settingn3a.setting = "n3a"
settingn3a.data_gen_params.n = 500
settingn3e = edict(deepcopy(settingn3b))
settingn3e.setting = "n3e"
settingn3e.data_gen_params.n = 500


#### n = 400
settingn1f = edict(deepcopy(settingn1))
settingn1f.setting = "n1f"
settingn1f.data_gen_params.n = 400
settingn1g = edict(deepcopy(settingn1b))
settingn1g.setting = "n1g"
settingn1g.data_gen_params.n = 400

settingn2f = edict(deepcopy(settingn2))
settingn2f.setting = "n2f"
settingn2f.data_gen_params.n = 400
settingn2g = edict(deepcopy(settingn2b))
settingn2g.setting = "n2g"
settingn2g.data_gen_params.n = 400

settingn3f = edict(deepcopy(settingn3))
settingn3f.setting = "n3f"
settingn3f.data_gen_params.n = 400
settingn3g = edict(deepcopy(settingn3b))
settingn3g.setting = "n3g"
settingn3g.data_gen_params.n = 400



#========================================================================================================
settings = edict()
settings.n1 = settingn1
settings.n1a = settingn1a
settings.n1b = settingn1b
settings.n1c = settingn1c
settings.n1d = settingn1d
settings.n1e = settingn1e
settings.n1f = settingn1f
settings.n1g = settingn1g

settings.n2 = settingn2
settings.n2a = settingn2a
settings.n2b = settingn2b
settings.n2c = settingn2c
settings.n2e = settingn2e
settings.n2f = settingn2f
settings.n2g = settingn2g

settings.n3 = settingn3
settings.n3a = settingn3a
settings.n3b = settingn3b
settings.n3c = settingn3c
settings.n3e = settingn3e
settings.n3f = settingn3f
settings.n3g = settingn3g
