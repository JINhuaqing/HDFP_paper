import numpy as np
from easydict import EasyDict as edict
from copy import deepcopy


_base_params = edict()
_base_params.SIS_params = edict({})
# basis 
_base_params.bsp_params = edict({})
# CV
_base_params.num_cv_fold = 10
_base_params.pen_params= edict({})

_base_params.hypo_params = edict({})



def get_base_params(model_type):
    cur_base_params = deepcopy(_base_params)
    cur_base_params.model_type = model_type

    
    if cur_base_params.model_type.startswith("linear"):
        cur_base_params.model_params = edict({})
    elif cur_base_params.model_type.startswith("logi"):
        cur_base_params.model_params = edict({})
    
    
    if cur_base_params.model_type.startswith("linear"):
        cur_base_params.opt_params= edict({
                   'beta': 10,
                 })
    elif cur_base_params.model_type.startswith("logi"):
        cur_base_params.opt_params= edict({
                   'beta': 1.5,
                 })

    return cur_base_params