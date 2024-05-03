import numpy as np
import pickle
from easydict import EasyDict as edict
import logging

def trunc_mean(values, alpha=0.05):
    """
    Calculate the truncated mean of a given array of values.

    Parameters:
        values (array-like): The input array of values.
        alpha (float, optional): The significance level used to determine the lower and upper limits for truncation. Default is 0.05.

    Returns:
        float: The truncated mean of the input values.

    """
    if alpha > 0:
        lowlmt, uplmt = np.quantile(values, [alpha/2, 1-alpha/2])
    else:
        lowlmt, uplmt = -np.inf, np.inf
    kpidx = np.bitwise_and(values>=lowlmt, values<=uplmt)
    return np.mean(values[kpidx])
def bcross_entropy_loss_truc(probs, y, alpha=0.05):
    """
    Calculates the binary cross-entropy loss between predicted probabilities and true labels.

    Args:
        probs (numpy.ndarray): Predicted probabilities.
        y (numpy.ndarray): True labels.

    Returns:
        float: Binary cross-entropy loss.
    """
    assert np.bitwise_or(y ==1, y==0).sum() == len(y), "True labels should be either 1 or 0!"
    eps = 1e-8
    probs[probs==0] = eps
    probs[probs==1] = 1-eps
    scores = -(np.log(probs)*y + np.log(1-probs)*(1-y))
    return trunc_mean(scores, alpha)

def _set_verbose_level(verbose, logger):
    if verbose == 0:
        verbose_lv = logging.ERROR
    elif verbose == 1:
        verbose_lv = logging.WARNING
    elif verbose == 2:
        verbose_lv = logging.INFO
    elif verbose == 3:
        verbose_lv = logging.DEBUG
    if len(logger.handlers)>0:
        logger.handlers[0].setLevel(verbose_lv)
    else:
        logger.setLevel(verbose_lv)

def _update_params(input_params, def_params, logger):
    for ky, v in input_params.items():
        if ky not in def_params.keys():
            logger.warning(f"Check your input, {ky} is not used.")
        else:
            if v is not None:
                def_params[ky] = v
    return edict(def_params)

def bcross_entropy_loss(probs, y, is_median=False):
    """
    Calculates the binary cross-entropy loss between predicted probabilities and true labels.

    Args:
        probs (numpy.ndarray): Predicted probabilities.
        y (numpy.ndarray): True labels.

    Returns:
        float: Binary cross-entropy loss.
    """
    assert np.bitwise_or(y ==1, y==0).sum() == len(y), "True labels should be either 1 or 0!"
    eps = 1e-8
    probs[probs==0] = eps
    probs[probs==1] = 1-eps
    if is_median:
        return -np.median(np.log(probs)*y + np.log(1-probs)*(1-y))
    else:
        return -np.mean(np.log(probs)*y + np.log(1-probs)*(1-y))

def get_local_min_idxs(x):
    """ This fn is to get the local minimals. 
        args:
            x: a vec;
        return:
            local minimal idxs in x.
    """
    x_diff = np.diff(x)
    idc_vec = np.diff(np.sign(x_diff))
    
    # if v=2, must be a local minimal
    true_2s = np.where(idc_vec==2)[0]
    
    idxs1 = np.where(idc_vec==1)[0]
    true_1s = []
    flag = 0
    while flag < len(idxs1):
        if flag == len(idxs1)-1:
            break
        if np.min(idc_vec[idxs1[flag]:idxs1[flag+1]]) >=0:
            true_1s.append(idxs1[flag])
            true_1s.append(idxs1[flag+1])
            flag += 2
        else:
            flag += 1
    true_1s = np.array(true_1s)
    
    true_0s = [[]]
    for ix in range(0, len(true_1s), 2):
        true_0s.append(np.arange(true_1s[ix]+1, true_1s[ix+1]))
        
    true_0s = np.concatenate(true_0s)
    
    all_loc_min_idxs = np.sort(np.concatenate([true_0s, true_1s, true_2s])) + 1
    return all_loc_min_idxs.astype(int)
        

def load_pkl_folder2dict(folder, excluding=[], including=["*"], verbose=True):
    """The function is to load pkl file in folder as an edict
        args:
            folder: the target folder
            excluding: The files excluded from loading
            including: The files included for loading
            Note that excluding override including
    """
    if not isinstance(including, list):
        including = [including]
    if not isinstance(excluding, list):
        excluding = [excluding]
        
    if len(including) == 0:
        inc_fs = []
    else:
        inc_fs = list(set(np.concatenate([list(folder.glob(nam+".pkl")) for nam in including])))
    if len(excluding) == 0:
        exc_fs = []
    else:
        exc_fs = list(set(np.concatenate([list(folder.glob(nam+".pkl")) for nam in excluding])))
    load_fs = np.setdiff1d(inc_fs, exc_fs)
    res = edict()
    for fil in load_fs:
        res[fil.stem] = load_pkl(fil, verbose)                                                                                                                                  
    return res

# save a dict into a folder
def save_pkl_dict2folder(folder, res, is_force=False, verbose=True):
    assert isinstance(res, dict)
    for ky, v in res.items():
        save_pkl(folder/f"{ky}.pkl", v, is_force=is_force, verbose=verbose)

# load file from pkl
def load_pkl(fil, verbose=True):
    if verbose:
        print(f"Load file {fil}")
    with open(fil, "rb") as f:
        result = pickle.load(f)
    return result

# save file to pkl
def save_pkl(fil, result, is_force=False, verbose=True):
    if not fil.parent.exists():
        fil.parent.mkdir()
        if verbose:
            print(fil.parent)
            print(f"Create a folder {fil.parent}")
    if is_force or (not fil.exists()):
        if verbose:
            print(f"Save to {fil}")
        with open(fil, "wb") as f:
            pickle.dump(result, f)
    else:
        if verbose:
            print(f"{fil} exists! Use is_force=True to save it anyway")
        else:
            pass
