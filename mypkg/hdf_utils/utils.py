import numpy as np
import torch

def gen_lam_seq(low, up, nlam):
    """
    Generate a sequence of lambda values.

    Parameters:
    low (float): The lower bound of the sequence.
    up (float): The upper bound of the sequence.
    nlam (int): The number of lambda values to generate.

    Returns:
    numpy.ndarray: A sequence of lambda values.
    """
    dlts =  np.linspace(-5, np.log(up-low), nlam-1)
    lam_seq = low + np.exp(dlts)
    lam_seq = np.concatenate([[low], lam_seq])
    return lam_seq

def gen_int_ws(npts, type_="simpson"):
    """
    Generate the weights for the numerical integartion
    args:
        npts: the num of pts for numerical integartion
        type_(str): the weights type
            type_ = "naive": Most naive way 
            type_ = "simpson": Using simpson rule to do the integration.
    returns:
        ws (Tensor): the integrations ws, npts vector
    """
    type_= type_.lower()
    if type_.startswith("nai"):
        ws = torch.ones(npts)/npts
    elif type_.startswith("sim"):
        Q = npts - 1
        ws = torch.ones(npts)/(3*Q)
        ws[2:-1:2] = ws[2:-1:2]*2
        ws[1:-1:2] = ws[1:-1:2]*4
        if (npts)%2 == 0:
           # apply Simpson's 3/8 rule on last 3 intervals
            ws[-1] = 3/8/Q
            ws[-2] = 9/8/Q
            ws[-3] = 9/8/Q
            ws[-4] = (1/Q) * (1/3 + 3/8)
    else: 
        raise NotImplementedError
    
    return ws

def integration_fn(fs, ws="simpson"):
    """
    Do numerical integeration for function f(x);
    args:
        fs (array, Tensor): a vector (num_pts) or a matrix (num_pts x num_fs)
        ws (array, Tensor, or str): the weights used for the integration 
            ws = "naive": Most naive way 
            ws = "simpson": Using simpson rule to do the integration.
            ws is vector: sum(ws) == 1 and ws >= 0
    returns:
        the integartion value (Tensor): the integrations, 1 x num_fs vector
    """
    if not isinstance(fs, torch.Tensor):
        fs = torch.tensor(fs)
        
    num_pts = fs.shape[0]
    if isinstance(ws, str):
        ws = gen_int_ws(num_pts, ws.lower())
    elif not isinstance(ws, torch.Tensor):
        ws = torch.tensor(ws)
        
        
    ws = ws[:, None]
    if fs.ndim == 1:
        fs = fs[:, None]
    
    return (fs*ws).sum(axis=0)
    
    
