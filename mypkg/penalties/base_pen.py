import numbers
import torch
import numpy as np
class PenaltyBase():
    """The penalty function
    """
    def __init__(self, lams, sel_idx=None):
        """args:
            lams: Penalty parameters
            sel_idx: The set of select idxs from 0 to d-1.
        """
        self.sel_idx = sel_idx 
        self.lams = lams
    
    
    def _preprocess(self, Gam):
        N, d = Gam.shape
        if self.sel_idx is None:
            self.sel_idx = torch.arange(d)
            
        vec = torch.zeros(d)
        vec[self.sel_idx] = 1
        vec = vec.bool()
        self.sel_idx = vec
        
        if isinstance(self.lams, numbers.Number):
            self.lams = torch.ones(len(self.sel_idx))*self.lams
            self.lams = self.lams.to(torch.get_default_dtype())
            
        if isinstance(self.lams, np.ndarray):
            self.lams = torch.tensor(self.lams)
            
    def __call__(self, Gam):
        pass