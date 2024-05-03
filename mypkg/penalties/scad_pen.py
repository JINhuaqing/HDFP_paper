from .base_pen import PenaltyBase
import torch
import pdb

class SCAD(PenaltyBase):
    def __init__(self, lams, a=2.7, C=1, sel_idx=None):
        """args:
            C: the parameter for the sqaure term. 
            a: parameters for SCAD, Scalar
            lams: Penalty parameters
            sel_idx: The set of select idxs from 0 to d-1.
        """
        super().__init__(lams=lams, sel_idx=sel_idx)
        self.a = a
        self.C = C
        assert self.a > (1/self.C+1)
        
    def evaluate(self, Gam):
        """
            evaluate the penalty value
            args:
                Gam: The parameters to be penalised. N x d, d num of ROIs
        """
        self._preprocess(Gam)
        sel_Gam = Gam[:, self.sel_idx]
        sel_Gam_norms = torch.norm(sel_Gam, dim=0)
        sel_Gam_norms_out = sel_Gam_norms.clone()
        
        idx1 = sel_Gam_norms <= self.lams
        idx2 = torch.bitwise_and(sel_Gam_norms>self.lams, sel_Gam_norms<=self.a*self.lams)
        idx3 = sel_Gam_norms > self.a*self.lams
        sel_Gam_norms_out[idx1] = self.lams[idx1]*sel_Gam_norms[idx1]
        sel_Gam_norms_out[idx3] = self.lams[idx3]**2*(self.a+1)/2
        
        tmp_num = self.lams[idx2]**2 + sel_Gam_norms[idx2]**2 - 2*self.a*self.lams[idx2]*sel_Gam_norms[idx2]
        sel_Gam_norms_out[idx2] = -tmp_num/2/(self.a-1)
        return sel_Gam_norms_out.sum()
    
    def __call__(self, Gam, C=None):
        """
            Apply the penalty. 
            We solve C/2 ||z-theta||^2 + SCAD_lambda,a(theta)
            args:
                Gam: raw solver of Gam
        """
        eps = 1e-10
        if C is not None:
            self.C = C
            
        self._preprocess(Gam)
        Gam_norms = torch.norm(Gam, dim=0)
        Gam_norms_out = Gam_norms.clone()
        
        #pdb.set_trace()
        idx1 = torch.bitwise_and(Gam_norms <= (self.lams + self.lams/self.C), self.sel_idx)
        idx2 = torch.bitwise_and(Gam_norms>(self.lams+self.lams/self.C), 
                                                Gam_norms<=self.a*self.lams)
        idx2 = torch.bitwise_and(idx2, self.sel_idx)
                                 
        Gam_norms_out[idx1] = Gam_norms[idx1] - self.lams[idx1]/self.C
        tmp_num = self.C * (self.a-1)*Gam_norms[idx2] - self.a * self.lams[idx2]
        tmp_den = self.C*self.a - self.C - 1
        Gam_norms_out[idx2] = tmp_num/tmp_den
        Gam_norms_out[Gam_norms_out<0] = 0
        
        # avoid divide-by-0 error
        Gam_out = Gam * Gam_norms_out/(Gam_norms+eps)
        return Gam_out
        
