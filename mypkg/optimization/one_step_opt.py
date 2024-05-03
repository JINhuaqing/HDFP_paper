import torch
import numpy as np
from easydict import EasyDict as edict
from utils.matrix import col_vec_fn, col_vec2mat_fn, gen_Dmat, conju_grad,  cholesky_inv
from projection import euclidean_proj_l1ball
from models.linear_model import LinearModel
from models.logistic_model import LogisticModel
from utils.misc import  _set_verbose_level, _update_params
from pprint import pprint
import pdb
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 

def theta_proj(thetal, q, N, R):
    """Proj theta to the space \|theta_1\|_1 + \sum_j \|theta_2j\|_2 < R
       args:
           thetal: the theta to be projected.
           q: Num of covaraites
           N: The bspline space dim
           R: the radius of the space
    """
    alpl = thetal[:q]
    GamlN= col_vec2mat_fn(thetal[q:], nrow=N)
    GamlN_l2norm = torch.norm(GamlN, dim=0)
    cat_vec = torch.cat([alpl, GamlN_l2norm])
    cat_vec_after_proj = torch.tensor(euclidean_proj_l1ball(cat_vec.numpy(), R))
    alpl_after_proj = cat_vec_after_proj[:q]
    GamlN_after_proj = cat_vec_after_proj[q:]*GamlN/GamlN_l2norm
    thetal_after_proj = torch.cat([alpl_after_proj, col_vec_fn(GamlN_after_proj)])
    return thetal_after_proj

class OneStepOpt():
    """One step for the optimization"""
    def __init__(self, Gamk, rhok, model, penalty,  
                 theta_init=None, verbose=0,  **other_paras):
        """args:
                Gamk: The Gam from last update, matrix of N x d
                rhok: The vector rho from last update, vector of dN
                model: likelihood model: LogisticModel or LinearModel
                penalty: The penalty fn, SCAD or (to be written)
                theta_init: The initial value of theta, vec of q+dN
                            Note theta = [alp, Gam/sqrt(N)]
                verbose: log level, from 0 to 3, increase the output
                other_paras: other parameters
                    alpha : the parameter for CPRSM, scalar, alpha and alp are different, alpha \in (0, 1)
                    beta: the parameter for CPRSM, scalar, beta>0
                    R: The radius of the projection space
                    N_eps: the stop criteria for Newton-related method
                    N_maxit: the max num of iterations for Newton-related method
                    linear_theta_update: 
                                if conjugate, use conj_grad
                                if cholesky_solve, use cholesky_solve
                                if cholesky_inv, get the cholesky inverse
                    linear_mat: The matrix for linear_theta_update:
                                if conjugate, None
                                if cholesky_solve, L where the cholesky decom of left_mat=LL^T under linear, 
                                if cholesky_inv, inverse of left_mat
                    is_BFGS: Whether using BFGS for updating theta or not, defualt is True. 
                             If "adaptive": try BFGS first; if not convergence, use Newton method.
        """
        _set_verbose_level(verbose, logger)
            
        
        if isinstance(model, LinearModel):
            paras = edict({
                'alpha': 0.9,
                'beta': 1,
                "R": 1e5,
                "linear_theta_update": "cholesky_inv",
                "linear_mat": None,
            })
        elif isinstance(model, LogisticModel):
            paras = edict({
                'alpha': 0.9,
                'beta': 1,
                "R": 1e5,
                "is_BFGS": "adaptive",
                "N_eps": 1e-4, 
                "N_maxit": 100,
            })
            
        paras = _update_params(other_paras, paras, logger)
        
        
        self.N, self.d = Gamk.shape
        self.q = model.Z.shape[-1]
        assert len(rhok) == self.N * self.d
        
        if theta_init is None:
            theta_init = torch.zeros(self.q+self.d*self.N)
    
        self.model = model
        self.penalty = penalty
        self.Gamk = Gamk
        self.rhok = rhok
        self.thetak = theta_init # in fact, we do not need this for linear model
        self.alpk = None
        self.paras = paras
        self.D = gen_Dmat(self.d, self.N, self.q)
        
        
        if isinstance(model, LinearModel):
            self.linear_theta_update = paras.linear_theta_update.lower()
            assert self.linear_theta_update in ["conjugate", "cholesky_solve", "cholesky_inv"]
            self.linear_mat = paras.linear_mat
        elif isinstance(model, LogisticModel):
            if isinstance(paras.is_BFGS, str):
                assert paras.is_BFGS.lower().startswith("ada")
            if  paras.is_BFGS:
                self.BFGS_success = True
            
        logger.debug(f"The paras is {paras}.")
        
    
    def _update_theta_linearmodel(self):
        """First step of optimization, update theta under linear model.
            Note that under the linear case, we do not need iterations.
        """
        if self.model.lin_tm_der is None:
            self.model._linear_term_der()
            
        right_vec = (self.D.T@self.rhok +
                     self.paras.beta*self.D.T@col_vec_fn(self.Gamk)/np.sqrt(self.N) + 
                     (self.model.Y.unsqueeze(-1)*self.model.lin_tm_der).mean(dim=0)/self.model.sigma2)
        if self.linear_theta_update.startswith("cholesky_inv"):
            if self.linear_mat is None:
                tmp_mat = self.model.lin_tm_der.T@self.model.lin_tm_der/len(self.model.Y) # (q+dN) x (q+dN)
                left_mat = tmp_mat/self.model.sigma2 + self.paras.beta * self.D.T@self.D
                self.linear_mat = cholesky_inv(left_mat)
            thetak_raw = self.linear_mat @ right_vec;
                
        elif self.linear_theta_update.startswith("conjugate"):
            tmp_mat = self.model.lin_tm_der.T@self.model.lin_tm_der/len(self.model.Y) # (q+dN) x (q+dN)
            left_mat = tmp_mat/self.model.sigma2 + self.paras.beta * self.D.T@self.D
            
            thetak_raw = conju_grad(left_mat, right_vec)
            
        elif self.linear_theta_update.startswith("cholesky_solve"):
            if self.linear_mat is None:
                tmp_mat = self.model.lin_tm_der.T@self.model.lin_tm_der/len(self.model.Y) # (q+dN) x (q+dN)
                left_mat = tmp_mat/self.model.sigma2 + self.paras.beta * self.D.T@self.D
                self.linear_mat = torch.linalg.cholesky(left_mat)
            thetak_raw = torch.cholesky_solve(right_vec.reshape(-1, 1), self.linear_mat).reshape(-1);
        else:
            raise TypeError(f"No such type, {self.linear_theta_update}")
            
        
        self.thetak = theta_proj(thetak_raw, self.q, self.N, self.paras.R) # projection
    
    def _update_theta(self):
        """First step of optimization, update theta with Newton method
           This step can be slow
        """
        def _obj_fn_der1(thetalk):
            alplk = thetalk[:self.q]
            Gamlk = col_vec2mat_fn(thetalk[self.q:], nrow=self.N)*np.sqrt(self.N)
            der1_p1 = -self.model.log_lik_der1(alplk, Gamlk)
            der1_p2 = -self.D.T @ self.rhok
            der1_p3 = self.paras.beta * (self.D.T@self.D@thetalk - self.D.T@col_vec_fn(self.Gamk)/np.sqrt(self.N))
            der1 = der1_p1 + der1_p2 + der1_p3
            return der1
        
        def _obj_fn_der2(thetalk):
            alplk = thetalk[:self.q]
            Gamlk = col_vec2mat_fn(thetalk[self.q:], nrow=self.N)*np.sqrt(self.N)
            der2_p1 = -self.model.log_lik_der2(alplk, Gamlk)
            der2_p2 = self.paras.beta * self.D.T @ self.D 
            der2 = der2_p1 + der2_p2 
            return der2
        thetal = self.thetak.clone()
        alpl = thetal[:self.q]
        Gaml = col_vec2mat_fn(thetal[self.q:], nrow=self.N)*np.sqrt(self.N)
        for ix in range(self.paras.N_maxit):
            der1 = _obj_fn_der1(thetal)
            der2 = _obj_fn_der2(thetal)
            
            theta_last = thetal.clone()
            der2_inv = torch.linalg.pinv(der2, hermitian=True, rtol=1e-7)
            thetal = theta_last - der2_inv @ der1 # update 
            
            stop_cv = torch.norm(thetal-theta_last)/torch.norm(thetal)
            if stop_cv <= self.paras.N_eps:
                break
        if ix == (self.paras.N_maxit-1):
            logger.warning("The NR algorithm may not converge")
        thetal = theta_proj(thetal, self.q, self.N, self.paras.R) # projection
        self.thetak = thetal
        
        
    def _update_theta_BFGS(self):
        """Update theta with BFGS, faster than Newtown
        """
        def _obj_fn(thetalk):
            alplk = thetalk[:self.q]
            Gamlk = col_vec2mat_fn(thetalk[self.q:], nrow=self.N)*np.sqrt(self.N)
            der0_p1 = -self.model.log_lik(alplk, Gamlk)
            der0_p2 = -self.rhok @ (self.D @ thetalk - col_vec_fn(self.Gamk)/np.sqrt(self.N))
            der0_p3 = self.paras.beta * torch.norm(self.D @ thetalk - col_vec_fn(self.Gamk)/np.sqrt(self.N), p=2)**2/2
            der0 = der0_p1 + der0_p2 + der0_p3
            return der0
        
        def _obj_fn_der1(thetalk):
            alplk = thetalk[:self.q]
            Gamlk = col_vec2mat_fn(thetalk[self.q:], nrow=self.N)*np.sqrt(self.N)
            der1_p1 = -self.model.log_lik_der1(alplk, Gamlk)
            der1_p2 = -self.D.T @ self.rhok
            der1_p3 = self.paras.beta * (self.D.T@self.D@thetalk - self.D.T@col_vec_fn(self.Gamk)/np.sqrt(self.N))
            der1 = der1_p1 + der1_p2 + der1_p3
            return der1
        
        def _obj_fn_der2(thetalk):
            alplk = thetalk[:self.q]
            Gamlk = col_vec2mat_fn(thetalk[self.q:], nrow=self.N)*np.sqrt(self.N)
            der2_p1 = -self.model.log_lik_der2(alplk, Gamlk)
            der2_p2 = self.paras.beta * self.D.T @ self.D 
            der2 = der2_p1 + der2_p2 
            return der2
        
        def _wolfe_cond_check(stepsizel, c1=1e-4, c2=0.9):
            """c1, c2 from wiki pg (Wolfe condition)
            """
            left1 = _obj_fn(thetal+stepsizel*dlt_thetal_raw)-_obj_fn(thetal)
            right1 = c1*stepsizel*dlt_thetal_raw @ der1
            
            left2 = - dlt_thetal_raw @ _obj_fn_der1(thetal+stepsizel*dlt_thetal_raw)
            right2 = -c2* dlt_thetal_raw @ der1
            return (left1 <= right1).float() +  (left2 <= right2).float()
        
        
        # for BFGS, we show choose stepsize each step
        # but I checked, fix stepsize=1 is a decent choice, so for speed, I fix it.
        # if BFGS is not satisfactory, you may consider to make more candidate stepsizes (on Nov 14, 2023)
        can_stepsizes = [1]
        #can_stepsizes = torch.linspace(1e-4, 5, 10)
        # initial
        thetal = self.thetak.clone()
        der1 = _obj_fn_der1(thetal)
        Binvl = torch.eye(thetal.shape[0])
        #Binvl = torch.linalg.pinv(_obj_fn_der2(thetal), hermitian=True, rtol=1e-7)
        # get step size
        stepsizel = 1
        for ix in range(self.paras.N_maxit):
            # get move step 
            dlt_thetal = - stepsizel * Binvl @ der1
            
            # get new thetal
            thetal = thetal + dlt_thetal
            
            # stop criterion 
            stop_cv = torch.norm(dlt_thetal)/torch.norm(thetal)
            if (stop_cv <= self.paras.N_eps) and (ix >=1):
                break
                
            # save old der1
            der1_last = der1.clone()
            
            # new der1
            der1 = _obj_fn_der1(thetal)
            
            # yl
            yl = der1 - der1_last
            
            # get new Binvl
            Binvl_p1 = torch.eye(thetal.shape[0]) - torch.outer(dlt_thetal, yl)/(torch.inner(dlt_thetal, yl))
            Binvl_p2 = torch.outer(dlt_thetal, dlt_thetal)/(torch.inner(dlt_thetal, yl))
            Binvl = Binvl_p1 @ Binvl @ Binvl_p1.T + Binvl_p2
            
            # get step size
            if len(can_stepsizes) > 1:
                dlt_thetal_raw = -Binvl @ der1
                obj_fns = [_obj_fn(thetal+can_stepsize*dlt_thetal_raw).item() for can_stepsize in can_stepsizes]
                stepsizel = can_stepsizes[np.argmin(obj_fns)]
            else:
                stepsizel = can_stepsizes[0]
            
            if False:
                _wolfe_cond_check(stepsizel)
        
        if ix == (self.paras.N_maxit-1):
            if not isinstance(self.paras.is_BFGS, str):
                logger.warning("The BFGS algorithm may not converge.")
            else:
                logger.info("The BFGS algorithm may not converge.")
            self.BFGS_success = False
        if (not isinstance(self.paras.is_BFGS, str)) or self.BFGS_success:
            thetal = theta_proj(thetal, self.q, self.N, self.paras.R) # projection
            self.thetak = thetal
                
    
    def _update_rho(self):
        """Second/Fourth step, update rho to get rho_k+1/2/rho_k"""
        assert self.thetak is not None
        self.rhok = self.rhok - self.paras.alpha*self.paras.beta*(self.D@self.thetak-col_vec_fn(self.Gamk)/np.sqrt(self.N))
        
    
    def _update_Gam(self):
        """Third step, update Gam to get Gam_k+1"""
        GamNk_raw = col_vec2mat_fn(self.D@self.thetak - self.rhok/self.paras.beta, nrow=self.N)
        GamNk = self.penalty(GamNk_raw, C=self.paras.beta)
        self.Gamk = GamNk * np.sqrt(self.N)
    
    def __call__(self):
        if isinstance(self.model, LinearModel):
            self._update_theta_linearmodel()
        else:
            if isinstance(self.paras.is_BFGS, str): 
                self._update_theta_BFGS() 
                if not self.BFGS_success:
                    logger.info("As BFGS algorithm may not converge, we use Newton method.")
                    self._update_theta()
            elif self.paras.is_BFGS:
                self._update_theta_BFGS()
            else:
                self._update_theta()
        self._update_rho()
        self._update_Gam()
        self._update_rho()
        self.alpk = self.thetak[:self.q]
