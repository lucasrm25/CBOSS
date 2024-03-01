'''
Author: Lucas Rath

Implements the Failure-Robust Constrained Hierarchical Expected Improvement (FRCHEI) acquisition function,
which can handle equality and inequality constraints:

    FRCHEI(x) = HEI(x) * PFeas(x)**beta_feas * PSucc(x)**beta_succ
'''

import numpy as np
from scipy import stats
from typing import Callable, List
import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from pyro.distributions.multivariate_studentt import MultivariateStudentT
from torch.distributions.bernoulli import Bernoulli
import scipy as sp
from CBOSS.utils.torch_utils import ModuleX
from CBOSS.bayesian_models.regression import BayesianRegression
from CBOSS.bayesian_models.classification import BayesianClassification

''' Acquisition Functions
============================================================== '''

class FRCHEI(ModuleX):
    ''' Failure-robust Constrained Hierarchical Expected Improvement (FRCHEI) acquisition function
        FRCHEI = HEI * PFeas**beta_feas * PSucc**beta_succ
    '''
    eps = 1e-10
    
    class PFeasModule(ModuleX):
        ''' Calculates Prod_i p(yi < 0 | x) given p(yi) ~ N(mu, sigma^2) or p(yi) ~ t(df, mu, sigma^2)
        '''
        def __init__(self, reg_g:List[BayesianRegression]) -> None:
            super().__init__()
            self.reg_g = reg_g
            
        @torch.no_grad()
        def forward(self, X_pred:torch.Tensor, include_noise:bool=False):
            assert X_pred.dim() == 2 and len(X_pred) == 1, 'X must be a 2D tensor with shape (1, d)'
            Pfeas = []
            for reg_gi in self.reg_g:
                ci_post = reg_gi(X_pred=X_pred, include_noise=include_noise)
                if isinstance(ci_post, MultivariateNormal):
                    Pfeas.append( Normal(loc=ci_post.loc, scale=ci_post.covariance_matrix**0.5).cdf(torch.tensor(0.0)) )
                elif isinstance(ci_post, MultivariateStudentT):
                    Pfeas.append( torch.as_tensor(sp.stats.t(df=ci_post.df.numpy(), loc=ci_post.loc, scale=ci_post.covariance_matrix**0.5).cdf(0.0)) )
                else:
                    raise NotImplementedError
            return torch.tensor(Pfeas).prod()
    
    def __init__(self, y_best:float, reg_f:BayesianRegression, reg_g:List[BayesianRegression]=None, class_h:BayesianClassification=None, beta_feas:float=1.0, beta_succ:float=1.0) -> None:
        super().__init__()
        self.reg_f = reg_f
        self.reg_g = reg_g
        self.Pfeas = self.PFeasModule(reg_g=reg_g) if reg_g is not None else None
        self.class_h = class_h
        self.y_best = y_best
        self.beta_feas = beta_feas
        self.beta_succ = beta_succ
        self.handle_constraints = self.reg_g is not None
        self.handle_failures = self.class_h is not None
    
    @torch.no_grad()
    def forward(self, X_pred:torch.Tensor, log:bool) -> torch.Tensor:
        assert X_pred.dim() == 2 and len(X_pred) == 1, 'X must be a 2D tensor with shape (1, d)'

        y_post = self.reg_f(X_pred=X_pred, include_noise=False)
        PFeas  = self.Pfeas(X_pred=X_pred, include_noise=False) if self.handle_constraints else None
        PSucc  = self.class_h(X_pred=X_pred).mean if self.handle_failures else torch.tensor(1.0, dtype=X_pred.dtype)
        
        assert isinstance(y_post, (MultivariateNormal, MultivariateStudentT)), 'reg_f must evaluate to a MultivariateNormal or MultivariateStudentT distribution object'
        assert isinstance(PSucc, torch.Tensor) or PSucc is None, 'PSucc must evaluate to a torch.Tensor object == p(reg_h(X) = 1 | dataset)'
        # assert isinstance(PSucc, Bernoulli) or PSucc is None, 'reg_h must evaluate to a Bernoulli distribution object == p(reg_h(X) | dataset)'
        
        if isinstance(y_post, MultivariateNormal):
            EI = expected_improvement(y_post=self.reg_f(X_pred=X_pred, diagonal=True, include_noise=False), y_best=self.y_best)
        elif isinstance(y_post, MultivariateStudentT):
            EI = hierarchical_expected_improvement(y_post=self.reg_f(X_pred=X_pred, diagonal=True, include_noise=False), y_best=self.y_best)

        if not log:
            FRCHEI = EI * PFeas**self.beta_feas * PSucc**self.beta_succ
            return FRCHEI
        else:
            log_FRCHEI = np.log(EI+self.eps) + self.beta_feas * np.log(PFeas+self.eps) + self.beta_succ * np.log(PSucc+self.eps)
            return log_FRCHEI

@torch.no_grad()
def expected_improvement(y_post:MultivariateNormal, y_best:float) -> torch.Tensor:
    ''' Calculates the expected improvement for MINIMIZING the function:  
            E[I(y_pred(X))]
                y_pred ~ p(y_pred|y)
                I(y_pred(X)) = E[ max(0, y_pred(X) - y_best) ]
        where y_post is a Gaussian distribution N(mu,sigma^2)
    Args:
        - y_post: p(y_pred|y)
        - y_best: minimum y observed so far
    '''
    mu = y_post.loc
    sigma = y_post.covariance_matrix.diagonal() ** 0.5
    epsilon_best = (y_best - mu) / sigma
    normal = Normal(loc=0., scale=1.0)
    EI = (y_best - mu) * normal.cdf(epsilon_best) + sigma * torch.exp(normal.log_prob(epsilon_best))
    EI = torch.clip(EI, min=0.0)
    return EI

@torch.no_grad()
def hierarchical_expected_improvement(y_post:MultivariateStudentT, y_best:float) -> torch.Tensor:
    ''' Calculates the expected improvement for MINIMIZING the function:  
            E[I(y_pred(X))]:
                y_post ~ p(y_pred|y)
                I(y_pred(X)) = E[ max(0, y_best - y_pred(X)) ]     --> positive value (0 means no exp. improvement)
        where y_post is a student-t distribution T(nu,mu,sigma^2)
    Args:
        - y_post: p(y_pred|y)
        - y_best: minimum y observed so far

    NOTE: at the moment we have to do a workaround using scipy.t distribution because 
    torch.distributions.StudentT.cdf method has not been implemented yet
    '''
    nu = y_post.df.numpy()
    mu = y_post.loc.numpy()
    sigma = y_post.covariance_matrix.diagonal().numpy() ** 0.5
    y_best = np.asarray(y_best)
    
    standard_t = stats.t(df=nu)
    epsilon_best = (y_best - mu) / sigma
    
    HEI = sigma * ( epsilon_best * np.exp(standard_t.logcdf(x=epsilon_best)) + (nu + epsilon_best**2)/(nu - 1) * np.exp(standard_t.logpdf(x=epsilon_best)) )
    HEI = torch.as_tensor(HEI)
    HEI = torch.clip(HEI, min=0.0)
    return HEI
