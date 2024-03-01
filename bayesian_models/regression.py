'''
Author: Lucas Rath
'''

import sys
import numpy as np
from numpy.linalg import solve
from typing import Tuple, Any
from functools import partial
import time
import torch
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from pyro.distributions.multivariate_studentt import MultivariateStudentT
from pyro.distributions.inverse_gamma import InverseGamma

from CBOSS.utils.torch_utils import *
from CBOSS.bayesian_models.means import ZeroMean
from CBOSS.bayesian_models.kernels import PolynomialKernel
from CBOSS.bayesian_models.features import IFeatures


''' Regression Modules
=============================================================='''

class BayesianRegression(ModuleX):
    def __new__(cls, X:torch.Tensor, y:torch.Tensor, *args, **kwargs) -> 'BayesianRegression':
        # if y is multimensional, then create a MultiOutputRegression object
        if y.dim() > 1:
            regs = [cls(X=X, y=y_i, *args, **kwargs) for y_i in y.T]
            return MultiOutputBayesianRegression(regressors=regs)
        else:
            return super().__new__(cls)
    
    def __init__(self, X:torch.Tensor, y:torch.Tensor, drop_duplicates:bool=True) -> None:
        super().__init__()
        self.set_training_data(X=X, y=y, drop_duplicates=drop_duplicates)
        
    @torch.no_grad()
    def set_training_data(self, X:torch.Tensor, y:torch.Tensor, drop_duplicates:bool=True):
        ''' Set training data and recalculate training features
        '''
        assert isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)
        assert len(X) == len(y), f'len(X)={len(X)} should be equal to len(y)={len(y)}'
        assert y.dim() == 1 and X.dim() == 2
        assert self.training
        # remove invalid targets
        idx_valid = ~ (torch.isnan(y) | torch.isinf(y))
        X, y = X[idx_valid], y[idx_valid]
        if drop_duplicates:
            # remove duplicate inputs
            idx_unique = np.unique(X, axis=0, return_index=True)[1]
            X = X[idx_unique]
            y = y[idx_unique]
        # store
        self.register_buffer('y', torch.as_tensor(y,dtype=self.dtype))
        self.register_buffer('X', torch.as_tensor(X,dtype=self.dtype))
        self.N = len(self.X)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, X_pred:torch.Tensor, diagonal:bool=False, include_noise:bool=False):
        ''' Calculates the predictive posterior of y: p( y_pred(X_pred) | dataset )
        '''
        if self.training:
            return self.predictive_prior(X_pred=X_pred, diagonal=diagonal, include_noise=include_noise)
        else:
            return self.predictive_posterior(X_pred=X_pred, diagonal=diagonal, include_noise=include_noise)

class MultiOutputBayesianRegression(ModuleX):
    ''' Bayesian Regression with multiple independent outputs
    '''
    def __init__(self, regressors:List[BayesianRegression]) -> None:
        super().__init__()
        self.regs = torch.nn.ModuleList(regressors)
        
    def __len__(self):
        return len(self.regs)

    def fit(self, *args, **kwargs):
        for reg in self.regs:
            reg.fit(*args, **kwargs)

    @torch.no_grad()
    def eval(self):
        for reg in self.regs:
            reg.eval()

    def forward(self, *args, **kwargs) -> List[Any]:
        outs = [reg(*args, **kwargs) for reg in self.regs]
        return outs


''' Bayesian Regression
=============================================================='''

class StudentTProcessRegression(BayesianRegression):
    ''' Student-t Process Regression.
    
    This model arised from using a Kernel Trick with a Bayesian linear regression method with 
    normal-inverse-gamma prior distribution P(alpha,sigma^2).

    Prior:
        p(alpha, sigma2) = p(alpha | sigma2) * p(sigma2)

            p(sigma2) = IG(L,sigma2m)
            p(alpha | sigma2) = N(mu_a, sigma2 * Sigma_a)

    Likelihood:
        y(X) | alpha, sigma2 ~ N( F(X) @ alpha, sigma2 * I_N )
    
    Parameters:
        {sigma2, alpha}

    See: https://distribution-explorer.github.io/continuous/inverse_gamma.html to help designing the gamma prior distr.
    '''
    def __init__(
            self,
            X:torch.Tensor, 
            y:torch.Tensor,
            mean_fun:torch.nn.Module = ZeroMean(), 
            kernel:torch.nn.Module = PolynomialKernel(variance_prior=0.05, degree=2),
            train_sigma2:bool = True,
            sigma2m_prior:float = 0.15,
            sigma2m_hyperprior:torch.distributions.Distribution = Gamma(2.,15.),
            L_prior:float = 2.5,
            L_hyperprior:torch.distributions.Distribution = Gamma(2., 0.5)
    ):
        super().__init__(X=X, y=y, drop_duplicates=True)
        self.K = kernel
        self.mu = mean_fun
        self.sigma2m_prior_ = HyperParameter(
            tensor=torch.tensor(sigma2m_prior), requires_grad=train_sigma2, constraint=Interval(1e-2,1e0),
            hyperprior = lambda self: sigma2m_hyperprior
        )
        self.L_prior_ = HyperParameter(
            tensor=torch.tensor(L_prior), requires_grad=train_sigma2, constraint=Interval(1e-1,1e2),
            hyperprior = lambda self: L_hyperprior
        )

    @property
    def a_prior(self):
        # return self.a_prior_()
        return self.L_prior_()/2.

    @property
    def b_prior(self):
        # return self.b_prior_()
        return self.sigma2m_prior_() * self.L_prior_() / 2.

    @property
    def L_prior(self):
        # return 2 * self.a_prior
        return self.L_prior_()

    @property
    def sigma2m_prior(self):
        # return self.b_prior / self.a_prior
        return self.sigma2m_prior_()

    @torch.no_grad()
    def eval(self):
        ''' Pre-store matrices that are kept constant during evaluation
        '''
        self.m_X = self.mu(self.X)
        self.K_XX = self.K(self.X,self.X)

        # auxiliary matrices needed for common matrix inversions
        self.C = torch.linalg.cholesky(self.K_XX + torch.eye(self.N), upper=False)                         # V = (K_XX + I)^0.5  ->  V @ V.T = K_XX + I
        self.CM = torch.linalg.solve_triangular(self.C, (self.y - self.m_X)[:,None], upper=False)[:,0]     # (K_XX + I)^0.5 \ (y - m_X)

        # calculate posterior terms
        self.a_post = self.a_prior + self.N/2
        self.b_post = 0.5 * ( self.L_prior * self.sigma2m_prior + torch.inner(self.CM,self.CM) )   # CM.T @ CM =  (y - m_X) * (K_XX + I)^-1 \ (y - m_X)
        self.sigma2m_post = self.b_post / self.a_post
        self.L_post = 2 * self.a_post

        ''' Put network in evaluation mode '''
        super().eval()

    def sigma2_prior(self):
        ''' Calculates the prior distribution p(sigma^2) = IG(a,b)
        '''  
        sigma2_prior = InverseGamma(
            concentration = self.a_prior, 
            rate          = self.b_prior
        )
        return sigma2_prior

    @torch.no_grad()
    def sigma2_posterior(self):
        ''' Returns the posterior distribution p(sigma2 | y)
        '''  
        sigma2_post = InverseGamma(
            concentration = self.a_post, 
            rate          = self.b_post
        )
        return sigma2_post

    def predictive_prior(self, X_pred:torch.Tensor, diagonal:bool=False, include_noise:bool=False):
        ''' Calculates the predictive prior of y: p(y|theta)
        '''
        Ip = torch.eye(len(X_pred))
        mu_prior = self.mu(X_pred)
        Sigma_prior = self.sigma2m_prior * ( Ip * include_noise + self.K(X_pred, X_pred) ) + 1e-4*Ip
        y_pred_prior = MultivariateStudentT(
            df          = self.L_prior,
            loc         = mu_prior, 
            scale_tril  = torch.linalg.cholesky(Sigma_prior) if not diagonal else torch.diag(torch.diag(Sigma_prior)**0.5)
        )
        return y_pred_prior 

    @torch.no_grad()
    def predictive_posterior(self, X_pred:torch.Tensor, diagonal:bool=False, include_noise:bool=True):
        ''' Calculates the predictive posterior of y: p(y_pred | y, theta)
        '''
        assert not self.training, 'the posterior can only be evaluated in `eval` mode'
        # X_pred = torch.as_tensor(X_pred)

        Ip = torch.eye(len(X_pred))
        K_XXp, K_XpXp = self.K(self.X,X_pred), self.K(X_pred,X_pred)
        m_Xp = self.mu(X_pred)

        # CK = (K_XX + I)^0.5 \ K_XXp
        CK = torch.linalg.solve_triangular(self.C, K_XXp, upper=False)

        # CM = (K_XX + I)^0.5 \ (y - m_X)
        # CK.mT @ CM = K_XXp.T @ (K_XX + I) \ (y - m_X)
        m_post = m_Xp + CK.mT @ self.CM
        # CK.mT @ CK = K_XXp.T @ (K_XX + I) \ K_XXp
        Sigma_post = self.sigma2m_post * ( Ip * include_noise + K_XpXp - CK.mT @ CK ) + 1e-4*Ip

        y_pred_post = MultivariateStudentT(
            df          = self.L_post,
            loc         = m_post, 
            scale_tril  = torch.linalg.cholesky(Sigma_post) if not diagonal else torch.diag(torch.diag(Sigma_post)**0.5)
        )
        return y_pred_post

    def theta_posterior_lobprob(self):
        ''' 
        Returns the log of the joint distribution: 
            `log p(theta,y) = log [ p(y|theta) * p(theta) ] = log p(y|theta) + log p(theta)`
        NOTE: this function is used to obtain the log_prob of the unnormalized posterior: 
            `p(theta|y) * p(y) = p(y|theta) * p(theta)`
        '''
        # log p(y|theta)
        y_logpdf = self.predictive_prior(X_pred=self.X, include_noise=True).log_prob(self.y)
        #  log p(theta)
        theta_logpdf = sum([p.hyperprior().log_prob(p()) for p in self.hyperparameters().values()])
        #  log p(y,theta)
        return y_logpdf + theta_logpdf
    
    def fit(self, *args, **kwargs):
        return self.MMAP(*args, **kwargs)

    def MMAP(self, lr:float=1.0, maxiter:int=300, tolerance_change:float=1e-9, tolerance_grad:float=1e-5, disp:bool=True) -> None:
        ''' Optimize hyperparameters according to MMAP (maximum marginal a posteriori) estimation:
            
            theta = argmax_{theta} log p( theta | y )

        Calculates the log of the marginal posterior of the hyper-prior parameters:
                p(theta|y) = p(y|theta) * log p(theta) / p(y)

            Applying the log, we have:
                logp(theta|y) = cnst + logp(y|theta) + logp(theta)
        '''
        assert self.training, 'MMAP can only be performed in train mode'
        # store hyper-parameters for logging purposes
        if disp:
            hp = {k:p().detach().numpy().copy() for k,p in self.hyperparameters().items()}

        optimizer = torch.optim.LBFGS(
            params=filter(lambda p: p.requires_grad, self.parameters()), 
            lr=lr, 
            max_iter=maxiter,
            tolerance_change=tolerance_change,
            tolerance_grad=tolerance_grad,
            line_search_fn='strong_wolfe'
        )
        def closure():
            # print('iteration')
            loss = - self.theta_posterior_lobprob()
            optimizer.zero_grad()
            loss.backward()
            return loss

        # Backpropagation
        start_time = time.time()
        optimizer.step(closure)
        end_time = time.time()

        if disp:
            hp_MMAP = {k:p().detach().numpy().copy() for k,p in self.hyperparameters().items()}

            # with np.printoptions(precision=4, suppress=True, threshold=100, edgeitems=50, linewidth=100000):
            with np.printoptions(formatter={'all':lambda x: f'{x:.2f}' if x > 1e-2 else f'{x:.1e}'}, edgeitems=50, linewidth=100000):
                print(f'\n========  TPReg. train report ============')
                print(f'\tTime elapsed: {end_time-start_time:.1f} s')
                print(f'\tnbr iterations: {optimizer.state[optimizer._params[0]].get("n_iter")}')
                print(f'\tnbr fun evals:  {optimizer.state[optimizer._params[0]].get("func_evals")}')
                print()
                for k,v in hp_MMAP.items():
                    print(f'\t{k:30s} : {np.atleast_1d(hp[k])} -> {np.atleast_1d(hp_MMAP[k])}')
                print(f'======== EOF TPReg. train report ==========\n')

class GaussianProcessRegression(BayesianRegression):
    ''' GP Regression.
    
    Prior:
        p(alpha) = N(mu_a, Sigma_a)

    Likelihood:
        y(X) | alpha, sigma2 ~ N( F(X) @ alpha, sigma2 * I_N )
    
    Parameters:
        {alpha}
    '''
    def __init__( 
            self, 
            X:torch.Tensor, 
            y:torch.Tensor,
            mean_fun:torch.nn.Module = ZeroMean(), 
            kernel:torch.nn.Module = PolynomialKernel(variance_prior=0.5, degree=2), 
            train_sigma2:bool = True,
            sigma2_prior:float = 0.15,
            sigma2_hyperprior:torch.distributions.Distribution = Gamma(2.,15.),
    ):
        '''
        Args:
            - X: (N,Nx) tensor of inputs
            - y: (N,) tensor of targets
            - mean_fun: mean function
            - kernel: kernel function
            - train_sigma2: whether to train sigma2
            - sigma2_prior: prior of sigma2
            - sigma2_hyperprior: hyperprior of sigma2
        '''
        super().__init__(X=X, y=y, drop_duplicates=True)
        self.K = kernel
        # self.K = torch.compile(kernel)
        self.mu = mean_fun
        self.sigma2 = HyperParameter(
            tensor=torch.tensor(sigma2_prior), requires_grad=train_sigma2, constraint=Interval(1e-2,1e0),
            hyperprior = lambda self: sigma2_hyperprior
        )
        # self.set_training_data(X=X, y=y)

    @torch.no_grad()
    def eval(self):
        ''' Pre-store matrices that are kept constant during evaluation
        '''
        self.m_X = self.mu(self.X)
        self.K_XX = self.K(self.X,self.X)

        # auxiliary matrices needed for common matrix inversions
        self.C = torch.linalg.cholesky(self.K_XX + torch.eye(self.N) * self.sigma2(), upper=False)         # V = (K_XX + I*sigma2)^0.5  ->  V @ V.T = K_XX + I
        self.CM = torch.linalg.solve_triangular(self.C, (self.y - self.m_X)[:,None], upper=False)[:,0]     # (K_XX + I*sigma2)^0.5 \ (y - m_X)

        ''' Put network in evaluation mode '''
        super().eval()

    def predictive_prior(self, X_pred:torch.Tensor, diagonal:bool=False, include_noise:bool=False):
        ''' Calculates the predictive prior of y: p(y|theta)
        '''
        Ip = torch.eye(len(X_pred))
        mu_prior = self.mu(X_pred)
        Sigma_prior = self.K(X_pred, X_pred) + Ip * self.sigma2() * include_noise + 1e-4*Ip
        y_pred_prior = MultivariateNormal(
            loc = mu_prior, 
            covariance_matrix = Sigma_prior if not diagonal else torch.diag(torch.diag(Sigma_prior))
        )
        return y_pred_prior

    @torch.no_grad()
    def predictive_posterior(self, X_pred:torch.Tensor, diagonal:bool=False, include_noise:bool=True):
        ''' Calculates the predictive posterior of y: p(y_pred | y, theta)
        '''
        assert not self.training, 'the posterior can only be evaluated in `eval` mode'
        # X_pred = torch.as_tensor(X_pred)

        Ip = torch.eye(len(X_pred))
        K_XXp, K_XpXp = self.K(self.X,X_pred), self.K(X_pred,X_pred)
        m_Xp = self.mu(X_pred)

        CK = torch.linalg.solve_triangular(self.C, K_XXp, upper=False)      # (K_XX + I*sigma2)^0.5 \ K_XXp

        # CK.mT @ self.CM = K_XXp.T (self.K_XX + I*sigma2) \ (y - m_X )
        m_post = m_Xp + CK.mT @ self.CM
        # CK.mT @ CK = K_XXp.T (self.K_XX + I*sigma2) \ K_XXp
        Sigma_post = K_XpXp - CK.mT @ CK + Ip * self.sigma2() * include_noise + 1e-4*Ip

        y_pred_post = MultivariateNormal(
            loc         = m_post, 
            covariance_matrix = Sigma_post if not diagonal else torch.diag(torch.diag(Sigma_post))
        )
        return y_pred_post

    def theta_posterior_lobprob(self):
        ''' 
        Returns the log of the joint distribution: 
            `log p(theta,y) = log [ p(y|theta) * p(theta) ] = log p(y|theta) + log p(theta)`
        NOTE: this function is used to obtain the log_prob of the unnormalized posterior: 
            `p(theta|y) * p(y) = p(y|theta) * p(theta)`
        '''
        # log p(y|theta)
        y_logpdf = self.predictive_prior(X_pred=self.X, include_noise=True).log_prob(self.y)
        #  log p(theta)
        theta_logpdf = sum([p.hyperprior().log_prob(p()) for p in self.hyperparameters().values()])
        #  log p(y,theta)
        return y_logpdf + theta_logpdf

    def fit(self, *args, **kwargs):
        return self.MMAP(*args, **kwargs)

    def MMAP(self, lr:float=1.0, maxiter:int=300, tolerance_change:float=1e-9, tolerance_grad:float=1e-5, disp:bool=True) -> None:
        ''' Optimize hyperparameters according to MMAP (maximum marginal a posteriori) estimation:
            
            theta = argmax_{theta} log p( theta | y )

        Calculates the log of the marginal posterior of the hyper-prior parameters:
                p(theta|y) = p(y|theta) * log p(theta) / p(y)

            Applying the log, we have:
                logp(theta|y) = cnst + logp(y|theta) + logp(theta)
        '''
        assert self.training, 'MMAP can only be performed in train mode'

        # store hyper-parameters for logging purposes
        if disp:
            hp = {k:p().detach().numpy().copy() for k,p in self.hyperparameters().items()}

        optimizer = torch.optim.LBFGS(
            params=filter(lambda p: p.requires_grad, self.parameters()), 
            lr=lr, 
            max_iter=maxiter,
            tolerance_change=tolerance_change,
            tolerance_grad=tolerance_grad,
            line_search_fn='strong_wolfe'
        )
        def closure():
            # print('iteration')
            loss = - self.theta_posterior_lobprob()
            optimizer.zero_grad()
            loss.backward()
            return loss

        # Backpropagation
        start_time = time.time()
        optimizer.step(closure)
        end_time = time.time()

        if disp:
            hp_MMAP = {k:p().detach().numpy().copy() for k,p in self.hyperparameters().items()}

            # with np.printoptions(precision=4, suppress=True, threshold=100, edgeitems=50, linewidth=100000):
            with np.printoptions(formatter={'all':lambda x: f'{x:.2f}' if x > 1e-2 else f'{x:.1e}'}, edgeitems=50, linewidth=100000):
                print(f'\n========  GPReg. train report ============')
                print(f'\tTime elapsed: {end_time-start_time:.1f} s')
                print(f'\tnbr iterations: {optimizer.state[optimizer._params[0]].get("n_iter")}')
                print(f'\tnbr fun evals:  {optimizer.state[optimizer._params[0]].get("func_evals")}')
                print()
                for k,v in hp_MMAP.items():
                    print(f'\t{k:30s} : {np.atleast_1d(hp[k])} -> {np.atleast_1d(hp_MMAP[k])}')
                print(f'======== EOF GPReg. train report ==========\n')

class LinearRegression:
    def __init__(self, X:np.ndarray, y:np.ndarray, features:IFeatures):
        self.features = features
        self._set_training_data(X=X, y=y)
        
    def _set_training_data(self, X:np.ndarray, y:np.ndarray):
        ''' Set training data and recalculate training features
        '''
        assert len(X) == len(y), f'len(X)={len(X)} should be equal to len(y)={len(y)}'
        assert np.ndim(y) == 1 and np.ndim(X) == 2
        # remove invalid targets
        idx_valid = np.logical_not( np.logical_or( np.isnan(y), np.isinf(y) ) )
        X, y = X[idx_valid], y[idx_valid]
        # remove duplicate inputs
        X, x_idx  = np.unique(X, axis=0, return_index=True)
        y = y[x_idx]
        # store
        self.y = y
        self.X = X
        self.F = self.features(X=X)
        self.N, self.Nf = self.F.shape
        
    def MLE_alpha(self, reg:float=1e-8):
        '''
        Args:
            - reg: penalization term used to mitigate the instability of inverting X.T @ X
        Returns:
            - alpha_mle: MLE of the alpha parameters = (X'*X + regParam*I)\(X'*y)

        TODO: replace reg by lambda, where lambda actually means the precision of alpha in the Bayesian framework
        '''
        return solve(self.F.T @ self.F + reg*np.eye(self.Nf), self.F.T @ self.y)

    def MLE_y_pred(self, X_pred:np.ndarray, reg:float=1e-8):
        F_pred = self.features(X=X_pred)
        # compute prediction mean
        y_MLE = F_pred @ self.MLE_alpha(reg=reg)
        return y_MLE
    
    def y_mean(self, X:np.ndarray, alpha_mean:np.ndarray):
        ''' Evaluates the linear model F(X) @ alpha_mean
        Args:
                - X_pred: <n,self.nVars>
                - alpha_mean: <na,>
        Returns:
                - y_mean: <n>
        '''
        # generate x_all (all basis vectors) based on model order)
        F = self.features(X=X)
        # compute prediction mean
        y_mean  = F @ alpha_mean
        return y_mean

    def score(self, X, y_true, alpha_mean):
        ''' Return the coefficient of determination R^2 of the prediction.
        Args:
                - X: <n,self.nVars>
                - y: <n,1> true values at X
        '''
        y_pred_mean = self.y_mean(X=X, alpha_mean=alpha_mean)
        u = ((y_true - y_pred_mean) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        r2 = (1 - u/v)
        return r2


# from CBOSS.bayesian_models.bhs import bhs

# class HorseShoeBayesianLinearRegression(LinearRegression):
#     ''' Bayesian linear regression method heavy-tailed horseshoe prior (sparsity inducing prior).
#     Posterior is not given analitically but can be sampled with a Gibbs sampler.

#     References: Makalic and Schmidt (2016). A simple sampler for the horseshoe estimator.
#     '''

#     def prior_sample_alpha(self, nsamples:int):
#         raise NotImplementedError

#     def posterior_sample_alpha(self, nsamples:int=int(1e3), burnin:int=100, thin:int=1):
#         ''' Sample an alpha vector from the posterior distribution of alpha conditioned on the dataset
#         This function saves the dataset to self.X and self.y

#         Returns:
#                 - alpha: <nCoefffs>  alpha vector sampled from the conditional posterior distribution
#         '''
#         F = self.F
#         y = self.y

#         # remove bias feature if included --> the bhs function estimates the bias term separately
#         if np.all(F[:,0] == 1):
#             F = F[:,1:]
            
#         N, nCoeffs = F.shape
        
#         # remove columns of zeros in F
#         check_zero = np.all(F == np.zeros((N, 1)), axis=0)
#         idx_zero   = np.where(check_zero == True)[0]
#         idx_nnzero = np.where(check_zero == False)[0]
#         F = F[:, idx_nnzero]

#         # call Gibbs sampler for nGibbs steps until it works
#         while(True):
#             # re-run if there is an error during sampling
#             try:
#                 alpha, alpha_0, _, _, _ = bhs(Xorg=F, yorg=y, nsamples=nsamples, burnin=burnin, thin=thin)
#             except Exception as e:
#                 print(f'Error during Gibbs sampling. Trying again.\n{e}', file=sys.stderr)
#                 continue
#             # run until alpha matrix does not contain any NaNs
#             if not np.isnan(alpha).any():
#                 break
#             else:
#                 print(f'Found NaNs in Gibbs sampling. Trying again.\n{e}', file=sys.stderr)
#                 continue

#         alpha_pad = np.zeros((nsamples, nCoeffs+1))
#         # set bias term
#         alpha_pad[:,0] = alpha_0
#         # set other tems
#         alpha_pad[:,idx_nnzero+1] = alpha.T
#         return alpha_pad

