'''
Author: Lucas Rath

Implements the Sparse Variational Gaussian Process (SVGP) model,
which can be used for both regression and classification.
'''

import numpy as np
from pathlib import Path
import torch
import time
from torch.distributions.gamma import Gamma
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from CBOSS.bayesian_models.means import ZeroMean
from CBOSS.bayesian_models.kernels import RBFKernel
from CBOSS.utils.torch_utils import *


class IGP(ModuleX):
    def __new__(cls, X:torch.Tensor, y:torch.Tensor, *args, **kwargs) -> 'IGP':
        return super().__new__(cls)
    
    def __init__(self, X:torch.Tensor, y:torch.Tensor,s:torch.Tensor, drop_duplicates:bool=True) -> None:
        super().__init__()
        self.set_training_data(X=X, y=y, s=s, drop_duplicates=drop_duplicates)
        
    @torch.no_grad()
    def set_training_data(self, X:torch.Tensor, y:torch.Tensor, s:torch.Tensor, drop_duplicates:bool=True):
        ''' Set training data and recalculate training features
        '''
        assert isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor) and isinstance(s, torch.Tensor)
        assert len(X) == len(y), f'len(X)={len(X)} should be equal to len(y)={len(y)}'
        assert y.dim() == 1 and s.dim() == 1 and X.dim() == 2
        assert s.dtype in [torch.int32, torch.int64] and all(s >= 0) and all(s < len(X))
        assert self.training
        # remove invalid targets
        idx_valid = ~ (torch.isnan(y) | torch.isinf(y))
        X, y = X[idx_valid], y[idx_valid]
        if drop_duplicates:
            # remove duplicate inputs
            idx_unique = np.unique(X, axis=0, return_index=True)[1]
            X = X[idx_unique]
            y = y[idx_unique]
            s = torch.as_tensor(np.intersect1d(s.numpy(), idx_unique))
        # store
        self.register_buffer('y', torch.as_tensor(y,dtype=self.dtype))
        self.register_buffer('s', torch.as_tensor(s,dtype=self.dtype))
        self.register_buffer('X', torch.as_tensor(X,dtype=self.dtype))
        self.N = len(self.X)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, X_pred:torch.Tensor, include_noise:bool=True):
        ''' Calculates the predictive posterior of y: p( y_pred(X_pred) | dataset )
        '''
        if self.training:
            return self.predictive_prior(X_pred=X_pred, include_noise=include_noise)
        else:
            return self.predictive_posterior(X_pred=X_pred, include_noise=include_noise)
        
    def predictive_prior(self, X_pred:torch.Tensor, include_noise:bool=True):
        ''' Calculates the predictive posterior of y: p( y_pred(X_pred) | dataset )
        '''
        if include_noise:
            return self.predictive_y_prior(X_pred=X_pred)
        else:
            return self.predictive_f_prior(X_pred=X_pred)
        
    def predictive_posterior(self, X_pred:torch.Tensor, include_noise:bool=True):
        ''' Calculates the predictive posterior of y: p( y_pred(X_pred) | dataset )
        '''
        if include_noise:
            return self.predictive_y_posterior(X_pred=X_pred)
        else:
            return self.predictive_f_posterior(X_pred=X_pred)

def minibatch(loss_func):
    ''' Decorator to use minibatching for a loss function (e.g. SVGP)
    Source: https://github.com/cics-nd/gptorch/blob/master/gptorch/models/sparse_gpr.py
    '''

    def wrapped(obj, X=None, y=None, *args, **kwargs):

        # Get from model:
        if obj.batch_size is not None:
            i = np.random.permutation(len(obj.X))[:obj.batch_size]
            X, y = obj.X[i], obj.y[i]
        else:
            X, y = obj.X, obj.Y

        return loss_func(obj, X, y, *args, **kwargs)

    return wrapped

def add_jitter(K:torch.Tensor, jitter:float=1e-6):
    ''' Add jitter to a kernel matrix
    '''
    return K + torch.eye(len(K), dtype=K.dtype) * jitter

class Likelihood(ModuleX):
    def __init__(self):
        super().__init__()
        self._ghquad = GaussHermiteQuadrature1D()
        
    def forward(self, f:torch.Tensor):
        ''' Returns the likelihood log p( y | f )
        '''
        raise NotImplementedError

    def propagate(self, pf:MultivariateNormal):
        ''' Calculates p(y* | y) = \int p(y* | f*) p(f* | y) df*
        '''
        raise NotImplementedError

    def ghquad(self, gaussian_dist:MultivariateNormal, y:torch.Tensor):
        ''' Calculates
            \int log p(y | f) q(f) df
        using Gaussian-Hermite Quadrature, where q(f) = gaussian_dist
        '''
        raise NotImplementedError

class GaussianLikelihood(Likelihood):
    def __init__(
        self, 
        train_sigma2:bool = True,
        sigma2_prior:float = 0.1,
        sigma2_hyperprior:torch.distributions.Distribution = Gamma(2.,15.),
    ) -> None:
        super().__init__()
        self.sigma2 = HyperParameter(
            tensor=torch.tensor(sigma2_prior), requires_grad=train_sigma2, constraint=GreaterThan(1e-8),
            hyperprior = lambda self: sigma2_hyperprior
        )

    def forward(self, f:torch.Tensor):
        ''' Returns the likelihood log p( y | f )
        '''
        return MultivariateNormal(loc=f, covariance_matrix=self.sigma2()*torch.eye(len(f)))

    def propagate(self, pf:MultivariateNormal):
        ''' Calculates p(y* | y) = \int p(y* | f*) p(f* | y) df*
        '''
        return MultivariateNormal(
            loc=pf.loc, covariance_matrix = add_jitter(pf.covariance_matrix, self.sigma2())
        )

    def ghquad(self, gaussian_dist:MultivariateNormal, y:torch.Tensor):
        ''' Calculates
            \int log p(y | f) q(f) df
        using Gaussian-Hermite Quadrature, where q(f) = gaussian_dist
        '''
        return self._ghquad(
            func = lambda f: self(f=y).log_prob(f),  # NOTE: p(y|f) = p(f|y)
            gaussian_dists = gaussian_dist 
        )
        # self._ghquad(
        #     func = lambda f: torch.tensor([
        #         self(y=fi).log_prob(y)
        #         for fi in f
        #     ]),
        #     gaussian_dists = gaussian_dist
        # ).sum()
        # torch.stack([
        #     self._ghquad(
        #         func = lambda f: torch.stack([self(y=torch.tensor([fi])).log_prob(yi) for fi in f[:,None]]),
        #         gaussian_dists = Normal(loc=qf_mui, scale=qf_vari.sqrt()) 
        #     )
        #     for qf_mui, qf_vari, yi in zip(gaussian_dist.loc, gaussian_dist.covariance_matrix.diagonal(), y)
        # ]).sum()

class BernoulliSigmoidLikelihood(Likelihood):
    ''' Represents:
        p(y | f) = Bernoulli( y ; sigmoid(f) )
    '''
    def __init__(self):
        super().__init__()
    
    def forward(self, f:torch.Tensor):
        ''' Returns the likelihood log p( y | f )
        '''
        return Bernoulli( torch.sigmoid(f) )

    def propagate(self, pf:MultivariateNormal):
        ''' Calculates p(y* | y) = \int p(y* | f*) p(f* | y) df*
        '''
        p = self._ghquad(
            func = lambda f: torch.exp(self(f=f).log_prob(torch.tensor(1.))), # == torch.sigmoid(f)
            gaussian_dists = pf
        )
        p = torch.clamp(p, min=1e-8, max=1-1e-8)
        return Bernoulli(probs=p)

    def ghquad(self, gaussian_dist:MultivariateNormal, y:torch.Tensor):
        ''' Calculates
            \int log p(y | f) q(f) df
        using Gaussian-Hermite Quadrature, where q(f) = gaussian_dist
        '''
        return self._ghquad(
            # func = lambda f: self(f=f).log_prob(y).sum(1),
            func = lambda f: self(f=f).log_prob((y==1).to(y.dtype)).sum(1),
            gaussian_dists = gaussian_dist
        ).sum()
        # torch.stack([
        #     self._ghquad(
        #         func = lambda f: torch.stack([self(y=torch.tensor([fi])).log_prob(yi) for fi in f[:,None]]),
        #         gaussian_dists = Normal(loc=qf_mui, scale=qf_vari.sqrt()) 
        #     )
        #     for qf_mui, qf_vari, yi in zip(gaussian_dist.loc, gaussian_dist.covariance_matrix.diagonal(), y)
        # ]).sum()
        # self._ghquad(
        #     func = lambda f: torch.stack([
        #         self(y=fi).log_prob(y)
        #         for fi in f
        #     ]),
        #     gaussian_dists = gaussian_dist
        # ).sum()

class SVGP(IGP):
    '''
    References:
        - https://github.com/GPflow/GPflow/blob/develop/gpflow/models/svgp.py
        - https://github.com/cics-nd/gptorch/blob/master/gptorch/models/sparse_gpr.py
    '''
    def __init__(
        self, 
        X:torch.Tensor, 
        y:torch.Tensor,
        s:torch.Tensor,
        mean_fun:torch.nn.Module = ZeroMean(), 
        kernel:torch.nn.Module = RBFKernel(length_scales=0.2, variance_prior=0.5),
        likelihood:Likelihood = GaussianLikelihood(),
        batch_size:int = None,
        optimizer:str = ['adam', 'lbfgs'][0],
        drop_duplicates:bool = True,
    ) -> None:
        super().__init__(X=X, y=y, s=s, drop_duplicates=drop_duplicates)
        self.optimizer = optimizer
        self.register_buffer('s', torch.as_tensor(s)) # ,dtype=torch.bool
        self.register_buffer('Xs', torch.as_tensor(self.X[s]))
        self.likelihood = likelihood
        self.K = kernel
        self.mu = mean_fun
        self.batch_size = batch_size
        self.Ns        = len(self.s)
        self.mu_u      = torch.nn.Parameter(torch.zeros(self.Ns), requires_grad=True)
        self.sigma2_u  = ConstrainedParameter(torch.ones(self.Ns), requires_grad=True, constraint=GreaterThan(1e-4))
        self.ghquad    = GaussHermiteQuadrature1D()
    
    @torch.no_grad()
    def eval(self):
        ''' Pre-store matrices that are kept constant during evaluation
        '''
        I = torch.eye(self.N).to(self.dtype)
        self.m_X  = self.mu(self.X)
        self.m_S  = self.m_X[self.s]
        self.K_XX = add_jitter(self.K(self.X,self.X))
        self.K_SS = add_jitter(self.K_XX[self.s][:,self.s])
        self.Sigma_u = torch.diag(self.sigma2_u())
        self.L = torch.linalg.cholesky(self.K_SS)
        # Put network in evaluation mode
        super().eval()
    
    @minibatch
    def elbo(self, X:torch.Tensor, y:torch.Tensor):
        
        Xs   = self.Xs
        I    = torch.eye(self.N).to(self.dtype)   
        m_X  = self.mu(X)
        m_S  = self.mu(Xs) # m_X[self.s]
        K_XX = add_jitter(self.K(X, X))
        K_SS = add_jitter(self.K(Xs, Xs)) # K_XX[self.s][:,self.s]
        K_XS = self.K(X, Xs) # K_XX[:,self.s]
        L = torch.linalg.cholesky(K_SS)  # K_SS = L @ L^T
        Sigma_u = torch.diag(self.sigma2_u())
        
        KL = kl_divergence(
            MultivariateNormal(loc=self.mu_u, covariance_matrix=Sigma_u), 
            MultivariateNormal(loc=torch.zeros(self.Ns), covariance_matrix=torch.eye(self.Ns))
        )

        A = torch.linalg.solve(K_SS, K_XS.T).T
        
        qfS_mu  = L @ self.mu_u
        qfS_cov = L @ Sigma_u @ L.T
        
        # Gaussian condition rule:  q(f_S)=N(L @ mu_u, L @ Sigma_u @ L^T),   q(f | f_S) = N( m_X + A @ (f_S - m_S), K_XX - A @ K_SS @ A^T )
        qf_mu  = m_X  + A @ (qfS_mu  - m_S )
        qf_cov = K_XX + A @ (qfS_cov - K_SS) @ A.T
        
        # variational distribution q(f) = p(f | f_S) q(f_S)
        qf = MultivariateNormal(
            loc = qf_mu,
            covariance_matrix = qf_cov
        )
        
        # \int log p(y | f) q(f) df
        E_log_likelihood = self.likelihood.ghquad( gaussian_dist=qf, y=y )
        E_log_likelihood *= self.N / len(X)
        
        ELBO = E_log_likelihood - KL
        return ELBO
        
    def fit(self, lr:float=1.0, maxiter:int=500, tolerance_change:float=1e-9, tolerance_grad:float=1e-7, disp:bool=True):
        ''' Maximize ELBO '''
        if disp:
            hp = {k:p().detach().numpy().copy() for k,p in self.hyperparameters().items()}
        
        def closure(optimizer):
            loss = - self.elbo(X=self.X, y=self.y)
            # loss = torch.log(loss)
            optimizer.zero_grad()
            loss.backward()
            # log
            if isinstance(optimizer.state['nevals'],int):
                optimizer.state['nevals'] += 1
            else:
                optimizer.state['nevals'] = 1
            optimizer.state['elbo'] = - loss.item()
            return loss

        if self.optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS(
                params=filter(lambda p: p.requires_grad, self.parameters()), 
                lr=lr, 
                max_iter=maxiter,
                tolerance_change=tolerance_change,
                tolerance_grad=tolerance_grad,
                line_search_fn='strong_wolfe'
            )
            # Backpropagation
            start_time = time.time()
            optimizer.step(lambda: closure(optimizer))
            end_time = time.time()
            
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, self.parameters()), 
                lr=lr,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=20, verbose=True)
            
            start_time = time.time()
            for it in range(maxiter):
                loss = closure(optimizer)
                optimizer.step()
                scheduler.step(loss)
                
                # break if learning rate drops below 1e-6
                if optimizer.param_groups[0]['lr'] < 1e-6:
                    break
            end_time = time.time()
        else:
            raise NotImplementedError
        
        if disp:
            hp_fit = {k:p().detach().numpy().copy() for k,p in self.hyperparameters().items()}
            with np.printoptions(formatter={'all':lambda x: f'{x:.2f}' if x > 1e-2 else f'{x:.1e}'}, edgeitems=50, linewidth=100000):
                print(f'\n========  SVGP train report ============')
                print(f'\tTime elapsed:   {end_time-start_time:.1f} s')
                print(f'\tnbr evals:      {optimizer.state["nevals"]}')
                print(f'\telbo:           {optimizer.state["elbo"]:.1e}')
                print(f'\t')
                for k in hp_fit.keys():
                    print(f'\t{k:30s} : {np.atleast_1d(hp[k])} -> {np.atleast_1d(hp_fit[k])}')
                print(f'======== EOF SVGP train report ==========\n')

    def predictive_f_prior(self, X_pred:torch.Tensor):
        Ip = torch.eye(len(X_pred))
        return MultivariateNormal(
            loc = self.mu(X_pred), 
            covariance_matrix = self.K(X_pred, X_pred) + 1e-4*Ip
        )
    
    def predictive_y_prior(self, X_pred:torch.Tensor):
        ''' Calculates the predictive posterior of y: p( y(X_pred) | y )
        '''
        f_pred = self.predictive_f_prior(X_pred=X_pred)
        y_pred = self.likelihood.propagate(pf=f_pred)
        return y_pred

    @torch.no_grad()
    def predictive_f_posterior(self, X_pred:torch.Tensor):
        ''' Calculates the predictive posterior of y: p( f(X_pred) | y )
        '''
        assert not self.training, 'the posterior can only be evaluated in `eval` mode'
        
        m_Xp   = self.mu(X_pred)
        m_S    = self.m_S
        K_XpXp = self.K(X_pred, X_pred)
        K_XpS  = self.K(X_pred, self.Xs)
        K_SS   = self.K_SS
        
        A = torch.linalg.solve(self.K_SS, K_XpS.T).T    # C = K_XpS @ K_SS^-1
        Ip = torch.eye(len(X_pred))
        Sigma_u = torch.diag(self.sigma2_u())
        
        qfS_mu  = self.L @ self.mu_u
        qfS_cov = self.L @ Sigma_u @ self.L.T
        
        mu_post    = m_Xp   + A @ (qfS_mu  - m_S )
        Sigma_post = K_XpXp + A @ (qfS_cov - K_SS) @ A.T + 1e-4*Ip

        return MultivariateNormal(
            loc = mu_post, 
            covariance_matrix = Sigma_post
        )

    @torch.no_grad()
    def predictive_y_posterior(self, X_pred:torch.Tensor):
        ''' Calculates the predictive posterior of y: p( y(X_pred) | y )
        '''
        f_pred = self.predictive_f_posterior(X_pred=X_pred)
        y_pred = self.likelihood.propagate(pf=f_pred)
        return y_pred