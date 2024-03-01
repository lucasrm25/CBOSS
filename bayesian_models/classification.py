'''
Author: Lucas Rath
'''

import os, sys
import numpy as np
import warnings
import time
import torch
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch.optim.lbfgs import _strong_wolfe
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from CBOSS.utils.torch_utils import *
from CBOSS.bayesian_models.means import ZeroMean
from CBOSS.bayesian_models.kernels import PolynomialKernel


'''
Classification
=============================================================
'''

class BayesianClassification(ModuleX):
    
    @torch.no_grad()
    def set_training_data(self, X:torch.Tensor, y:torch.Tensor):
        ''' Set training data and recalculate training features
        '''
        assert isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)
        assert len(X) == len(y), f'len(X)={len(X)} should be equal to len(y)={len(y)}'
        assert y.dim() == 1 and X.dim() == 2
        assert self.training
        
        assert torch.all((y == 0) | (y == 1)), 'y array must contain only 0 or 1'
        y = torch.clone(y)
        y[y==0] = -1    # NOTE: the Bayesian classification method requires y \in [-1,1], which is not so untuitive as [0,1]
        
        # remove invalid targets
        idx_valid = ~ (torch.isnan(y) | torch.isinf(y))
        X, y = X[idx_valid], y[idx_valid]
        # remove duplicate inputs
        idx_unique = np.unique(X, axis=0, return_index=True)[1]
        X = X[idx_unique]
        y = y[idx_unique]
        # store
        self.register_buffer('y', torch.as_tensor(y,dtype=self.dtype))
        self.register_buffer('X', torch.as_tensor(X,dtype=self.dtype))
        self.N, self.nx = self.X.shape
    
    def forward(self, X_pred:torch.Tensor, **kwargs):
        if self.training:
            raise NotImplementedError
        else:
            return self.predictive_y_posterior(X_pred, **kwargs)

@torch.jit.script
def _laplace_newton_step(f_hat, y, m_X, K_XX):
    pi = torch.sigmoid(f_hat)
    # Eq: 3.15: nabla_{f_x} log p(y | f_x)
    r = (y + 1)/2 - pi
    # Line 4: W = - nabla nabla^T log p(y | f_x)
    W = torch.diag(pi * (1 - pi))
    # Line 5
    Wcho = W**0.5      # W**0.5 - torch.linalg.cholesky(W) --> matrix is diagonal
    Wcho_K_XX = Wcho @ K_XX
    B = torch.eye(len(f_hat)) + Wcho_K_XX @ Wcho
    L = torch.linalg.cholesky( B, upper=False)
    ''' Only works for zero mean function
    # Line 6
    b = W @ f_hat + r
    # Line 7
    a = b - Wcho @ torch.cholesky_solve((Wcho_K_XX @ b)[:,None], L, upper=False)[:,0]
    f_hat = K_XX @ a
    '''
    # Lines 6 and 7 -> adjusted for possibly nonzero mean function 
    e = f_hat - m_X
    Kiv_e = torch.linalg.solve(K_XX, e)
    # g = nabla log p(f_x | y)
    g = r - Kiv_e   # gradient
    V = torch.linalg.solve_triangular( L, Wcho_K_XX, upper=False)
    # H = ( nabla nabla^T log p(f_x | y) )^-1
    H = V.mT @ V - K_XX  # Hessian inverse

    # Line 10: Compute log marginal likelihood in loop and use as convergence criterion
    log_p_y_fx = torch.nn.functional.logsigmoid(y * f_hat).sum()
    e_Kiv_e = torch.inner(e, Kiv_e)
    # phi is the objective function to be MAXIMIZED = log p(f_x|y) + cst.
    phi = log_p_y_fx - 0.5 * e_Kiv_e
    # mll = log p(y)
    mll = (
        - 0.5 * e_Kiv_e
        + log_p_y_fx
        - torch.log(torch.diag(L)).sum()        # == -0.5 log det(B)
    )

    return phi, g, H, r, W, Wcho, B, L, e, Kiv_e, e_Kiv_e, log_p_y_fx, mll

class LaplaceGaussianProcessClassification(BayesianClassification):
    ''' Hierarchical Bayesian linear regression method with normal-inverse-gamma prior distribution on P(alpha,sigma^2).

    Prior:
        p(f) = GP(mu, K)

    Likelihood:
        y | f  ~ PI_i Bernoulli( y_i ; p = sigmoid(f_i) )

    References: 
        - Gaussian process for machine learning - Rasmussen and Williams 2006. Ch 3.3, pg 39
    '''
    def __init__(self, 
        X:torch.Tensor, 
        y:torch.Tensor, 
        mean_fun:torch.nn.Module = ZeroMean(), 
        kernel:torch.nn.Module = PolynomialKernel(variance_prior=0.5, degree=2), 
        f_hat_init = None,
        ):
        ''' 
            X: <N,nx> array
            y: <N> array in [0,1]
            f_hat_init: <Nf> allows the user to guess the posterior mode location -> might be taken from previous optimizations 
        '''
        super().__init__()

        self.set_training_data(X=X, y=y)
    
        self.K = kernel
        self.mu = mean_fun

        # initial guess for the posterior mean E[f_x|y]
        self.f_hat = torch.nn.Parameter(
            data = f_hat_init if f_hat_init is not None else self.y * 0.1,
            requires_grad=False
        )
        self._ghquad = GaussHermiteQuadrature1D()

    @torch.no_grad()
    def eval(self):
        ''' Pre-store matrices that are kept constant during evaluation
        '''
        self.m_X = self.mu(self.X)
        self.K_XX = self.K(self.X,self.X)

        # auxiliary matrices needed for common matrix inversions
        self.C = torch.linalg.solve( self.K_XX, self.f_hat - self.m_X )

        ''' Put network in evaluation mode '''
        super().eval()

    def _posterior_mode(self, m_X, K_XX, maxiter:int=500, lr=1.0, tolerance_change:float=1e-6, tolerance_grad:float=1e-6, line_search:bool=True):
        ''' Mode-finding of binary Laplace GPC based on Newton-Raphson method

        Reference: 
            - Rasmussen and Williams 2006, Algorithm 3.1

        Goal is to find the posterior mode, i.e.:
            f_hat = argmax log p(f_x|y) = argmin - log p(f_x|y)
                or
            nabla log p(f_x=f_hat|y) = 0
        
        where 
            p(f_x | y) = p(y | f_x) * p(f_x) * p(y)
        '''

        if not line_search:
            optimizer = torch.optim.Adagrad(params = [self.f_hat], lr=lr)

        has_converged = False
        mll_old = -np.inf
        niter = 0
        while niter < maxiter and not has_converged:

            phi, g, H, r, W, Wcho, B, L, e, Kiv_e, e_Kiv_e, log_p_y_fx, mll = _laplace_newton_step(f_hat=self.f_hat, y=self.y, m_X=m_X, K_XX=K_XX)
            delta = H @ g
            niter += 1
            # convergence criterion: mll has not decreased
            mll_step = mll - mll_old
            mll_old = mll
            
            if not line_search:
                ''' Newton Step
                '''
                optimizer.zero_grad()
                self.f_hat.grad = delta
                optimizer.step()            # f_hat = f_hat - lr * delta
                # with torch.no_grad():
                #     self.f_hat.copy_(self.f_hat - lr * delta )
            else:

                ''' Strong-Wolfe linear search: https://en.wikipedia.org/wiki/Wolfe_conditions
                '''
                def obj_func(f_hat, lr, d):
                    phi, g, H, r, W, Wcho, B, L, e, Kiv_e, e_Kiv_e, log_p_y_fx, mll = _laplace_newton_step(f_hat=f_hat - lr * d, y=self.y, m_X=m_X, K_XX=K_XX)
                    return -phi, H @ g

                phi_new, g, lr, ls_func_evals = _strong_wolfe(
                    obj_func = obj_func,
                    x = self.f_hat, 
                    t = lr,         # learning rate = step length
                    d = delta,      # (- H * nabla_x f(x))^T
                    f = -phi,       # f(x) = - log_p_fx_y
                    g = g,          # nabla_x f(x) = g
                    gtd = torch.inner(delta, g),      # d . nabla_x f(x)
                    max_ls = max(300, maxiter-niter-1)
                )
                # print(f'evaluated : {ls_func_evals}')
                niter += ls_func_evals

                with torch.no_grad():
                    self.f_hat.copy_(self.f_hat - lr * delta)

            # iter_arr        += [niter]
            # mll_step_arr    += [mll_step.cpu().numpy().tolist()]
            # delta_norm      += [torch.linalg.norm(delta).cpu().numpy().tolist()]
            # g_norm          += [torch.linalg.norm(g).cpu().numpy().tolist()]

            if g.abs().max() < tolerance_grad:
                has_converged = True
                break
            # lack of progress
            if delta.mul(lr).abs().max() < tolerance_change:
                has_converged = True
                break

            # # if mll_step < 1e-6:
            # if torch.linalg.norm(delta) < eps or torch.linalg.norm(g) < eps:
            #     has_converged = True
            #     break

        # self.f_hat = f
        self.r = r
        self.L = L
        self.Wcho = Wcho
        self.Sigma = -H

        return self.f_hat, mll, (has_converged, niter, torch.linalg.norm(delta), torch.linalg.norm(g))

    # @torch.no_grad()
    def fit(self, maxiter:int=1, maxevals_per_iter:int=int(1e3), lr:float=1.0, tolerance_change:float=1e-5, tolerance_grad:float=1e-5, disp:bool=True):
        ''' Fit model by interleavedly optimizing hyper-parameters (maximizing mll) and 
        finding the posterior mode based on the Laplace approximation

        NOTE:
            f_hat is updated using Newton-Raphson, while the hyper-parameters are updated by
            maximizing the marginal likelihood (Rasmussen - equation 5.20).
            The issue is that the MLE depends on f_hat and h_hat depends on f_hat, so they have to
            be optimized together.

        Based ono Rasmussen - Algorithm 5.1 - pg 126
        '''
        assert self.training, 'MMAP can only be performed in train mode'
        assert maxiter >= 1

        # store hyper-parameters for logging purposes
        if disp:
            hp = {k:p().detach().numpy().copy() for k,p in self.hyperparameters().items()}

        optimizer = torch.optim.LBFGS(
            params = filter(lambda p: p.requires_grad, self.parameters()), # == theta       # [self.f_hat], 
            lr=lr, 
            max_iter=maxiter-1,
            tolerance_change = tolerance_change,
            tolerance_grad = tolerance_grad,
            line_search_fn='strong_wolfe'
        )
        optimizer.state['nevals'] = 0

        def nmll_theta():
            ''' Returns the Laplace approximation for `- log p(y | theta)`
            Equation 3.32
            '''
            m_X = self.mu(self.X)
            K_XX = self.K(self.X,self.X)
            with torch.no_grad():
                # self.f_hat.copy_(self.f_hat * 0)
                _, _, (has_converged, nevals, residual_delta, residual_g) = self._posterior_mode(
                    m_X=m_X, K_XX=K_XX,
                    maxiter=maxevals_per_iter,
                    tolerance_change = tolerance_change,
                    tolerance_grad = tolerance_grad,
                    line_search=True
                )
                optimizer.state['has_converged'] = has_converged
                optimizer.state['residual_delta'] = residual_delta
                optimizer.state['residual_g'] = residual_g
                optimizer.state['nevals'] += nevals

            # calculate gradients for theta. Note that K_XX(theta)
            if has_converged:
                phi, g, H, r, W, Wcho, B, L, e, Kiv_e, e_Kiv_e, log_p_y_fx, mll = _laplace_newton_step(f_hat=self.f_hat, y=self.y, m_X=m_X, K_XX=K_XX)
                nmll = - mll
            else:
                nmll = torch.tensor(0., requires_grad=True)

            optimizer.zero_grad()
            nmll.backward()
            return nmll

        # Backpropagation
        start_time = time.time()
        optimizer.step(nmll_theta)
        end_time = time.time()

        # adjust posterior mode for the last time
        with torch.no_grad():
            m_X = self.mu(self.X)
            K_XX = self.K(self.X,self.X)
            _, _, (has_converged, nevals, residual_delta, residual_g) = self._posterior_mode(
                m_X=m_X, K_XX=K_XX,
                maxiter=maxevals_per_iter,
                tolerance_change = tolerance_change,
                tolerance_grad = tolerance_grad,
                line_search=True
            )

        if disp:
            hp_MMAP = {k:p().detach().numpy().copy() for k,p in self.hyperparameters().items()}
            if not optimizer.state['has_converged']:
                warnings.warn(f'{__class__.__name__}.fit() did not converge')
            with np.printoptions(formatter={'all':lambda x: f'{x:.2f}' if x > 1e-2 else f'{x:.1e}'}, edgeitems=50, linewidth=100000):
                print(f'\n======== GPClass. train report ===========')
                print(f'\tTime elapsed: {end_time-start_time:.1f} s')
                # print(f'\tnbr iterations: {optimizer.state[optimizer._params[0]].get("n_iter")}')
                # print(f'\tnbr fun evals:  {optimizer.state[optimizer._params[0]].get("func_evals")}')
                print(f'\tnbr evals:      {optimizer.state["nevals"]}')
                print(f'\tresidual_delta: {optimizer.state["residual_delta"]:.1e}')
                print(f'\tresidual_g:     {optimizer.state["residual_g"]:.1e}')
                print()
                for k,v in hp_MMAP.items():
                    print(f'\t{k:30s} : {np.atleast_1d(hp[k])} -> {np.atleast_1d(hp_MMAP[k])}')
                print(f'==== ENDOF GPClass. train report ===========\n')
        return optimizer.state

    @torch.no_grad()
    def predictive_y_posterior(self, X_pred:torch.Tensor):
        ''' Calculates the predictive probability distribution: p( y_pred | y )

            TODO: use Gauss-Hermite quadrature to approximate the integral
        '''
        assert not self.training
        # X_pred = torch.as_tensor(X_pred)

        m_Xp = self.mu(X_pred)
        K_XXp, K_XpXp = self.K(self.X,X_pred), self.K(X_pred,X_pred)
        K_XpX = K_XXp.mT

        # NOTE: 0 = nabla p(f_X | y) = r - K \ (f - m_X)  ->  K \ (f - m_X) = r
        # E_fXp_y = m_Xp + K_XpX @ self.r                  # E[ f_X | y ] 

        # A = torch.linalg.solve(self.K_XX, K_XpX.T).T
        # E_fXp_y   = m_Xp   + A @ (self.f_hat - self.m_X )
        E_fXp_y = m_Xp + K_XpX @ self.C
        # Var_fXp_y = K_XpXp + A @ (self.Sigma - self.K_XX) @ A.T + 1e-4*torch.eye(len(X_pred))
        V = torch.linalg.solve_triangular( self.L, self.Wcho @ K_XpX.mT, upper=False)
        Var_fXp_y = K_XpXp - V.mT @ V + 1e-6*torch.eye(len(X_pred))
        
        p = self._ghquad(
            func = lambda f: torch.sigmoid(f),#torch.exp(Bernoulli( torch.sigmoid(f)).log_prob(torch.tensor(1.))), # == torch.sigmoid(f)
            gaussian_dists = MultivariateNormal(
                loc = E_fXp_y, 
                covariance_matrix = Var_fXp_y
            )
        )
        p = torch.clamp(p, min=1e-8, max=1-1e-8)
        
        p = torch.sigmoid( E_fXp_y )

        return Bernoulli(probs=p)

