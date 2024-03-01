''' 
Author: Lucas Rath

Implements a Gibs sampler for Bayesian linear regression with horseshoe prior (BHS-LR).
The method `bhs` samples from the posterior distribution of the parameters alpha:
	f(x) = X @ alpha

Adapted from "Ricardo Baptista and Matthias Poloczek - 2018, Bayesian Optimization and Combinatorial Structures"

References: 
- Makalic and Schmidt 2016, A simple sampler for the horseshoe estimator
- Baptista and Poloczek 2018, Bayesian Optimization and Combinatorial Structures
'''

import numpy as np
import scipy as sp
from numba import njit


''' Numba help functions
=====================' '''

@njit(fastmath=True, cache=True)
def np_apply_along_axis(func1d, axis, arr):
  # https://github.com/numba/numba/issues/1269#issuecomment-472574352
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result

@njit(fastmath=True, cache=True)
def np_mean(array, axis):
  return np_apply_along_axis(np.mean, axis, array)

@njit(fastmath=True, cache=True)
def np_std(array, axis):
  return np_apply_along_axis(np.std, axis, array)

''' Sampler
=====================' '''

@njit(fastmath=True, cache=True)
def bhs(Xorg:np.ndarray=np.array([[]]), yorg:np.ndarray=np.array([[]]), nsamples:int=1, burnin:int=0, thin:int=1):
    ''' Implementation of the Bayesian horseshoe linear regression hierarchy.
    
    Parameters:
      Xorg     = regressor matrix [n x p]
      yorg     = response vector  [n x 1]
      nsamples = number of samples for the Gibbs sampler (nsamples > 0)
      burnin   = number of burnin (burnin >= 0)
      thin     = thinning (thin >= 1)
   
    Returns:
      beta     = regression parameters  [p x nsamples]
      b0       = regression param. for constant [1 x nsamples]
      s2       = noise variance sigma^2 [1 x nsamples]
      t2       = hypervariance tau^2    [1 x nsamples]
      l2       = hypervariance lambda^2 [p x nsamples]

    References:
    A simple sampler for the horseshoe estimator
    E. Makalic and D. F. Schmidt
    arXiv:1508.03884, 2015
   
    The horseshoe estimator for sparse signals
    C. M. Carvalho, N. G. Polson and J. G. Scott
    Biometrika, Vol. 97, No. 2, pp. 465--480, 2010
   
    (c) Copyright Enes Makalic and Daniel F. Schmidt, 2015
    Adapted to python by Ricardo Baptista, 2018
    
    Adapted to Numba by Author: Lucas Rath, 2022
    '''

    n, p = Xorg.shape

    # Normalize data
    X, _, _, y, muY = standardise(Xorg, yorg)

    # Return values
    beta = np.zeros((p, nsamples))
    s2 = np.zeros((1, nsamples))
    t2 = np.zeros((1, nsamples))
    l2 = np.zeros((p, nsamples))

    # Initial values
    sigma2  = 1.
    lambda2 = np.random.uniform(0,1,p)
    tau2    = 1.
    nu      = np.ones(p)
    xi      = 1.

    # pre-compute X'*X (used with fastmvg_rue)
    XtX = X.T @ X # np.matmul(X.T,X)  

    # Gibbs sampler
    k = 0
    iter = 0
    while(k < nsamples):

        # Sample from the conditional posterior distribution
        sigma = np.sqrt(sigma2)
        Lambda_star = tau2 * np.diag(lambda2)
        # Determine best sampler for conditional posterior of beta's
        if (p > n) and (p > 200):
            b = fastmvg(X/sigma, y/sigma, sigma2*Lambda_star)
        else:
            b = fastmvg_rue(X/sigma, XtX/sigma2, y/sigma, sigma2*Lambda_star)

        # Sample sigma2
        e = y - np.dot(X,b)
        shape = (n + p) / 2.
        scale = np.dot(e.T,e)/2. + np.sum(b**2/lambda2)/tau2/2.
        sigma2 = 1. / np.random.gamma(shape, 1./scale)

        # Sample lambda2
        scale = 1./nu + b**2./2./tau2/sigma2
        # lambda2 = 1. / np.random.exponential(1./scale)
        lambda2 = np.array([1. / np.random.exponential(1./s) for s in scale])

        # Sample tau2
        shape = (p + 1.)/2.
        scale = 1./xi + np.sum(b**2./lambda2)/2./sigma2
        tau2 = 1. / np.random.gamma(shape, 1./scale)

        # Sample nu
        scale = 1. + 1./lambda2
        # nu = 1. / np.random.exponential(1./scale)
        nu = np.array([1. / np.random.exponential(1./s) for s in scale])

        # Sample xi
        scale = 1. + 1./tau2
        xi = 1. / np.random.exponential(1./scale)

        # Store samples
        iter = iter + 1;
        if iter > burnin:
            # thinning
            if (iter % thin) == 0:
                beta[:,k] = b
                s2[:,k]   = sigma2
                t2[:,k]   = tau2
                l2[:,k]   = lambda2
                k         = k + 1

    # Re-scale coefficients
    #div_vector = np.vectorize(np.divide)
    #beta = div_vector(beta.T, normX)
    #b0 = muY-np.dot(muX,beta)
    b0 = muY

    return (beta, b0, s2, t2, l2)

@njit(fastmath=True, cache=True)
def fastmvg(Phi:np.ndarray, alpha:np.ndarray, D:np.ndarray):
    ''' Fast sampler for multivariate Gaussian distributions (large p, p > n) of 
    the form N(mu, S), where
		mu = S Phi' y
		S  = inv(Phi'Phi + inv(D))
	Reference: 
		Fast sampling with Gaussian scale-mixture priors in high-dimensional 
		regression, A. Bhattacharya, A. Chakraborty and B. K. Mallick
		arXiv:1506.04778
	'''

    n, p = Phi.shape

    d = np.diag(D)
    u = np.random.randn(p) * np.sqrt(d)
    delta = np.random.randn(n)
    v = np.dot(Phi,u) + delta
    #w = np.linalg.solve(np.matmul(np.matmul(Phi,D),Phi.T) + np.eye(n), alpha - v)
    #x = u + np.dot(D,np.dot(Phi.T,w))
    
    # mult_vector = np.vectorize(np.multiply)
    # Dpt = mult_vector(Phi.T, d[:,np.newaxis])
    ### np.allclose( np.vectorize(np.multiply)(Phi.T, d[:,np.newaxis]), Phi.T * d[:,None] )
    ### np.allclose( np.vectorize(np.multiply)(Phi.T, d[:,np.newaxis]), Phi.T * np.expand_dims(d,1) )
    Dpt = Phi.T * np.expand_dims(d,1) # Phi.T * d[:,None]
    
    w = np.linalg.solve( Phi @ Dpt + np.eye(n), alpha - v )
    # w = sp.linalg.solve( Phi @ Dpt + np.eye(n), alpha - v, check_finite=True, assume_a='sym' )
    x = u + np.dot(Dpt,w)

    return x

@njit(fastmath=True, cache=True)
def fastmvg_rue(Phi:np.ndarray, PtP:np.ndarray, alpha:np.ndarray, D:np.ndarray):
    '''Another sampler for multivariate Gaussians (small p) of the form
		N(mu, S), where
		mu = S Phi' y
		S  = inv(Phi'Phi + inv(D))
    
    Here, PtP = Phi'*Phi (X'X is precomputed)
    
    Reference:
		Rue, H. (2001). Fast sampling of gaussian markov random fields. Journal
		of the Royal Statistical Society: Series B (Statistical Methodology) 
		63, 325-338
	'''

    p = Phi.shape[1]
    Dinv = np.diag(1./np.diag(D))

    # regularize PtP + Dinv matrix for small negative eigenvalues
    try:
        L = np.linalg.cholesky(PtP + Dinv)
    except:
        mat  = PtP + Dinv
        Smat = (mat + mat.T)/2.
        maxEig_Smat = np.max(np.linalg.eigvals(Smat))
        L = np.linalg.cholesky(Smat + maxEig_Smat*1e-15*np.eye(Smat.shape[0]))

    v = np.linalg.solve(L, np.dot(Phi.T,alpha))
    m = np.linalg.solve(L.T, v)
    w = np.linalg.solve(L.T, np.random.randn(p))

    x = m + w

    return x

@njit(fastmath=True, cache=True)
def standardise(X, y):
    ''' Standardize the covariates to have zero mean and x_i'x_i = 1 
    '''

    # set params
    n = X.shape[0]
    meanX = np_mean(X, axis=0)
    stdX  = np_std(X, axis=0) * np.sqrt(n)

    # Standardize X's
    #sub_vector = np.vectorize(np.subtract)
    #X = sub_vector(X, meanX)
    #div_vector = np.vectorize(np.divide)
    #X = div_vector(X, stdX)

    # Standardize y's
    meany = np.mean(y)
    y = y - meany

    return (X, meanX, stdX, y, meany)
