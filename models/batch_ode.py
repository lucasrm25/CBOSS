
import numpy as np
from tqdm import tqdm
from functools import cached_property, cache
from collections import namedtuple
import warnings

''' ---------------------
Continuous-Time integration
----------------------'''


class eRK_eODE():
    ''' Explicit Runge-Kutta solver for explicit ODE

    References:
        - https://github.com/scipy/scipy/blob/dde50595862a4f9cede24b5d1c86935c30f1f88a/scipy/integrate/_ivp/rk.py
    '''
    C: np.ndarray = NotImplemented
    A: np.ndarray = NotImplemented
    B: np.ndarray = NotImplemented
    E: np.ndarray = NotImplemented

    # def __init__(self, fun, rtol:float=1e-3, atol:float=1e-6):
    #     self.rtol = rtol
    #     self.atol = atol
    #     # self.nbr_stages = len(self.A)

    @classmethod
    @property
    @cache
    def nbr_stages(cls):
        return len(cls.A)

    @classmethod
    def _estimate_error(cls, K, dt):
        return np.einsum('b,s,bsd->bd',dt,cls.E,K)  # np.dot(K.T, self.E) * dt

    @classmethod
    def _estimate_error_norm(cls, K, dt, scale):
        '''
        Args:
            - K: <b,s+1,d> 
            - dt: <b,> 
            - scale: <b,d>
        '''
        return np.linalg.norm(cls._estimate_error(K, dt) / scale, axis=-1)

    @classmethod
    def _rk_step(self, f, p, x, t, dt):
        '''
        Args:
            - p: <b,> 
            - x: <b,d> 
            - t: <b,> 
            - dt: <b,> 
        '''
        batchsize, d = x.shape
        s = self.nbr_stages

        K = np.zeros( (batchsize, s+1, d))
        for ki in range(s):
            # with warnings.catch_warnings(record=True) as w:
            # self.A.shape = <s,s-1>
            K[:,ki,:] = f(
                p,
                t + self.C[ki] * dt,
                x + np.einsum('b,s,bsd->bd', dt, self.A[ki,:ki+1], K[:,:ki+1,:])
                # x + np.einsum('b,s,bsd->bd', dt, self.A[ki], K[:,:-1])
            )
        x_new = x + np.einsum('b,s,bsd->bd', dt, self.B, K[:,:-1])
        f_new = f(p, t + dt, x_new)
        # The last row is a linear combination of the previous rows with coefficients
        K[:,-1] = f_new

        return x_new, f_new, K
    
    @classmethod
    def solve(self, f, p, x0, t, rtol:float=1e-3, atol:float=1e-6, verbose=False):
        '''
        Args:
            p: <b,...> list of batch-specific function parameters
            t: <b,N> time array where solution is seeked. t[0] is the initial time for x0
            x0: <b,d> initial state for each batch
            f(t_k, x_k, u_k, p): <batch,dim>  ode function, e.g. f = @(p,t,x,u) lambda*x;
        Returns:
            x: <b,N,d> array with solution
            stable_mask: <b,> flag indicating simulation success
        '''

        assert x0.ndim==2 and t.ndim==2 and t.shape[0]==x0.shape[0], 'input dimensions do not agree'
        batchsize, d = x0.shape
        N = t.shape[1]  # number of time steps
        x = np.zeros((batchsize,N,d)) * np.nan
        K = np.zeros((batchsize,self.nbr_stages+1,d)) * np.nan
        x[:,0,:] = x0
        dt = t[:,1:] - t[:,:-1]

        # mask indicating stable batch samples
        smask = np.ones(batchsize,dtype=bool)
        error_norm = np.ones(batchsize) * np.inf

        t_range = tqdm(range(1,N)) if verbose else range(1,N)
        for it in t_range:

            x[smask,it,:], _, K[smask] =  self._rk_step(f=f, p=p[smask], x=x[smask,it-1,:], t=t[smask,it-1], dt=dt[smask,it-1])

            smask = (
                smask & 
                ~np.any(np.isnan(x[:,it,:]), axis=-1) & 
                ~np.any(np.isinf(x[:,it,:]), axis=-1) & 
                ~np.any(np.abs(x[:,it,:]) >= 1e10, axis=-1) &
                ~np.any(np.abs(K) >= 1e20, axis=(1,2))
            )

            scale = atol + np.maximum(np.abs(x[:,it,:]), np.abs(x[:,it-1,:])) * rtol    # <batchsize,d>
            error_norm[smask] = self._estimate_error_norm(K=K[smask], dt=dt[smask,it-1], scale=scale[smask])
            smask = smask & (error_norm < 1)
            
            if not np.any(smask):
                break

        return namedtuple('eRK45_eODE_result','x success')(x,smask) #dict(x=x, success=smask)

class eRK45_eODE(eRK_eODE):
    C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
    ])
    B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40])


# RKx = [c  A]
#       [0  b]
# RK4 = np.array( 
#         [[0,0,0,0,0],
#         [.5,.5,0,0,0],
#         [.5,0,.5,0,0],
#         [1,0,0,1,0],
#         [0,1/6,1/3,1/3,1/6]])
# RK2 = np.array( 
#         [[0,0,0], 
#         [.5,.5,0], 
#         [0,0,1]])
# RK1 = np.array( 
#         [[0,0], 
#         [0,1.]])

# def _eRK_eODE( f, p, x0, t, ButcherT, verbose=False):
#     ''' Explicit Runge-Kutta solver for explicit ODE
#     Args:
#         p: <b,...> list of batch-specific function parameters
#         t: <b,N> time array where solution is seeked. t[0] is the initial time for x0
#         x0: <b,d> initial state for each batch
#         f(t_k, x_k, u_k, p): <batch,dim>  ode function, e.g. f = @(p,t,x,u) lambda*x;
#     Returns:
#         x: <b,N,d> array with solution
#         stable_mask: <b,> flag indicating simulation success
#     '''
#     assert x0.ndim==2 and t.ndim==2 and t.shape[0]==x0.shape[0], 'input dimensions do not agree'
#     batchsize, d = x0.shape
#     N = t.shape[1]  # number of time steps
#     x = np.zeros((batchsize,N,d)) * np.nan
#     x[:,0,:] = x0
#     dt = t[:,1:] - t[:,:-1]

#     # split Butcher Tableu
#     A, b, c = ButcherT[:-1,1:], ButcherT[-1,1:], ButcherT[:-1,0]
#     nbr_stages = np.size(ButcherT,0)-1

#     stable_mask = np.ones(batchsize,dtype=bool)

#     t_range = tqdm(range(1,N)) if verbose else range(1,N)
#     for it in t_range:
#         K = np.zeros( (batchsize, nbr_stages, d))
#         for ki in range(nbr_stages):
#             bi = stable_mask
#             with warnings.catch_warnings(record=True) as w:
#                 K[bi,ki,:] = f(
#                     p[bi],
#                     t[bi,it-1]   + c[ki] * dt[bi,it-1], 
#                     x[bi,it-1,:] + dt[bi,it-1,None] * np.einsum('s,bsd->bd',A[ki],K[bi])
#                 )
#                 stable_mask = stable_mask & ~np.any(np.isnan(K),(1,2)) & ~np.any(K>=1e10,(1,2))
#         # assert not np.any(np.isnan(K))
#         x[:,it,:] = x[:,it-1,:] + dt[:,it-1,None] * np.einsum('s,bsd->bd',b,K)

#         rtol=1e-3
#         atol=1e-6
#         scale = atol + np.maximum(np.abs(x[:,it-1,:]), np.abs(x[:,it,:])) * rtol
#         error_estimate = np.dot(K.T, self.E) * h
#         error_norm_estimate = np.linalg.norm(self._estimate_error(K, h) / scale)

#         stable_mask = stable_mask & ~np.any(np.isnan(x[:,it,:]),-1) & ~np.any(x[:,it,:]>=1e10,-1)

#     return dict(x=x, success=stable_mask)
