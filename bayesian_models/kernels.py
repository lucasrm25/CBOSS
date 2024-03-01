'''
Author: Lucas Rath
'''

from typing import List
import math
from itertools import accumulate, chain
import torch
from torch.nn.parameter import Parameter
from torch.distributions.beta import Beta
from scipy.spatial.distance import pdist, cdist, squareform
# from gpytorch.kernels.rbf_kernel import RBFKernel
# from gpytorch.kernels import ScaleKernel
from gpytorch.kernels.kernel import sq_dist
from CBOSS.utils.torch_utils import ModuleX, HyperParameter, GreaterThan, Interval
from CBOSS.bayesian_models.features import PolynomialFeatures
from CBOSS.models.search_space import ProductSpace, OneHotEncoder, NominalSpace, BinarySpace

# @torch.compile
@torch.jit.script
def _hamming_kernel(X1:torch.Tensor, X2:torch.Tensor, length_scales:torch.Tensor):
    '''
    References:
        - Think Global and Act Local: Bayesian Optimization over High-Dimensional Categorical and Mixed Search Spaces - Wan et al. 2021
        - https://github.com/xingchenwan/Casmopolitan/blob/main/bo/kernels.py#L223
    '''
    # expand dims -> NOTE: this operation is really memory consuming
    diff = X1[:,None] - X2[None,:]
    # nonzero location = different cat
    # diff[torch.abs(diff) > 1e-5] = 1
    # invert, to now count same cats
    kron = torch.logical_not(diff)      # <N1,N2,nf>
    K = torch.exp(torch.sum(kron * length_scales, dim=-1) / torch.sum(length_scales))       # <N1,N2>

    # # ! DEPRECATED: slow but memory efficient
    # N1, d = X1.shape
    # K = torch.vstack([
    #     ((X1_i == X2) * length_scales).sum(1) / d
    #     for X1_i in X1
    # ])
    # K = torch.exp(K)
    #### print( torch.abs(K-K_slow).max())
    return K

class HammingKernel(ModuleX):
    '''
    Evaluates a Hamming Kernel
        K(x,x') = exp( 1/d * sum_1^d ( lengthscale_i * delta(x_i,x'_i)  ) )
    where d is the dimension of x or the number of features

    Parameters:
        - lenghtscales: <d>
    '''
    def __init__(self, length_scales:list):
        super().__init__()
        self.length_scales = HyperParameter(tensor=torch.tensor(length_scales), requires_grad=True, constraint=Interval(1e-4,1e3))

    def forward(self, X1:torch.Tensor, X2:torch.Tensor):
        assert isinstance(X1,torch.Tensor) and isinstance(X2,torch.Tensor)
        assert X1.shape[1] == X2.shape[1]
        assert self.length_scales().shape==() or self.length_scales().shape == (X1.shape[1],)

        K = _hamming_kernel(X1=X1, X2=X2, length_scales=self.length_scales())

        assert K.shape == (len(X1),len(X2))
        return K

# @torch.compile
@torch.jit.script
def _binary_discrete_diffusion_kernel(X1:torch.Tensor, X2:torch.Tensor, length_scales:torch.Tensor):
    hammdist = torch.abs(X1[:,None] - X2[None,:])      # <N1,N2,nf>
    idx_eq = hammdist < 1e-8
    hammdist[idx_eq] = 0
    hammdist[~idx_eq] = 1
    K = torch.prod( torch.pow( torch.tanh(length_scales), hammdist), dim=-1)     # <N1,N2>    
    return K

class BinaryDiscreteDiffusionKernel(ModuleX):
    '''
    Evaluates a discrete Diffusion Kernel

    Parameters:
        - lenghtscales: <d>

    References:
        - Bayesian Optimization over Hybrid Spaces - Deshwal et al. 2021
    '''
    def __init__(self, length_scales:list):
        super().__init__()
        self.length_scales = HyperParameter(tensor=torch.as_tensor(length_scales), requires_grad=True, constraint=Interval(1e-4,1e3))

    def forward(self, X1:torch.Tensor, X2:torch.Tensor):
        ''' Evaluates Kernel

        Reference: https://github.com/xingchenwan/Casmopolitan/blob/main/bo/kernels.py#L223
        '''
        assert isinstance(X1,torch.Tensor) and isinstance(X2,torch.Tensor)
        assert X1.shape[1] == X2.shape[1]
        assert self.length_scales().shape==() or self.length_scales().shape == (X1.shape[1],)

        K = _binary_discrete_diffusion_kernel(X1=X1, X2=X2, length_scales=self.length_scales())

        assert K.shape == (len(X1),len(X2))
        return K

# @torch.compile
@torch.jit.script
def _discrete_diffusion_kernel(X1:torch.Tensor, X2:torch.Tensor, length_scales:torch.Tensor, cardinalities:torch.Tensor):
    hammdist = torch.abs(X1[:,None] - X2[None,:])      # <N1,N2,nf>
    # nonzero location = different values.
    idx_eq = hammdist < 1e-8
    hammdist[idx_eq] = 0
    hammdist[~idx_eq] = 1
    exp = torch.exp(-cardinalities*length_scales)   # <nf>
    K = torch.prod( # <N1,N2>
        torch.pow( 
            (1 - exp) / (1 + (cardinalities - 1) * exp), 
            hammdist
        ), 
        dim=-1
    )
    return K

class DiscreteDiffusionKernel(ModuleX):
    '''
    Evaluates a discrete Diffusion Kernel

    Parameters:
        - lenghtscales: <d>

    References:
        - Bayesian Optimization over Hybrid Spaces - Deshwal et al. 2021
    '''
    def __init__(self, length_scales:list, product_space:ProductSpace):
        super().__init__()
        self.length_scales = HyperParameter(tensor=torch.as_tensor(length_scales), requires_grad=True, constraint=Interval(1e-4,1e3))
        self.product_space = product_space
        self.cardinalities = torch.Tensor(
            self.product_space.B_mask * 2 + 
            self.product_space.N_mask * [len(s.bounds) for s in self.product_space.subspaces]
        )

    def forward(self, X1:torch.Tensor, X2:torch.Tensor):
        ''' Evaluates Kernel

        Reference: https://github.com/xingchenwan/Casmopolitan/blob/main/bo/kernels.py#L223
        '''
        assert isinstance(X1,torch.Tensor) and isinstance(X2,torch.Tensor)
        assert X1.shape[1] == X2.shape[1]
        assert self.length_scales().shape==() or self.length_scales().shape == (X1.shape[1],)

        K = _discrete_diffusion_kernel(X1=X1, X2=X2, length_scales=self.length_scales(), cardinalities=self.cardinalities)

        assert K.shape == (len(X1),len(X2))
        return K

class RBFKernel(ModuleX):
    def __init__(self, length_scales:list, variance_prior:float=0.5):
        super().__init__()
        self.variance_prior = HyperParameter(tensor=torch.as_tensor(variance_prior), requires_grad=True, constraint=Interval(1e-4,1e3))
        self.length_scales = HyperParameter(tensor=torch.as_tensor(length_scales), requires_grad=True, constraint=Interval(1e-4,1e3)) 
        
    def forward(self, X1:torch.Tensor, X2:torch.Tensor):
        assert isinstance(X1,torch.Tensor) and isinstance(X2,torch.Tensor)
        assert X1.shape[1] == X2.shape[1]
        
        l = self.length_scales()
        dist = torch.cdist( x1=X1/l, x2=X2/l, p=2 )
        K = torch.exp( -0.5 * dist**2 )
        K = self.variance_prior() * K
        
        # K_test = self.variance_prior()*sq_dist(x1=X1/l, x2=X2/l).div_(-2).exp_()
        # assert torch.abs(K - K_test).max() < 1e-6
        
        return K

class PolynomialKernel(ModuleX):
    '''
    Evaluates a Polynomial kernel:
        K(X,X') = F(X) @ Cov @ F(X').T
    where 
        F(x) = [1, x1, ..., x_d, x1*x2, ..., xd-1*xd, ...]  - polynomial feature vector of degree `degree`
        Cov  = I_d * variance_prior                         - variance for the polynomial coefficients (same covariance for all dimensions)

    Parameters:
        - variance_prior
    '''
    def __init__(self, variance_prior:float=0.5, degree:int=2, **kwargs):
        super().__init__()
        self.features = PolynomialFeatures(degree=degree, **kwargs)
        self.variance_prior = HyperParameter(tensor=torch.tensor(variance_prior), requires_grad=True, constraint=Interval(1e-4,1e3))

    def forward(self, X1:torch.Tensor, X2:torch.Tensor):
        ''' Evaluates Kernel
        Arguments:
            - X1: <N1,d>
            - X2: <N2,d>
        '''
        assert isinstance(X1,torch.Tensor) and isinstance(X2,torch.Tensor)
        assert X1.shape[1] == X2.shape[1]

        F1 = torch.as_tensor(self.features(X=X1), dtype=self.dtype, device=self.device)
        F2 = torch.as_tensor(self.features(X=X2), dtype=self.dtype, device=self.device)
        Nf = F1.shape[1]

        K_alpha = torch.eye(Nf, dtype=self.dtype, device=self.device) * self.variance_prior()
        K = F1 @ K_alpha @ F2.T
        assert K.shape == (len(X1),len(X2))
        return K

class OneHotEncodedKernel(ModuleX):
    ''' This module encodes the categorical inputs as one-hot vectors and then applies a kernel function.
    '''
    def __init__(self, kernel:ModuleX, product_space:ProductSpace):
        super().__init__()
        self.kernel = kernel
        self.encoder = OneHotEncoder(product_space=product_space)
        self.product_space = product_space
        
    def onehot(self, X:torch.Tensor):
        Xoh = torch.hstack([
            X[:,[j]] 
            if not is_N
            else (
                X[:,[j]].reshape((-1,1)) ==\
                (
                    torch.tensor(self.product_space.subspaces[j].bounds).reshape((1,-1))
                    if len(self.product_space.subspaces[j].bounds) > 2
                    else
                    torch.tensor(self.product_space.subspaces[j].bounds).reshape((1,-1))[:,1]
                )
            ).int()
            for j,is_N in enumerate(self.product_space.N_mask)    # j,is_N = list(enumerate(self.ps.N_mask))[0]
        ])
        return Xoh
        # return torch.vstack([   
        #     xi
        #     if i not in self.product_space.id_N
        #     else
        #     torch.nn.functional.one_hot( xi, num_classes=len(self.product_space.subspaces[i].categories) )
        #     for i, xi in enumerate(X.T)
        # ]).T
    
    @staticmethod
    def onehot_len(product_space:ProductSpace):
        return sum([
            1 * isinstance(s, BinarySpace) + 
            1 * (isinstance(s, NominalSpace) and len(s.bounds)==2) + 
            len(s.bounds) * (isinstance(s, NominalSpace) and len(s.bounds)>2)
            for s in product_space.subspaces
        ])
            
    def forward(self, X1:torch.Tensor, X2:torch.Tensor):
        X1e = self.onehot(X=X1)
        X2e = self.onehot(X=X2)
        return self.kernel(X1=X1e, X2=X2e)

class ProductKernel(ModuleX):
    '''
    Calculates the product of the given list of kernels:
        K1(X1,X2) * K2(X1,X2) * K2(X1,X2) * ...
    '''
    def __init__(self, kernelList:List[torch.nn.Module]):
        super().__init__()
        self.kernelList = torch.nn.ModuleList(modules=kernelList)
    
    def __call__(self, X1, X2):
        assert isinstance(X1,torch.Tensor) and isinstance(X2,torch.Tensor)
        # K_X1X2_list = torch.stack([K(X1,X2) for K in self.kernelList])
        K_X1X2_list = [K(X1,X2) for K in self.kernelList]
        K_X1X2 = math.prod(K_X1X2_list)
        return K_X1X2

class AdditiveKernel(ModuleX):
    '''
    Calculates the sum of the given list of kernels:
        K1(X1,X2) + K2(X1,X2) + K2(X1,X2) * ...
    '''
    def __init__(self, kernelList:List[torch.nn.Module]):
        super().__init__()
        self.kernelList = torch.nn.ModuleList(modules=kernelList)
    
    def __call__(self, X1, X2):
        assert isinstance(X1,torch.Tensor) and isinstance(X2,torch.Tensor)
        # K_X1X2_list = torch.stack([K(X1,X2) for K in self.kernelList])
        K_X1X2_list = [K(X1,X2) for K in self.kernelList]
        K_X1X2 = sum(K_X1X2_list)
        return K_X1X2

class CombKernel(ModuleX):
    '''
    Calculates the weighted sum of the sum of kernels and the product of kernels
        K = lambda * ( K1 * K2 * ...) + (1 - lambda) * (K1 + K2 + ...)
    '''
    def __init__(self, kernelList:List[torch.nn.Module], lambd:float=0.5):
        super().__init__()
        self.kernelList = torch.nn.ModuleList(modules=kernelList)
        self.lambd = HyperParameter(
            torch.tensor(lambd), constraint=Interval(0.,1.), requires_grad=True,
            alpha = Parameter(torch.tensor(1.5), requires_grad=False),
            beta  = Parameter(torch.tensor(1.5), requires_grad=False),
            hyperprior = lambda self: Beta(self.alpha, self.beta)
        )
    
    def __call__(self, X1, X2):
        assert isinstance(X1,torch.Tensor) and isinstance(X2,torch.Tensor)
        # K_X1X2_list = torch.stack([K(X1,X2) for K in self.kernelList])
        K_X1X2_list = [K(X1,X2) for K in self.kernelList]
        K_X1X2_sum  = sum(K_X1X2_list)
        K_X1X2_prod =  math.prod(K_X1X2_list)
        K_X1X2 = self.lambd() * K_X1X2_prod + (1 - self.lambd()) * K_X1X2_sum
        assert K_X1X2.shape == (len(X1),len(X2))
        return K_X1X2
