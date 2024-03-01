''' Author: Lucas Rath
'''

import numpy as np
import torch
from torch.nn.parameter import Parameter
from pyro.distributions.multivariate_studentt import MultivariateStudentT
from torch.distributions.multivariate_normal import MultivariateNormal
from pyro.distributions import Unit
from torch.distributions import Distribution
from typing import Callable, Dict, List, Union
import abc
import types
from copy import copy, deepcopy


''' -------------- 
General
-------------- '''

def inputs_as_tensor(func:Callable):
    def modified_func(*args, **kwargs):
        tensor_args = [torch.as_tensor(a) for a in args]
        tensor_kwargs = {k:torch.as_tensor(a) for k,a in kwargs.items()}
        return func(*tensor_args, **tensor_kwargs)
    return modified_func

''' -------------- 
Module Extension
-------------- '''

class ModuleX(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__dtype__ = torch.nn.Parameter(torch.tensor(0.,dtype=torch.float32,))
        # self.register_buffer('__dtype__', )
        
    @property
    def dtype(self):
        # param_example = next(self.parameters(), None)
        # return param_example.dtype if param_example is not None else torch.float32
        params = list(self.parameters())
        return params[0].dtype if len(params) else torch.float32

    @property
    def device(self):
        # param_example = next(self.parameters(), None)
        # return param_example.device if param_example is not None else torch.cpu
        params = list(self.parameters())
        return params[0].device if len(params) else "cpu"

    def hyperparameters(self):
        return {k:v for k,v in self.named_modules() if isinstance(v,HyperParameter)}

    # @classmethod
    # @staticmethod
    # def __hyperparameters(module, _hp_dict:dict={}):
    #     _hp_dict |= {k:v for k,v in module.named_modules() if isinstance(v,HyperParameter)}
    #     modules = {k:v for k,v in module.named_modules() if isinstance(v,torch.nn.Module)}
    #     for k,v in modules.items():
    #         _hp_dict |= ModuleX.__hyperparameters(v, _hp_dict)
    #     return _hp_dict

''' -------------- 
Scaling
-------------- '''

class Scaler():
    def __init__(self, with_mean:bool=True, with_std:bool=True, copy:bool=True):
        self.with_mean = with_mean
        self.with_std  = with_std
        self.copy 	   = copy

    def fit(self, X:np.ndarray):
        assert X.ndim==2, 'expecting input with 2 dimensions'
        
        self._mean = np.nanmean(X, axis=0) if self.with_mean else np.zeros(X.shape[1])
        self._std  = np.nanstd(X, axis=0)  if self.with_std  else np.ones(X.shape[1])
        
        # idx_valid =  ~ (np.isnan(X) | np.isinf(X))
        # self._mean = np.zeros_like(X.mean(0))
        # if self.with_mean:
        #     self._mean = np.array([ Xi[idx_valid_i].mean() for idx_valid_i, Xi in zip(idx_valid.T, X.T) ]) 
        # self._std = np.ones_like(X.std(0))
        # if self.with_std:
        #     self._std = np.array([ Xi[idx_valid_i].std() for idx_valid_i, Xi in zip(idx_valid.T, X.T) ]) 
        return self

    def transform(self, X:Union[np.ndarray,torch.Tensor,List[MultivariateStudentT],List[MultivariateNormal]]):
        if isinstance(X,torch.Tensor) or isinstance(X,np.ndarray):
            X_new = np.copy(X) if self.copy else X
            X_new = (X_new - self._mean) / self._std
            return X_new
        elif isinstance(X,list):
            dist_list = X
            if isinstance(dist_list[0], MultivariateStudentT):
                transformed_dist = [
                    MultivariateStudentT(
                        df  = dist_i.df,
                        loc = (dist_i.loc - self._mean[i])  / self._std[i],
                        scale_tril = dist_i.scale_tril / self._std[i]
                    )
                    for i, dist_i in enumerate(dist_list)
                ]
                return transformed_dist
            elif isinstance(dist_list[0], MultivariateNormal):
                transformed_dist = [
                    MultivariateNormal(
                        loc = (dist_i.loc - self._mean[i])  / self._std[i],
                        scale_tril = dist_i.scale_tril / self._std[i]
                    )
                    for i, dist_i in enumerate(dist_list)
                ]
                return transformed_dist
            else:
                raise TypeError()
        else:
            raise TypeError()

    def inv_transform(self, X:Union[np.ndarray,torch.Tensor,List[MultivariateStudentT],List[MultivariateNormal]]):
        if isinstance(X,torch.Tensor) or isinstance(X,np.ndarray):
            X_new = copy(X) if self.copy else X
            X_new = X_new * self._std + self._mean
            return X_new
        elif isinstance(X,list):
            dist_list = X
            if isinstance(dist_list[0], MultivariateStudentT):
                transformed_dist = [
                    MultivariateStudentT(
                        df  = dist_i.df,
                        loc = dist_i.loc * self._std[i] + self._mean[i],
                        scale_tril = dist_i.scale_tril * self._std[i]
                    )
                    for i, dist_i in enumerate(dist_list)
                ]
                return transformed_dist
            elif isinstance(dist_list[0], MultivariateNormal):
                transformed_dist = [
                    MultivariateNormal(
                        loc = dist_i.loc * self._std[i] + self._mean[i],
                        scale_tril = dist_i.scale_tril * self._std[i]
                    )
                    for i, dist_i in enumerate(dist_list)
                ]
                return transformed_dist
            else:
                raise TypeError()
        else:
            raise TypeError()

''' -------------- 
Constraints transformations 
-------------- '''

def sigmoid(x):
    return torch.sigmoid(x)

def inv_sigmoid(x):
    return torch.log(x) - torch.log(1 - x)

def softplus(x):
    return torch.nn.functional.softplus(x)

def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))

def soft_abs(x, eps=1e-6):
    return torch.sqrt( x**2 + eps**2 ) - eps


''' -------------- 
Constraints and Hyperparameter
-------------- '''

class Constraint(abc.ABC, torch.nn.Module):
    def __init__(self, transform:Callable=None, inv_transform:Callable=None, lower_bound=-torch.inf, upper_bound=torch.inf):
        super().__init__()
        self._transform = transform
        self._inv_transform = inv_transform
        # we make bounds as Parameters instances so they are stored in state_dict 
        # (storing only the raw Parameter value is not enough, since the transformed value to be used in the algorithm deppends on the bound values)
        # but since we do never want to learn them, we set requires_grad=False
        self.lower_bound = Parameter(torch.tensor(lower_bound), requires_grad=False) # if lower_bound is not None else -torch.inf
        self.upper_bound = Parameter(torch.tensor(upper_bound), requires_grad=False) # if upper_bound is not None else torch.inf
    
    @abc.abstractmethod
    def transform(self, x): 
        pass
    
    @abc.abstractmethod
    def inv_transform(self, x): 
        pass
    
    def __str__(self):
        return f'{self.__class__.__name__} ({self.lower_bound.tolist()},{self.upper_bound.tolist()})'
    
    def __repr__(self):
        return self.__str__()

class DummyConstraint(Constraint):
    def __init__(self):
        super().__init__()

    def transform(self, x): 
        return x

    def inv_transform(self, x): 
        return x 

class Interval(Constraint):
    def __init__(self, lower_bound=0., upper_bound=1.):
        super().__init__(transform=sigmoid, inv_transform=inv_sigmoid, lower_bound=lower_bound, upper_bound=upper_bound)

    def transform(self, x):
        return (self._transform(x) * (self.upper_bound - self.lower_bound)) + self.lower_bound

    def inv_transform(self, x):
        return self._inv_transform((x - self.lower_bound) / (self.upper_bound - self.lower_bound))

class GreaterThan(Constraint):
    def __init__(self, lower_bound=0.):
        super().__init__(transform=softplus, inv_transform=inv_softplus, lower_bound=lower_bound)

    def transform(self, x):
        return self._transform(x) + self.lower_bound

    def inv_transform(self, x):
        return self._inv_transform(x - self.lower_bound)

class LessThan(Constraint):
    def __init__(self, upper_bound=0.0):
        super().__init__(transform=softplus, inv_transform=inv_softplus, upper_bound=upper_bound)

    def transform(self, x):
        return -self._transform(-x) + self.upper_bound

    def inv_transform(self, x):
        return -self._inv_transform(-(x - self.upper_bound)) 

class ConstrainedParameter(ModuleX):
    def __init__(self, tensor:torch.Tensor, constraint:Constraint=DummyConstraint(), requires_grad=True):
        super().__init__()
        self.constraint = constraint #.to(tensor.device)
        self.raw = Parameter(self.constraint.inv_transform(tensor), requires_grad=requires_grad)  # inverse of sigmoid
    
    def __call__(self):
        return self.constraint.transform(self.raw)
    
    def __str__(self):
        return f'Constrained Parameter:\n\tTensor:     {self().data.__str__()}\n\tConstraint: {self.constraint.__str__()}'
    
    def __repr__(self):
        return self.__str__()

def printParameterList(model, sformat = '{:60s} {:8s} {:20s} {:30s} {:50s}'):
    ''' Print model parameters
    '''
    print('\nParameter list:')
    print(sformat.format('Name','Type','Size','True Value', 'Constraint'))
    print(sformat.format('-'*40,'-'*6,'-'*15,'-'*20,'-'*40))
    pretty = lambda list_: [f"{element:.4f}" for element in list_.flatten()]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        base_name = name
        base_module = model
        while "." in base_name:
            components = base_name.split(".")
            submodule_name = components[0]
            submodule = getattr(base_module, submodule_name)
            base_module = submodule
            base_name = ".".join(components[1:])

        constraint = None
        if type(base_module).__name__ == 'ConstrainedParameter':
            constraint = base_module.constraint
            name = ".".join(name.split(".")[:-1]) # exclude str after last '.'

        # if param.requires_grad:
        print(sformat.format(
            name, 
            type(param.data).__name__, 
            list(param.size()).__str__(), 
            ' '.join( pretty(param if constraint is None else constraint.transform(param)) ),
            constraint.__str__()
        )) 
    print('\n')

class HyperParameter(ConstrainedParameter):
    ''' This class represents Hyperparameters objects

    It facilitates the implementation of Empirical Bayes approaches such as MLE or MMAP.
    The idea is that it can add an additional level of hyperpriors, by placing another parameterized prior over this hyperparameter.
    Therefore, this hyperparameter is modelled as a random parameter during training and as a deterministic variable during evaluation.
    '''
    # def _hyperprior_logprob(self):
    #     '''
    #     Example:
    #         `return Gamma(self.zeta, self.iota).log_prob(self())`
    #     '''
    #     return torch.tensor(0.)

    # def hyperprior_logprob(self):
    #     assert self.training
    #     return self._hyperprior_logprob()

    def __init__(self, 
            tensor: torch.Tensor, constraint: Constraint = DummyConstraint(), requires_grad=True, 
            hyperprior:Callable[['HyperParameter'],torch.distributions.Distribution]=None, **params
        ):
        super().__init__(tensor, constraint, requires_grad)
        for k,p in params.items():
            setattr(self, k, p)

        if hyperprior is not None:
            self.hyperprior = types.MethodType(hyperprior, self)

    def hyperprior(self) -> Distribution:
        ''' Default is the uninformative prior: `p(hyperparam) /propto 1`
        '''
        return Unit(0.)

    def __str__(self):
        return f'Constrained HYPER-Parameter:\n\tTensor:     {self().data.__str__()}\n\tConstraint: {self.constraint.__str__()}'

''' -----
Tensor operations that are done with batched vectors <b,i> and batched matrices <b,i,j>
----- '''

def skew(vec:torch.Tensor):
    ''' 
    Generates a skew-symmetric matrix given a vector w
    '''
    S = torch.zeros([3,3], device=vec.device)

    S[0,1] = -vec[2]
    S[0,2] =  vec[1]
    S[1,2] = -vec[0]
    
    S[1,0] =  vec[2]
    S[2,0] = -vec[1]
    S[2,1] =  vec[0]
    return S

def batch(fun, vec:torch.Tensor):
    ''' Execute fun along the vec first dimension. 
    Returns the stacked batch of results
    '''
    # add extra dimension if batch dimension is missing
    bvec = vec.unsqueeze(0) if vec.dim()==1 else vec
    # stack batch results into the first dimension (the batch dimension)
    return torch.stack( [fun(v) for v in bvec] )

def rotZ(angle):
    return torch.tensor([
        [torch.cos(angle),-torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0],
        [0,                0,                1]
    ], device=angle.device)

def inputInfo(q:torch.Tensor):
    device, batchSize = q.device, q.shape[0]
    return device, batchSize

def bmm(m1:torch.Tensor, m2:torch.Tensor):
    ''' batch matrix-matrix multiplication
    '''
    return m1 @ m2

def bmv(m:torch.Tensor, v:torch.Tensor):
    ''' batch matrix-vector multiplication
    '''
    if m.dim()==3 and v.dim()==2:
        return torch.einsum('bij,bj->bi',m,v)
    elif m.dim()==2 and v.dim()==2:
        return torch.einsum('ij,bj->bi',m,v)
    elif m.dim()==3 and v.dim()==1:
        return torch.einsum('bij,j->bi',m,v)
    else:
        raise Exception(f'invalid matrix-vector multiplication: m.shape:{m.shape} and v.shape:{v.shape}')

def bvm(v:torch.Tensor, m:torch.Tensor):
    ''' batch vector-matrix multiplication
    '''
    if m.dim()==3 and v.dim()==2:
        return torch.einsum('bi,bij->bj',v,m)
    elif m.dim()==2 and v.dim()==2:
        return torch.einsum('bi,ij->bj',v,m)
    elif m.dim()==3 and v.dim()==1:
        return torch.einsum('i,bij->bj',v,m)
    else:
        raise Exception(f'invalid vector-matrix multiplication: m.shape:{m.shape} and v.shape:{v.shape}')

def binner(v1:torch.Tensor, v2:torch.Tensor):
    ''' batch inner product between vectors
    '''
    if v1.dim()==2 and v2.dim()==2:
        return torch.einsum('bi,bi->b',v1,v2).unsqueeze(-1)
    elif v1.dim()==2 and v2.dim()==1:
        return torch.einsum('bi,i->b',v1,v2).unsqueeze(-1)
    elif v1.dim()==1 and v2.dim()==2:
        return torch.einsum('i,bi->b',v1,v2).unsqueeze(-1)
    else:
        raise Exception(f'invalid inner product: v1.shape:{v1.shape} and v2.shape:{v2.shape}')

def bT(m:torch.Tensor):
    ''' batch transpose of matrix (transpose the last two dimensions)
    '''
    # assert m.dim() > 2, 'batch transpose requires input dimension >= 3 (1 batch dimension + 2 matrix dimensions)'
    return m.transpose(-1,-2)

def blstsq(A,b):
    ''' solves batch of least squares problem  x = pinv(A) @ b
    Args:
        A: <b,m,m>, b:<b,m>
    Returns:
        x: <b,m>
    '''
    return bmv( torch.pinverse(A) , b )

def bsolve(A,b):
    ''' solves batch of least squares problem  x = pinv(A) @ b
    Args:
        A: <b,m,m>, b:<b,m>
    Returns:
        x: <b,m>
    '''
    return torch.solve( b.transpose(-1,-2), A )[0]

def beye(batchSize, n, device):
    ''' batch eye matrix 
    '''
    return torch.eye(n,device=device).repeat(batchSize,1,1)

def beye_like(tensor):
    ''' tensor must have the size <batchSize, n, n>
    '''
    assert tensor.dim == 3 and tensor.shape[-1] == tensor.shape[-1]
    batchSize, n, device = tensor.shape[0], tensor.tensor.shape[-1], tensor.device
    return beye(batchSize, n, device)


''' ---------
Loss Functions
--------- '''

def weighted_mse_loss(output, target, weight, dim=[]):
    return torch.mean(weight * (output - target) ** 2, dim=dim)

def weighted_L1_loss(output, target, weight):
    return torch.mean(weight * torch.abs(output - target))

def weighted_NRMSE(output, target, weight, dim=[]):
    rmse = torch.sqrt(weighted_mse_loss(output, target, weight, dim=dim))
    return rmse/(torch.amax(output,dim=dim)-torch.amin(output, dim=dim))

def weighted_r2(output, target, weight, dim=[]):
    target_mean = torch.mean(target, dim=dim, keepdim=True)
    ss_tot = torch.sum( weight * (target - target_mean) ** 2, dim=dim)
    ss_res = torch.sum( weight * (target - output) ** 2, dim=dim)
    r2 = 1 - ss_res/ss_tot
    return r2
