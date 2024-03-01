''' 
Author: Lucas Rath

Management of Search spaces
Adapted from the MIPEGO project: https://github.com/wangronin/MIP-EGO
'''

from typing import Iterable, Union, List, TypeVar
import numpy as np
from itertools import accumulate, chain
from abc import abstractmethod
from enum import Enum
from copy import deepcopy
from functools import cached_property, reduce

class OneHotEncoder():
    ''' This class is used to encode categorical varibles into one-hot binary vectors.
    Other variable types are left unchanged.
    
    NOTE: categorical subspaces are encoded as binary if the number of categories equal 2
    '''
    def __init__(self, product_space:'ProductSpace'):
        self.ps = product_space

        # map nominal subspaces that are `onehot` encoded
        self.onehot_domains = np.asarray(list(chain(*[
            [i]
            if i not in self.ps.id_N or len(self.ps.subspaces[i].bounds) == 2   # NOTE: treat categorical variables with 2 categories as binary
            else [i]*len(self.ps.subspaces[i].bounds) 
            for i in range(len(self.ps))
        ])))

    def encode(self, X):
        return np.hstack([
            X[:,[j]] 
            if not is_N
            else (
                X[:,[j]].reshape((-1,1)) ==\
                (
                    np.array(self.ps.subspaces[j].bounds, dtype=object).reshape((1,-1))
                    if len(self.ps.subspaces[j].bounds) > 2
                    else
                    np.array(self.ps.subspaces[j].bounds, dtype=object).reshape((1,-1))[:,1]
                )
            ).astype(int)
            for j,is_N in enumerate(self.ps.N_mask)    # j,is_N = list(enumerate(self.ps.N_mask))[0]
        ]).astype(float)

    def decode(self, Xe):
        Xe_dims = []
        for i in range(Xe.shape[1]):
            if i in self.ps.id_N:
                if len(self.ps.subspaces[i].bounds) > 2:
                    Xe_dims += [np.asarray(self.ps.subspaces[i].bounds, dtype=object)[ np.where(Xe[:,i==self.onehot_domains])[1]]]
                else:
                    Xe_dims += [np.asarray(self.ps.subspaces[i].bounds, dtype=object)[ Xe[:,i==self.onehot_domains].astype(int).squeeze()]]
            elif i in self.ps.id_B or i in self.ps.id_O:
                Xe_dims += [Xe[:,i==self.onehot_domains].astype(int).squeeze()]
            elif i in self.ps.id_C:
                Xe_dims += [Xe[:,i==self.onehot_domains].astype(float).squeeze()]
        return np.vstack(Xe_dims).T

class IntegerEncoder():
    ''' This class is used to encode categorical varibles into integer values.
    Other variable types are left unchanged.
    '''
    def __init__(self, product_space:'ProductSpace'):
        self.ps = product_space
        
    def encode(self, X):
        return np.hstack([
            X[:,[j]] 
            if not is_N
            else (
                np.where(X[:,[j]] == np.array(self.ps.subspaces[j].bounds, dtype=object))[1][:,None]
            )
            for j,is_N in enumerate(self.ps.N_mask)
        ]).astype(float)
        
    def decode(self, Xe):
        Xe_dims = []
        for i, Xe_i in enumerate(Xe.T):
            if i in self.ps.id_N:
                Xe_dims += [np.asarray(self.ps.subspaces[i].bounds, dtype=object)[Xe_i.astype(int).squeeze()]]
            elif i in self.ps.id_B or i in self.ps.id_O:
                Xe_dims += [Xe_i.astype(int)]
            elif i in self.ps.id_C:
                Xe_dims += [Xe_i.astype(float)]
        return np.vstack(Xe_dims).T

class Var_Type(Enum):
        Continuous  = 'C'
        Ordinal     = 'O'
        Nominal     = 'N'
        Binary      = 'B'

class SearchSpace(object):
    '''  Base class for one dimensional Search Spaces
    '''
    def __init__(self, bounds:Iterable, var_name:str, var_type:str, description:str=''):
        # assert hasattr(bounds[0], '__iter__') and not isinstance(bounds[0], str)
        self.bounds   = tuple(bounds)
        self.var_type = var_type
        self.var_name = var_name
        self.description = description

    @abstractmethod
    def sample(self, N:int=1, random_generator=np.random):
        """The output is a list of shape (N, 1)
        """
        raise NotImplementedError

    def __len__(self):
        return 1

    def __iter__(self):
        pass

    def __mul__(self, space):
        '''Returns the Product Space of the two `SearchSpace`s
        '''
        if isinstance(space, SearchSpace):
            return ProductSpace(self, space)
        elif isinstance(space, ProductSpace):
            return ProductSpace(self, *space.subspaces)
        elif isinstance(space, int):
            N = space  # rename variable: N=number of repetitions
            if N == 1:
                return self
            copies = [deepcopy(self) for _ in range(N)]
            # change names
            for i,c in enumerate(copies):
                c.var_name += f'_{i}'
            return reduce(lambda a,b: a*b, copies)
        else:
            raise TypeError(f'Error creating product space: {space} must be a `SearchSpace`, `ProductSpace` or integer')

    def __rmul__(self,space):
        return self.__mul__(space=space)
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        _ =  f'Search Space of 1 variable: \n'
        _ += f' bounds: {str(self.bounds)} \n'
        return _

class ContinuousSpace(SearchSpace):
    """Continuous (real-valued) Search Space
    """
    def __new__(cls, bounds:Union[List[Iterable],Iterable]=None, var_name:str='c'):
        ''' If a multidimensional ContinuousSpace is to be created, then return a 
        ProductSpace object of each ContinuousSpace dimension
        '''
        if isinstance(bounds,list) and isinstance(bounds[0],list): # multidimensional Nominal Space is specified
            continuous_spaces = [cls(bounds=c, var_name=f'{var_name}_{id_c}') for id_c, c in enumerate(bounds) ]
            return ProductSpace(*continuous_spaces)
        else:
            obj = object.__new__(cls)
            return obj

    def __init__( self, bounds, var_name='c'):
        super().__init__(bounds=bounds, var_name=var_name, var_type=Var_Type.Continuous.value)
        assert self.bounds[0] < self.bounds[1]
    
    def sample(self, N=1, random_generator=np.random):
        return ((self.bounds[1] - self.bounds[0]) * random_generator.rand(N, 1) + self.bounds[0])

    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return f'{self.var_name} ({self.__class__.__name__}) - bounds: {self.bounds}'

class NominalSpace(SearchSpace):
    """Nominal (categorical) Search Space
    """
    def __new__(cls, categories:Union[List[Iterable],Iterable], var_name:str='n'):
        ''' If a multidimensional NominalSpace is to be created, then return a ProductSpace object of each NominalSpace dimension
        '''
        if isinstance(categories[0],list): # multidimensional Nominal Space is specified
            nominal_spaces = [cls(categories=c, var_name=f'{var_name}_{id_c}') for id_c, c in enumerate(categories) ]
            return ProductSpace(*nominal_spaces)
        else:
            obj = object.__new__(cls)
            return obj

    def __getnewargs__(self):
        return (self.bounds, self.var_name)

    def __init__(self, categories:Iterable, var_name:str='n'):
        ''' Handle onedimensional NominalSpace
        '''
        super().__init__(bounds=categories, var_name=var_name, var_type=Var_Type.Nominal.value)

    @property
    def categories(self):
        return self.bounds

    def sample(self, N=1, random_generator=np.random):
        idx = random_generator.randint(0, len(self.bounds), (N,1))
        X = np.asarray(self.bounds)[idx]
        return X

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'{self.var_name} ({self.__class__.__name__}) - categories: {self.bounds}'

class OrdinalSpace(SearchSpace):
    """Ordinal (integer) Search Space
    """
    def __new__(cls, bounds, var_name:str='o'):
        ''' If a multidimensional NominalSpace is to be created, then return a ProductSpace object of each NominalSpace dimension
        '''
        if isinstance(bounds[0],list):
            # if len(categories) <= 1: raise TypeError
            nominal_spaces = [cls(bounds=b, var_name=f'{var_name}_{id_b}') for id_b, b in enumerate(bounds) ]
            product_space = list(accumulate( nominal_spaces, func=ProductSpace ))[-1]
            return product_space
        else:
            ordinal_space = object.__new__(cls)
            return ordinal_space

    def __getnewargs__(self):
        return (self.bounds, self.var_name)

    def __init__(self, bounds, var_name='o'):
        super().__init__(bounds=bounds, var_name=var_name, var_type=Var_Type.Ordinal.value)
        assert self.bounds[0] < self.bounds[1]

    def sample(self, N=1, random_generator=np.random):
        return random_generator.randint(self.bounds[0], self.bounds[1]+1, (N,1))

    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return f'{self.var_name} ({self.__class__.__name__}) - bounds: {self.bounds}'

class BinarySpace(SearchSpace):
    def __init__(self, var_name='b'):
        super().__init__(bounds=(0,1), var_name=var_name, var_type=Var_Type.Binary.value)

    def sample(self, N=1, random_generator=np.random):
        return random_generator.randint(self.bounds[0], self.bounds[1]+1, (N,1))
    
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return f'{self.var_name} ({self.__class__.__name__})'

class ProductSpace():
    """Cartesian product of the search spaces
    """
    def __init__(self, *spaces:SearchSpace):
        assert np.all([isinstance(s,SearchSpace) for s in spaces])

        self.subspaces = list(spaces)
        self.bounds   = [s.bounds for s in spaces]
        self.var_name = [s.var_name for s in spaces]
        self.var_type = [s.var_type for s in spaces] 

        self.C_mask = np.asarray(self.var_type) == Var_Type.Continuous.value
        self.O_mask = np.asarray(self.var_type) == Var_Type.Ordinal.value
        self.N_mask = np.asarray(self.var_type) == Var_Type.Nominal.value
        self.B_mask = np.asarray(self.var_type) == Var_Type.Binary.value
        
        self.id_C = np.nonzero(self.C_mask)[0]
        self.id_O = np.nonzero(self.O_mask)[0]
        self.id_N = np.nonzero(self.N_mask)[0]
        self.id_B = np.nonzero(self.B_mask)[0]

        self.onehot_domains = np.asarray(list(chain(*[
            [i] if i not in self.id_N else [i]*len(self.subspaces[i].bounds) 
            for i in range(len(self))
        ])))

    def sample(self, N=1, random_generator=np.random):
        return np.hstack([ 
            np.asarray(s.sample(N,random_generator=random_generator)).astype(object) 
            for s in self.subspaces
        ])

    def __len__(self):
        return len(self.var_type)

    def __mul__(self, space):
        if isinstance(space,ProductSpace):
            return ProductSpace(*self.subspaces, *space.subspaces)
        elif isinstance(space,SearchSpace):
            return ProductSpace(*self.subspaces, space)
        else:
            raise TypeError

    def __rmul__(self, space):
        return self.__mul__(space=space)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        _ =  f'Product Space of {len(self)} variables: \n'
        for i in range(len(self)):
            _ += f'\t{self.subspaces[i]}\n'
        return _