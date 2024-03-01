'''
Author: Lucas Rath
'''

import numpy as np
from scipy.spatial import distance_matrix
from itertools import combinations, combinations_with_replacement
from sklearn import preprocessing

class IFeatures():
    def __init__(self, hyperparams:list, bounds:list, feat_selection:np.ndarray=None):
        '''
        Args:
             - feat_selection: array of integer indices, specifying which features should be selected 
        '''
        self.hyperparams = hyperparams
        self.bounds = bounds
        self.feat_selection = feat_selection

    @property
    def feat_selection(self):
        return self._feat_selection
        
    @feat_selection.setter
    def feat_selection(self, new_feat_selection:np.ndarray=None):
        assert \
            new_feat_selection is None or \
            (
                isinstance(new_feat_selection,np.ndarray) and \
                new_feat_selection.dtype == 'int' and \
                len(set(new_feat_selection)) == len(new_feat_selection)\
            ),\
            'feat_selection argument must be a numpy integer array without repetitive indices'
        self._feat_selection = new_feat_selection

    def __call__(self, X):
        ''' 
        Args:
            - X: <N, Nx> N input vectors of size Nx
        Returns 
            - F: <N Nf> mapping of the input vectors to any feature space
        '''
        return X

class PolynomialFeatures_DEPRECATED(IFeatures):
    def __init__(self, order:int, include_bias:bool=True, interaction_only:bool=True, feat_selection:np.ndarray=None):
        super().__init__(hyperparams=[], bounds=[], feat_selection=feat_selection)
        self.order = order
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        # self.nCoeffs = np.sum([comb(self.nVars, o) for o in range(self.order+1)])     # every combination up to self.order
    
    def __call__(self, X):
        ''' f(X) = alpha_0 + sum_i alpha_i * x_i + sum_{i,j>i} alpha_{ij} * x_i * x_j + ...
                 = F(X) * alpha
        Args:
        - X: <N, nx>
        Returns:
        - F: <N, Nf>
        '''
        comb_fun = combinations if self.interaction_only else combinations_with_replacement
        # Find number of variables
        N, nx = X.shape
        # Generate matrix to store results
        F = X
        for ord_i in range(2, self.order+1):
            # generate all combinations of indices
            idxs = np.array(list(comb_fun(np.arange(nx), ord_i)), dtype=int)
            
            if np.size(idxs) > 0:
                # generate products of input variables
                x_comb = X[:, idxs]
                F = np.append(F, np.prod(x_comb, axis=2), axis=1)

        if self.include_bias:
            F = np.c_[np.ones(N), F]

        if self.feat_selection is not None:
            F = F[:,self.feat_selection]

        return F.astype(np.float64)

class PolynomialFeatures(IFeatures, preprocessing.PolynomialFeatures):
    def __init__(self,  interaction_only:bool=True, include_bias:bool=True, order:str='C', feat_selection:np.ndarray=None, **kwargs):
        IFeatures.__init__(self, hyperparams=[], bounds=[], feat_selection=feat_selection)
        preprocessing.PolynomialFeatures.__init__(
            self, 
            interaction_only=interaction_only, 
            include_bias=include_bias,
            order=order,
            **kwargs
        )
    
    def __call__(self, X):
        ''' f(X) = alpha_0 + sum_i alpha_i * x_i + sum_{i,j>i} alpha_{ij} * x_i * x_j + ...
                 = F(X) * alpha
        Args:
        - X: <N, nx>
        Returns:
        - F: <N, Nf>
        '''
        F = self.fit_transform(X)

        if self.feat_selection is not None:
            F = F[:,self.feat_selection]

        return F #.astype(np.float64)

class RadialBasisFeatures(IFeatures):
    ''' Gaussian radial basis features
    '''
    def __init__(self, X:np.ndarray, centers:np.ndarray, length_scales:np.ndarray):
        assert centers.shape[1] == X.shape[1] and len(length_scales) == len(centers)
        self.X = X
        self.centers = centers
        bounds = [[1e-6, None]] * len(length_scales)
        super().__init__(hyperparams=length_scales, bounds=bounds)
    
    def __call__(self, X):
        ''' f(x) = exp( -(x-x1)/ls1 ) + ... + exp( -(x-xN)/lsN )
        Args:
        - X: <N,Nx>
        Returns:
        - F: <N, Nf>
        '''
        length_scales = self.hyperparams
        return np.exp( - distance_matrix(X,self.centers)**2 / (2*length_scales**2) ) 

class SigmoidFeatures(IFeatures):
    ''' Sigmoid features
    
    References: 
        - Bishop 2006 ~pg139
    '''
    def __init__(self, X:np.ndarray, length_scales:np.ndarray):
        self.X = X
        bounds = [[1e-6, None]] * len(length_scales)
        super().__init__(hyperparams=length_scales, bounds=bounds)
    
    def __call__(self, X):
        ''' f(x) = sigmoid( (x-x1)/ls1 ) + ... + sigmoid( (x-xN)/lsN )
        Args:
        - X: <N,Nx>
        Returns:
        - F: <N, Nf>
        '''
        length_scales = self.hyperparams
        return 1 / (1 + np.exp( - distance_matrix(X,self.X) / length_scales ))
