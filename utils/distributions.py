
import numpy as np
from scipy import stats

class multivariate_t(stats._multivariate.multivariate_t_frozen):
    def __init__(self, loc=None, shape=1, df=1, allow_singular=False, seed=None):
        shape = 0.5 * shape + 0.5 * np.transpose(shape) # + 1e-6*np.eye(len(shape))# enforce shape is positive definite
        super().__init__(loc=loc, shape=shape, df=df, allow_singular=allow_singular, seed=seed)

    @property
    def cov(self):
        if self.df > 2:
            return self.df/(self.df-2) * self.shape
        else:
            return self.shape * np.nan

    @property
    def mean(self):
        if self.df > 1:
            return self.loc
        else:
            return self.loc * np.nan

    @property
    def mode(self):
        return self.loc

    @property
    def median(self):
        return self.loc

    def __matmul__(self, factor):
        factor = np.asarray(factor)
        assert factor.shape == self.shape.shape, f'wrong multiplication of multivariate_t'
        return multivariate_t(
            df = self.df,
            loc = factor @ self.loc,
            shape = factor @ self.shape @ factor.T
        )
    
    def __mul__(self, factor):
        factor = np.asarray(factor)
        assert factor.ndim in [0,1]
        assert factor.shape == () or factor.shape == (1,) or factor.shape == self.loc.shape, f'wrong multiplication of multivariate_t'
        return self @ (factor * np.eye(len(self.loc)))
        
    def __add__(self, bias):
        bias = np.asarray(bias)  
        assert bias.shape == () or bias.shape == (1,) or bias.shape == self.loc.shape, f'wrong summation of multivariate_t'
        return multivariate_t(
            df = self.df,
            loc = self.loc + bias,
            shape = self.shape
        )