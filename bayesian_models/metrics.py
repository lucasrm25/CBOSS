'''
Author: Lucas Rath
'''

import os
import numpy as np
from scipy import stats
from typing import Callable
import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from pyro.distributions.multivariate_studentt import MultivariateStudentT
import scipy as sp
from CBOSS.utils.torch_utils import inputs_as_tensor

''' metrics
============================================================== '''

@inputs_as_tensor
def MAE(y_true:torch.Tensor, y_pred:torch.Tensor):
    ''' extension of sklearn.metrics.mean_absolute_error that ignores NaN values
    '''
    assert y_pred.ndim==1 and y_true.ndim==1
    idx_valid = ~ ( torch.isnan(y_true) | torch.isnan(y_pred) )
    return torch.nn.functional.l1_loss(input=y_true[idx_valid], target=y_pred[idx_valid])

@inputs_as_tensor
def NMAE(y_true:torch.Tensor, y_pred:torch.Tensor):
    ''' extension of sklearn.metrics.mean_absolute_error that ignores NaN values
    '''
    return MAE(y_true, y_pred) / torch.std(y_true[~torch.isnan(y_true)])

@inputs_as_tensor
def RMSE(y_true:torch.Tensor, y_pred:torch.Tensor):
    ''' extension of sklearn.metrics.mean_absolute_error that ignores NaN values
    '''
    assert y_pred.ndim==1 and y_true.ndim==1
    idx_valid = ~ ( torch.isnan(y_true) | torch.isnan(y_pred) )
    return torch.sqrt(torch.nn.functional.mse_loss(input=y_pred[idx_valid], target=y_true[idx_valid]))

@inputs_as_tensor
def NRMSE(y_true:torch.Tensor, y_pred:torch.Tensor):
    ''' extension of sklearn.metrics.mean_absolute_error that ignores NaN values
    '''
    return RMSE(y_true, y_pred) / torch.std(y_true[~torch.isnan(y_true)])

''' Utils
============================================================== '''

def corrcoef2(x, y, rowvar:bool=False):
    ''' Function for calculating correlation matrix between matrix A and B
    https://stackoverflow.com/a/30143754/10495567
    '''
    if not rowvar:
        x = x.T
        y = y.T
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    x_mx = x - x.mean(1)[:, None]
    y_my = y - y.mean(1)[:, None]

    # Sum of squares across rows
    ssx = (x_mx**2).sum(1)
    ssy = (y_my**2).sum(1)

    # Finally get corr coeff
    corrcoef = np.dot(x_mx, y_my.T) / np.sqrt(np.dot(ssx[:, None],ssy[None]))
    return corrcoef


