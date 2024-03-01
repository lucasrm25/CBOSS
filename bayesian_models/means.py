'''
Author: Lucas Rath
'''

import torch
from CBOSS.utils.torch_utils import ModuleX
from CBOSS.bayesian_models.features import PolynomialFeatures

class ZeroMean(ModuleX):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.zeros(len(X), dtype=X.dtype)