import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from pathlib import Path
import unittest

from CBOSS.utils.Tee import Tee
from CBOSS.bayesian_models.acquisitions import *
from CBOSS.bayesian_models.metrics import *
from CBOSS.bayesian_models.kernels import *
from CBOSS.bayesian_models.means import *


class Test_KernelsAndMeanFcts(unittest.TestCase):
    n_vars:int = 10
    N1:int = 100
    N2:int = 100

    def setUp(self) -> None:
        self.X1 =  torch.as_tensor(np.random.binomial(1, 0.5, [self.N1,self.n_vars]).astype(int))
        self.X2 =  torch.as_tensor(np.random.binomial(1, 0.5, [self.N2,self.n_vars]).astype(int))

    def kernel_check(self, kernel, X1, X2):
        N1,_ = X1.shape
        N2,_ = X2.shape
        K_X1X1 = kernel(X1,X1)
        K_X1X2 = kernel(X1,X2)
        self.assertTrue( torch.allclose(K_X1X1, K_X1X1.T), 'covariance matrix is not symmetric' )
        self.assertTrue( torch.linalg.eigvals(K_X1X1).real.min() + 1e-8 >= 0, 'covariance matrix is not positive definite' )
        self.assertTrue( K_X1X2.shape == (N1,N2), 'covariance matrix has wrong shape' )

    def meanfct_check(self, meanfun, X):
        self.assertTrue( meanfun(X).shape ==  (len(X),) , 'mean function outputs wrong shape')

    def test_kernels(self, disp=False):
        Kpoly = PolynomialKernel(variance_prior=0.5, degree=2).double()
        Khamm = HammingKernel(length_scales=[1.]*self.n_vars).double()
        Kdiff = DiscreteDiffusionKernel(length_scales=[1.]*self.n_vars, product_space=BinarySpace()*self.n_vars).double()
        Kprod = ProductKernel(kernelList=[Kpoly,Khamm])
        Kadd  = AdditiveKernel(kernelList=[Kpoly,Khamm])
        Kcomb = CombKernel(kernelList=[Kpoly,Khamm])

        kernels = [Kdiff, Kpoly, Khamm, Kprod, Kadd, Kcomb]

        # check kernels
        list(map( lambda kernel: self.kernel_check(kernel,self.X1,self.X2), kernels));

        if disp:
            print(Kpoly(self.X1[:2],self.X1[:2]))
            print(Khamm(self.X1[:2],self.X1[:2]))
            print(Khamm(self.X1[:2],self.X1[:2]))

    def test_meanfcts(self):
        Mzero = ZeroMean().double()

        meanfcts = [Mzero]

        list(map( lambda meanfcts: self.meanfct_check(meanfcts,self.X1), meanfcts ))

if __name__ == "__main__":

    with Tee( Path(__file__).parent / 'log', Path(__file__).stem, print_stdout=False) as T:
        t = unittest.main(
            verbosity=2, exit=False, catchbreak=True,
            # argv=['ignored', '-v', 'Test_BayesianModels.test_TorchStudentTProcessRegression']
        )