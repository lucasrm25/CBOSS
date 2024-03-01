from pathlib import Path
import unittest
import numpy as np
from pathlib import Path
import torch
import time
from typing import Union, Tuple, List, Callable, Optional
import matplotlib.pyplot as plt
from CBOSS.bayesian_models.svgp import *
from CBOSS.utils.Tee import Tee

log_dir = Path(__file__).parent / 'log'

class Test_SVGP_Classification(unittest.TestCase):
    
    def test_classification_1D(self):
                
        X1 = torch.tensor([0.150, 0.218, 0.453, 0.196, 0.638, 0.523, 0.541, 0.455, 0.632, 0.309, 0.330])
        X2 = torch.tensor([0.9, 0.868, 0.706, 0.672, 0.742, 0.813, 0.617, 0.456, 0.838, 0.730, 0.841])
        # y1 = torch.zeros_like(X1)
        y1 = - torch.ones_like(X1)
        y2 = torch.ones_like(X2)
        X = torch.concatenate([X1, X2], axis=0)[:,None]
        y = torch.concatenate([y1, y2], axis=0)
        
        N  = len(y)
        Ns = N//2
        s = torch.rand(N).argsort()[:Ns]

        X_pred = torch.linspace(0,1,300)[:,None].double()
        
        svgpc = SVGP(
            X=X, y=y, s=s,
            kernel = RBFKernel(length_scales=1.0, variance_prior=0.1),
            likelihood = BernoulliSigmoidLikelihood(),
            batch_size = N,
            optimizer='lbfgs'
        ).double()
        
        svgpc.fit(lr=0.1, maxiter=1000)
        svgpc.eval()
        f_pred = svgpc.predictive_f_posterior(X_pred)
        y_pred = svgpc.predictive_y_posterior(X_pred)
        # y_pred = f_pred
        
        fig, ax = plt.subplots()
        ax.scatter(X, (y+1)/2 , c='k', marker='x', label='data')
        ax.plot(X_pred, y_pred.mean, c='r', label='pred')
        ax.fill_between(X_pred.squeeze(), y_pred.mean - 2*y_pred.variance, y_pred.mean + 2*y_pred.variance, color='r', alpha=0.2)
        # axr = ax.twinx()
        # axr.plot(X_pred, f_pred.mean, c='tab:blue', label='pred')
        # plt.fill_between(X_pred.squeeze(), f_pred.mean - 2*f_pred.variance, f_pred.mean + 2*f_pred.variance, color='tab:blue', alpha=0.2)
        plt.legend()
        plt.title('SVGP-Classification')
        fig.savefig( log_dir / f'{Path(__file__).stem}_class_1D.png' )
        plt.close()
        
    def test_classification_banana(self, N:int=500, Ns:int=50):
        
        import pandas as pd
        
        dataset = pd.read_csv( Path(__file__).parent / 'banana.csv')
        
        dataset['At1'] = (dataset['At1'] - dataset['At1'].mean()) / dataset['At1'].std()
        dataset['At2'] = (dataset['At2'] - dataset['At2'].mean()) / dataset['At2'].std()
        
        idx = np.random.permutation(len(dataset))[:N]
        
        X = torch.tensor(dataset.iloc[idx, :-1].values).double()
        y = torch.tensor(dataset.iloc[idx, -1].values).double()
        
        s = torch.rand(N).argsort()[:Ns]

        # X_pred = torch.linspace(0,1,300)[:,None].double()
        # sample grid in [0,1]^2
        X_pred = torch.stack(torch.meshgrid(torch.linspace(-3,3,30), torch.linspace(-3,3,30), indexing='ij'), dim=-1).reshape(-1,2).double()
        
        svgpc = SVGP(
            X=X, y=y, s=s,
            kernel = RBFKernel(length_scales=1.0, variance_prior=0.1),
            likelihood = BernoulliSigmoidLikelihood(),
            batch_size = N,
            optimizer='lbfgs'
        ).double()
        
        svgpc.fit(lr=0.1, maxiter=1000)
        svgpc.eval()
        f_pred = svgpc.predictive_f_posterior(X_pred)
        y_pred = svgpc.predictive_y_posterior(X_pred)
        # y_pred = f_pred
        
        fig, ax = plt.subplots()
        ax.scatter(X[y==1,0], X[y==1,1], c='tab:blue', marker='o', label='y=+1', alpha=0.5)
        ax.scatter(X[y==-1,0], X[y==-1,1], c='tab:red', marker='o', label='y=-1', alpha=0.5)
        # plot the boundary where y_pred.mean = 0.5
        ax.contour(X_pred[:,0].reshape(30,30), X_pred[:,1].reshape(30,30), y_pred.mean.reshape(30,30), levels=[0.5], colors='k', linewidths=3)
        # plot inducing points
        ax.scatter(X[s,0], X[s,1], c='k', marker='x', label='inducing points')
        plt.legend()
        plt.title('SVGP-Classification')
        fig.savefig( log_dir / f'{Path(__file__).stem}_class_banana.png' )
        plt.close()

    
class Test_SVGP_Regression(unittest.TestCase):
    def test_regression(self):
        
        def objective_reg(x):
            # https://machinelearningmastery.com/1d-test-functions-for-function-optimization/
            return -(1.4 - 3.0 * x) * np.sin(18.0 * x)
        sigma2 = 0.2
        
        N  = 500
        Ns = 20
        X = (torch.rand(N,1) * 1.0 + 0.0)
        s = torch.rand(N).argsort()[:Ns]
        y = torch.as_tensor([objective_reg(x) for x in X])
        y_noisy = y + torch.randn(N) * sigma2

        X_pred = torch.linspace(0,1,100)[:,None].double()
        y_target = torch.as_tensor([objective_reg(x) for x in X_pred])
        
        svgpr = SVGP(
            X=X, y=y_noisy, s=s,
            kernel = RBFKernel(length_scales=0.2, variance_prior=0.5),
            likelihood = GaussianLikelihood(sigma2_prior=0.01, sigma2_hyperprior=1.),
            batch_size = N,
            optimizer = ['adam', 'lbfgs'][1]
        ).double()
        
        svgpr.fit(lr=1.0, maxiter=300)
        svgpr.eval()
        y_pred = svgpr.predictive_y_posterior(X_pred)
        
        
        fig = plt.figure()
        plt.scatter(X, y_noisy, c='k', marker='x', label='data')
        plt.plot(X_pred, y_target, c='k', label='target')
        plt.plot(X_pred, y_pred.mean, c='r', label='pred')
        plt.fill_between(X_pred.squeeze(), y_pred.mean - 2*y_pred.stddev, y_pred.mean + 2*y_pred.stddev, color='r', alpha=0.2)
        plt.legend()
        plt.title('SVGP-Regression')
        fig.savefig( log_dir / f'{Path(__file__).stem}_svgpr.png' )
        plt.close()


if __name__ == '__main__':
    with Tee( Path(__file__).parent / 'log', Path(__file__).stem) as T:
        t = unittest.main(verbosity=2, exit=False, catchbreak=True)