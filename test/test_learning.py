import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix
import typing as tp
from dataclasses import dataclass, field
import pandas as pd
import unittest
import warnings
import yaml
from joblib import Parallel, delayed
from glob import glob

from CBOSS.models.search_space import OneHotEncoder
import CBOSS.models.models as Models
from CBOSS.models.models import Constrained_Binary_Quad_Problem, Contamination_Control_Problem, SEIR_Model, Lorenz_Model
from CBOSS.bayesian_models.features import PolynomialFeatures
from CBOSS.utils.torch_utils import inputs_as_tensor
from CBOSS.utils.python_utils import rset, rget
from CBOSS.bayesian_models.kernels import PolynomialKernel, HammingKernel, DiscreteDiffusionKernel, CombKernel
from CBOSS.bayesian_models.regression import StudentTProcessRegression, GaussianProcessRegression, HorseShoeBayesianLinearRegression, LinearRegression
from CBOSS.bayesian_models.classification import LaplaceGaussianProcessClassification
from CBOSS.utils.Tee import Tee2

# import warnings
# warnings.filterwarnings("error")

'''
Metrics
---------------------
    Args:
        - y_true: <N>
        - y_pred_samples: <batch, N>
    Returns:
        - <batch, N>
'''

@inputs_as_tensor
def AE(y_true:torch.Tensor, y_pred_samples:torch.Tensor):
    assert y_pred_samples.ndim==1 and y_true.ndim==1
    idx_valid = ~ ( torch.isnan(y_true) | torch.isnan(y_pred_samples) )
    abs_error = torch.abs(y_true[idx_valid] - y_pred_samples[idx_valid])
    return abs_error

@inputs_as_tensor
def NAE(y_true:torch.Tensor, y_pred_samples:torch.Tensor):
    return AE(y_true=y_true, y_pred_samples=y_pred_samples) / y_true[~torch.isnan(y_true)].std()

'''
Prediction functions
---------------------
'''

def regression_MLE(X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, degree=2):
    y_test_mean = []
    y_test_std  = []

    linreg = LinearRegression(
        X = X_train, 
        y = y_train, 
        features = PolynomialFeatures(degree=degree)
    )
    # sample coefficients from the posterior
    y_test_mean = linreg.MLE_y_pred(X_pred=X_test)
    y_test_std = 0 * y_test_mean
    return y_test_mean, y_test_std

def regression_TP(
    X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, 
    degree:int=2, MMAP:bool=True, 
    kernel_type:str=['poly', 'hamm', 'diff', 'comb__poly_hamm', 'comb__poly_diff'][0],
):
    n_vars = X_train.shape[1]

    Kpoly = PolynomialKernel(variance_prior=0.5, degree=degree)
    Khamm = HammingKernel(length_scales=[1.]*n_vars)
    Kdiff = DiscreteDiffusionKernel(length_scales=[1.]*n_vars)

    # Kprod = ProductKernel(kernelList=[Kpoly,Khamm])
    # Kadd  = AdditiveKernel(kernelList=[Kpoly,Khamm])
    Kcomb__poly_hamm = CombKernel(kernelList=[Kpoly,Khamm])
    Kcomb__poly_diff = CombKernel(kernelList=[Kpoly,Kdiff])

    K = dict(poly=Kpoly, hamm=Khamm, diff=Kdiff, comb__poly_hamm=Kcomb__poly_hamm, comb__poly_diff=Kcomb__poly_diff)
    
    linreg = StudentTProcessRegression(
        X = X_train.copy(), y = y_train.copy(),
        kernel = K[kernel_type],
        train_sigma2 = True
    ).double()

    if MMAP:
        linreg.MMAP()
    linreg.eval()

    # prediction in chunks to avoid memory overflow due to large covariance matrices
    y_post_mean = []
    y_post_std  = []
    # y_post_samples = []
    for X_test_split in torch.split( torch.as_tensor(X_test), 500):
        # NOTE: if we sample from the posterior, then we have to calculate the whole cov. matrix
        y_post = linreg.predictive_posterior(X_pred=torch.as_tensor(X_test_split), diagonal=True, include_noise=False)
        # y_post_samples += [y_post.sample((nbr_samples,))]
        y_post_mean += y_post.mean.tolist()
        y_post_std  += y_post.stddev.tolist()

    return np.array(y_post_mean), np.array(y_post_std)
    # y_post_samples = torch.cat(y_post_samples, -1)
    # return y_post_samples

def regression_GP(
    X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, 
    degree:int=2, MMAP:bool=True, 
    kernel_type:str=['poly', 'hamm', 'diff', 'comb__poly_hamm', 'comb__poly_diff'][0],
):
    n_vars = X_train.shape[1]

    Kpoly = PolynomialKernel(variance_prior=0.5, degree=degree)
    Khamm = HammingKernel(length_scales=[1.]*n_vars)
    Kdiff = DiscreteDiffusionKernel(length_scales=[1.]*n_vars)

    # Kprod = ProductKernel(kernelList=[Kpoly,Khamm])
    # Kadd  = AdditiveKernel(kernelList=[Kpoly,Khamm])
    Kcomb__poly_hamm = CombKernel(kernelList=[Kpoly,Khamm])
    Kcomb__poly_diff = CombKernel(kernelList=[Kpoly,Kdiff])

    K = dict(poly=Kpoly, hamm=Khamm, diff=Kdiff, comb__poly_hamm=Kcomb__poly_hamm, comb__poly_diff=Kcomb__poly_diff)
    
    linreg = GaussianProcessRegression(
        X = X_train.copy(), y = y_train.copy(),
        kernel = K[kernel_type],
        train_sigma2 = True
    ).double()

    if MMAP:
        linreg.MMAP()
    linreg.eval()

    # prediction in chunks to avoid memory overflow due to large covariance matrices
    y_post_mean = []
    y_post_std  = []
    # y_post_samples = []
    for X_test_split in torch.split( torch.as_tensor(X_test), 500):
        # NOTE: if we sample from the posterior, then we have to calculate the whole cov. matrix
        y_post = linreg.predictive_posterior(X_pred=torch.as_tensor(X_test_split), diagonal=True)
        # y_post_samples += [y_post.sample((nbr_samples,))]
        y_post_mean += y_post.mean.tolist()
        y_post_std  += y_post.stddev.tolist()

    return np.array(y_post_mean), np.array(y_post_std)
    # y_post_samples = torch.cat(y_post_samples, -1)
    # return y_post_samples

def regression_HS(X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, nbr_samples:int=50, degree:int=2):
    linreg_HS = HorseShoeBayesianLinearRegression(
        X = X_train, 
        y = y_train, 
        features = PolynomialFeatures(degree=degree)
    )
    # sample coefficients from the posterior
    alpha_post_samples = linreg_HS.posterior_sample_alpha(burnin=100, thin=10, nsamples=nbr_samples)
    y_post_samples = np.vstack([
        linreg_HS.y_mean(X=X_test, alpha_mean=alpha)
        for alpha in alpha_post_samples
    ])
    # denormalize predictions
    return y_post_samples.mean(0), y_post_samples.std(0)
    # return y_post_samples    # <nbr_samples, N>

def classification_GP(
    X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, 
    degree:int=2,
    kernel_type:str=['poly', 'hamm', 'diff', 'comb__poly_hamm', 'comb__poly_diff'][0],
):
    n_vars = X_train.shape[1]

    Kpoly = PolynomialKernel(variance_prior=0.01, degree=degree)
    Khamm = HammingKernel(length_scales=[1.]*n_vars)
    Kdiff = DiscreteDiffusionKernel(length_scales=[1.]*n_vars)

    # Kprod = ProductKernel(kernelList=[Kpoly,Khamm])
    # Kadd  = AdditiveKernel(kernelList=[Kpoly,Khamm])
    Kcomb__poly_hamm = CombKernel(kernelList=[Kpoly,Khamm])
    Kcomb__poly_diff = CombKernel(kernelList=[Kpoly,Kdiff])

    K = dict(poly=Kpoly, hamm=Khamm, diff=Kdiff, comb__poly_hamm=Kcomb__poly_hamm, comb__poly_diff=Kcomb__poly_diff)

    gpc = LaplaceGaussianProcessClassification(
        X = X_train.copy(), y = y_train.copy(),
        kernel = K[kernel_type],
    ).float()

    gpc.fit(lr=0.1, maxiter=1)
    gpc.eval()

    # prediction in chunks to avoid memory overflow due to large covariance matrices
    y_post_mean = []
    for X_test_split in torch.split( torch.as_tensor(X_test), 500):
        # NOTE: if we sample from the posterior, then we have to calculate the whole cov. matrix
        y_post = gpc.predictive_y_posterior(X_pred=torch.as_tensor(X_test_split))
        # y_post_samples += [y_post.sample((nbr_samples,))]
        y_post_mean += y_post.tolist()

    return np.array(y_post_mean)
    # y_post_samples = torch.cat(y_post_samples, -1)
    # return y_post_samples

'''
Interfaces
------------
'''

@dataclass()
class Reg_Metrics:
    fun: tp.Callable
    kwargs: tp.Optional[tp.Dict] = None
    name: str = ""
    mean:list = field(default_factory=list)
    std:list  = field(default_factory=list)

@dataclass()
class Reg_Model:
    fun: tp.Callable
    kwargs: tp.Optional[tp.Dict] = None
    name: str = ""
    plot_kwargs: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)

@dataclass()
class Model:
    name:str
    X_train:list = field(default_factory=list)
    y_train:list = field(default_factory=list)
    X_test:list  = field(default_factory=list)
    y_test:list  = field(default_factory=list)
    # 
    y_reg_models:list = field(default_factory=list)

class ITest_Reg_with_Model(unittest.TestCase): # 

    exp_name:str       = 'TEST_y'
    train_sizes:list   = np.linspace(50,6000, 12, dtype=int) # range(50, 100+1, 50) #range(50, 400+1, 50)
    test_size:int      = 100

    pred_models = [
        Reg_Model(fun=regression_TP, name='TPR_Kcomb_Kpoly2_Kdiff', plot_kwargs=dict(label='TPReg (Kcomb[Kpoly(deg=2), Kdiff])', color='tab:pink',   linestyle='-'), kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_diff')),
        Reg_Model(fun=regression_TP, name='TPR_Kcomb_Kpoly2_Khamm', plot_kwargs=dict(label='TPReg (Kcomb[Kpoly(deg=2), Khamm])', color='tab:green',  linestyle='-'), kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_hamm')),
        Reg_Model(fun=regression_TP, name='TPR_Kdiff',              plot_kwargs=dict(label='TPReg (Kdiff)',                      color='tab:brown',  linestyle='-'), kwargs=dict(degree=2, MMAP=True, kernel_type='diff')),
        Reg_Model(fun=regression_TP, name='TPR_Khamm',              plot_kwargs=dict(label='TPReg (Khamm)',                      color='tab:orange', linestyle='-'), kwargs=dict(degree=2, MMAP=True, kernel_type='hamm')),
        Reg_Model(fun=regression_TP, name='TPR_Kpoly2',             plot_kwargs=dict(label='TPReg (Kpoly(deg=2))',               color='tab:blue',   linestyle='-'), kwargs=dict(degree=2, MMAP=True, kernel_type='poly')),
        Reg_Model(fun=regression_HS, name='HorseShoe_BLR2',         plot_kwargs=dict(label='HorseShoe BLR (deg=2)',              color='tab:purple', linestyle='-'), kwargs=dict(degree=2, nbr_samples=50)),
    ]

    reg_metrics = [
        Reg_Metrics(fun=AE,   name='MAE'),
        Reg_Metrics(fun=NAE,  name='NMAE'),
        # LinReg_Metrics(fun=RSE,  name='RMSE'),
        # LinReg_Metrics(fun=NRSE, name='NRMSE'),
    ]

    X_train:np.ndarray = None
    y_train:np.ndarray = None
    y_test:np.ndarray  = None

    def setUp(self):
        sns.set_style(style="darkgrid")

    def test_linreg_with_model(self):

        ''' Predict and evalaute metrics for each model
        '''
        for pm in self.pred_models:
            for metric in self.reg_metrics:
                pm.metrics[metric.name] = []

            for train_size in tqdm(self.train_sizes, desc=f'Processing {pm.name}'):

                X_train = self.X_train[:train_size]
                y_train = self.y_train[:train_size]

                res = pm.fun(X_train=X_train, y_train=y_train, X_test=self.X_test, **pm.kwargs)
                # pm.preds += [res]
                pm.means += [res[0]]
                pm.stds  += [res[1]]

                for metric in self.reg_metrics:
                    pm.metrics[metric.name] += [metric.fun( y_true=self.y_test, y_pred_samples=pm.means[-1])]
                    print(f'{pm.name} - trsize({train_size: 5d}) - {metric.name}: {pm.metrics[metric.name][-1].mean():.3f} +- {pm.metrics[metric.name][-1].std():.3f}')


        ''' Plot prediction ERROR vs Number of training samples
        -------------------------------------------------------------------------'''

        rg = self.train_sizes[-1] - self.train_sizes[0]
        jitter = np.linspace(-0.02*rg, 0.02*rg, len(self.pred_models)).reshape(-1,1)

        for m in self.reg_metrics:
            fig = plt.figure()
            for ipm, pm in enumerate(reversed(self.pred_models)):
                means = np.array([v.mean().numpy() for v in pm.metrics[m.name]])
                stds  = np.array([v.std().numpy()  for v in pm.metrics[m.name]])
                plt.plot(
                    self.train_sizes, means, 
                    c=pm.color, label=pm.name, ls=pm.linstyle,
                    lw=2, ms=3, marker=MarkerStyle('s', 'none'), markeredgewidth=1, 
                )
                # plt.errorbar( 
                    # x=self.train_sizes + jitter[ipm], y=means, yerr=stds*0,
                    # c=pm.color, label=pm.name, ls=pm.linstyle,
                    # lw=2, ms=6, capsize=3, capthick=3, marker=MarkerStyle('s', 'none'), markeredgewidth=2, 
                # )
            plt.grid(True)
            plt.legend(prop={'size':6})
            plt.title(f'{m.name} test performance vs number of training samples')
            plt.xlabel('Number of training samples')
            plt.ylabel(f'{m.name}')
            plt.tight_layout()
            fig.savefig( Path(__file__).parent / 'log' / f'{Path(__file__).stem}_{self.exp_name}__{m.name}_x_NbrSamples.png', dpi=300 )
            plt.close()


        ''' Plot violin plot for the prediction ERROR
        -------------------------------------------------------------------------'''

        for m in self.reg_metrics:
            # df = pd.DataFrame(data={
            #     pm.name: pm.metrics[m.name][-1] for pm in self.pred_models
            # })
            df = pd.DataFrame(data={
                pm.name: pm.means[-1] - self.y_test for pm in self.pred_models
            })

            fig = plt.figure(figsize=(8,8))
            sns.violinplot(df, orient='h', cut=0, scale="count")
            plt.title(f'Violin plot for test prediction error\nTraining size = {max(self.train_sizes)}')
            plt.tight_layout()
            fig.savefig( Path(__file__).parent / 'log' / f'{Path(__file__).stem}_{self.exp_name}__{m.name}_violin.png', dpi=300 )
            plt.close()


        ''' Plot prediction test predictions for training size = tr_size_plot
        -------------------------------------------------------------------------'''
        nsamples = min(30,len(self.y_test))
        idx_valid = ~np.isnan(self.y_test)
        idx_eval = np.where( idx_valid )[0][:nsamples]

        x_vals = np.arange(nsamples) + np.linspace(-0.2,0.2,len(self.pred_models)).reshape(-1,1)
        # pd.DataFrame(
        #     data = {
        #         pm.name: pm.means[-1][:nsamples]
        #         for pm in self.pred_models
        #     }
        # )

        fig = plt.figure(figsize=(14,6))
        plt.scatter(range(nsamples), self.y_test[idx_eval], c='k', s=10**2, marker=MarkerStyle('o', 'none'), label='True values')
        for i, pm in enumerate(reversed(self.pred_models)):
            plt.errorbar( 
                x=x_vals[i], y=pm.means[-1][idx_eval], yerr=pm.stds[-1][idx_eval],
                c=pm.color, label=pm.name,
                ls="", lw=1, ms=6, capsize=3, capthick=3, marker=MarkerStyle('o', 'full'), markeredgewidth=2, 
            )
        plt.grid(True)
        plt.legend(prop={'size':6})
        plt.title(f'Test predictions for training size = {max(self.train_sizes)}')
        plt.xlabel('Test sample number')
        plt.ylabel(f'Predictions')
        plt.tight_layout()
        fig.savefig( Path(__file__).parent / 'log' / f'{Path(__file__).stem}_{self.exp_name}__test_preds', dpi=300 )
        plt.close()

class ITest_Class_with_Model(unittest.TestCase): # 

    exp_name:str       = 'TEST_l'
    train_sizes:list   = np.linspace(200,2000, 4, dtype=int) # range(50, 100+1, 50) #range(50, 400+1, 50)
    test_size:int      = 6000

    pred_models = [
        Reg_Model(fun=classification_GP, name='CGP (deg=2)', plot_kwargs=dict(color='tab:black'), kwargs=dict(degree=2)),
    ]

    X_train:np.ndarray = None
    y_train:np.ndarray = None
    y_test:np.ndarray  = None

    def setUp(self):
        sns.set_style(style="darkgrid")

    def test_linclass_with_model(self):

        ''' Predict and evalaute metrics for each model
        '''
        for pm in self.pred_models:
            
            pm.metrics = dict(
                MAE=[], Recall=[], Precision=[], Accuracy=[], Balance=[]
            )

            for train_size in tqdm(self.train_sizes, desc=f'Processing {pm.name}'):

                X_train = self.X_train[:train_size]
                y_train = self.y_train[:train_size]

                res = pm.fun(X_train=X_train, y_train=y_train, X_test=self.X_test, **pm.kwargs)
                # pm.preds += [res]
                pm.means += [res]

                CM = confusion_matrix(y_true=self.y_test.astype(int), y_pred=(res >= 0.5).astype(int), normalize='all')
                tn,fp,fn,tp = CM.ravel()
                pm.metrics['MAE']       += [np.abs(self.y_test - res).mean()]
                pm.metrics['Recall']    += [tp / (tp + fn)]
                pm.metrics['Precision'] += [tp / (tp + fp)]
                pm.metrics['Accuracy']  += [(tp + tn) / (tn + fp + fn + tp)]
                pm.metrics['Balance']   += [(tp + fp) / (tn + fp + fn + tp)]

                for k, v in pm.metrics.items():
                    print(f'{pm.name} - trsize({train_size: 5d}) - {k:10s}: {v[-1]:.3f}')


        ''' Plot prediction ERROR vs Number of training samples
        -------------------------------------------------------------------------'''

        # rg = self.train_sizes[-1] - self.train_sizes[0]
        # jitter = np.linspace(-0.02*rg, 0.02*rg, len(self.pred_models)).reshape(-1,1)

        metrics = list(self.pred_models[0].metrics.keys())

        for m in metrics:
            fig = plt.figure()
            for ipm, pm in enumerate(self.pred_models):
                means = np.array(pm.metrics[m])
                plt.plot(
                    self.train_sizes, means, 
                    c=pm.color, label=pm.name, ls=pm.linstyle,
                    lw=1, ms=3, marker=MarkerStyle('^', 'none'), markeredgewidth=1, 
                )

            plt.grid(True)
            plt.legend(prop={'size':6})
            plt.title(f'{m} test performance vs number of training samples')
            plt.xlabel('Number of training samples')
            plt.ylabel(f'{m}')
            plt.ylim([0,1])
            plt.tight_layout()
            fig.savefig( Path(__file__).parent / 'log' / f'{Path(__file__).stem}_{self.exp_name}__{m}_x_NbrSamples.png', dpi=300 )
            plt.close()


        ''' Plot violin plot for the prediction ERROR
        -------------------------------------------------------------------------'''
        
        metrics = list(self.pred_models[0].metrics.keys())

        for m in metrics:
            # df = pd.DataFrame(data={
            #     pm.name: pm.metrics[m.name][-1] for pm in self.pred_models
            # })
            df = pd.DataFrame(data={
                pm.name: pm.means[-1] - self.y_test for pm in self.pred_models
            })

            fig = plt.figure(figsize=(8,8))
            sns.violinplot(df, orient='h', cut=0, scale="count")
            plt.title(f'Violin plot for test prediction error\nTraining size = {max(self.train_sizes)}')
            plt.tight_layout()
            fig.savefig( Path(__file__).parent / 'log' / f'{Path(__file__).stem}_{self.exp_name}__{m}_violin.png', dpi=300 )
            plt.close()


        ''' Plot prediction test predictions for training size = tr_size_plot
        -------------------------------------------------------------------------'''
        nsamples = min(30,len(self.y_test))
        idx_valid = ~np.isnan(self.y_test)
        idx_eval = np.where( idx_valid )[0][:nsamples]

        x_vals = np.arange(nsamples) + 0*np.linspace(-0.2,0.2,len(self.pred_models)).reshape(-1,1)

        fig = plt.figure(figsize=(14,6))
        plt.scatter(range(nsamples), self.y_test[idx_eval], c='k', s=10**2, marker=MarkerStyle('o', 'none'), label='True values')
        for i, pm in enumerate(self.pred_models):
            plt.plot( 
                x_vals[i], pm.means[-1][idx_eval],
                c=pm.color, label=pm.name,
                ls="", lw=1, ms=6, marker=MarkerStyle('o', 'none'), markeredgewidth=2, 
            )
        plt.grid(True)
        plt.legend(prop={'size':6})
        plt.title(f'Test predictions for training size = {max(self.train_sizes)}')
        plt.xlabel('Test sample number')
        plt.ylabel(f'Predictions')
        plt.tight_layout()
        fig.savefig( Path(__file__).parent / 'log' / f'{Path(__file__).stem}_{self.exp_name}__test_preds', dpi=300 )
        plt.close()


'''
Test classes
--------------
'''

class Test_Learning_EquationDiscovery(unittest.TestCase): # 

    create_dataset_from_scratch:bool    = False
    train_from_scratch:bool             = False
    reruns:int          = 10
    nbr_jobs:int        = 10

    exp_name:str        = 'TEST_y'
    train_sizes:list    = np.linspace(50,1000, 10, dtype=int) # range(50, 100+1, 50) #range(50, 400+1, 50)
    test_size:int       = 50000

    cfg_file = Path(__file__).parent /'..'/'..'/'configs_equation_discovery.yaml'
    log_dir  = Path(__file__).parent / 'log'
    
    model_names = [
        'NonLinearDampedOscillator_k5',
        'CylinderWake_k3',
        'Lorenz_k3',
        'SEIR_k3',
        'ChuaOscillator_k3',
    ]

    pred_models = [
        Reg_Model(fun=regression_TP, name='TPR_Kcomb_Kpoly2_Kdiff', plot_kwargs=dict(label='TPReg (Kcomb[Kpoly(deg=2), Kdiff])', color='tab:pink',   linestyle='-'), kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_diff')),
        Reg_Model(fun=regression_TP, name='TPR_Kcomb_Kpoly2_Khamm', plot_kwargs=dict(label='TPReg (Kcomb[Kpoly(deg=2), Khamm])', color='tab:green',  linestyle='-'), kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_hamm')),
        Reg_Model(fun=regression_TP, name='TPR_Kdiff',              plot_kwargs=dict(label='TPReg (Kdiff)',                      color='tab:brown',  linestyle='-'), kwargs=dict(degree=2, MMAP=True, kernel_type='diff')),
        Reg_Model(fun=regression_TP, name='TPR_Khamm',              plot_kwargs=dict(label='TPReg (Khamm)',                      color='tab:orange', linestyle='-'), kwargs=dict(degree=2, MMAP=True, kernel_type='hamm')),
        Reg_Model(fun=regression_TP, name='TPR_Kpoly2',             plot_kwargs=dict(label='TPReg (Kpoly(deg=2))',               color='tab:blue',   linestyle='-'), kwargs=dict(degree=2, MMAP=True, kernel_type='poly')),
        Reg_Model(fun=regression_HS, name='HorseShoe_BLR2',         plot_kwargs=dict(label='HorseShoe BLR (deg=2)',              color='tab:purple', linestyle='-'), kwargs=dict(degree=2, nbr_samples=50)),
    ] + [
        Reg_Model(fun=regression_GP, name='GPR_Kcomb_Kpoly2_Kdiff', plot_kwargs=dict(label='GPReg (Kcomb[Kpoly(deg=2), Kdiff])', color='tab:pink',  linestyle='--'), kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_diff')),
        Reg_Model(fun=regression_GP, name='GPR_Kcomb_Kpoly2_Khamm', plot_kwargs=dict(label='GPReg (Kcomb[Kpoly(deg=2), Khamm])', color='tab:green', linestyle='--'), kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_hamm')),
        Reg_Model(fun=regression_GP, name='GPR_Kdiff',              plot_kwargs=dict(label='GPReg (Kdiff)',                      color='tab:brown', linestyle='--'), kwargs=dict(degree=2, MMAP=True, kernel_type='diff')),
        Reg_Model(fun=regression_GP, name='GPR_Khamm',              plot_kwargs=dict(label='GPReg (Khamm)',                      color='tab:orange',linestyle='--'), kwargs=dict(degree=2, MMAP=True, kernel_type='hamm')),
        Reg_Model(fun=regression_GP, name='GPR_Kpoly2',             plot_kwargs=dict(label='GPReg (Kpoly(deg=2))',               color='tab:blue',  linestyle='--'), kwargs=dict(degree=2, MMAP=True, kernel_type='poly')),
        
    ]

    reg_metrics = [
        Reg_Metrics(fun=AE,   name='MAE'),
        Reg_Metrics(fun=NAE,  name='NMAE'),
        # LinReg_Metrics(fun=RSE,  name='RMSE'),
        # LinReg_Metrics(fun=NRSE, name='NRMSE'),
    ]

    X_train:np.ndarray = None
    y_train:np.ndarray = None
    y_test:np.ndarray  = None

    def setUp(self):
        sns.set_style(style="darkgrid")

        with open(self.cfg_file, 'r') as f:
            self.cfg = yaml.safe_load(f)

    def _generate_train_test_data(self, model, fac:int=5):
        
        encoder = OneHotEncoder(product_space=model.product_space)

        N_train = max(self.train_sizes)
        N_train_rr = N_train * self.reruns
        N_test  = self.test_size
        N = ( N_train_rr + N_test ) * fac
        X = model.sample(N).astype(int)

        print('\tevaluating...')
        y, c, l = model.evaluate(X=X, verbose=True)
        print('\tevaluating...done!')

        # encode
        Xe = encoder.encode(X=X).astype(int)

        idx_success,_ = np.where(l == 1)
        assert len(idx_success) >= N_train_rr + N_test, 'there are too many failures, please increase the number of samples'

        def _select_data(X, target):
            # select data for target
            idx_valid,_     = np.where(~np.isnan(target))
            X_train         = X[idx_valid][:N_train_rr]
            target_train    = target[idx_valid][:N_train_rr]
            X_test          = X[idx_valid][-N_test:]
            target_test     = target[idx_valid][-N_test:]
            return X_train, target_train, X_test, target_test

        def _normalize_data(X_train, X_test, target_train, target_test):
            # normalize data
            target_train_std, target_train_mean = np.nanstd(target_train), np.nanmean(target_train)
            # save results
            target_train_norm = (target_train.squeeze()  - target_train_mean) / target_train_std
            target_test_norm  = (target_test.squeeze()  - target_train_mean) / target_train_std
            return target_train_norm, target_test_norm


        Xe_l_train, l_train, Xe_l_test, l_test = _select_data(X=Xe, target=l)
        Xe_l_train = np.split( Xe_l_train, self.reruns )
        l_train    = np.split( l_train, self.reruns )

        Xe_c_train, c_train, Xe_c_test, c_test = _select_data(X=Xe, target=c)
        c_train_norm, c_test_norm = _normalize_data(
            X_train=Xe_c_train, X_test=Xe_c_test, target_train=c_train, target_test=c_test
        )
        Xe_c_train   = np.split( Xe_c_train, self.reruns )
        c_train_norm = np.split( c_train_norm, self.reruns )

        Xe_y_train, y_train, Xe_y_test, y_test = _select_data(X=Xe, target=y)
        y_train_norm, y_test_norm = _normalize_data(
            X_train=Xe_y_train, X_test=Xe_y_test, target_train=y_train, target_test=y_test
        )
        Xe_y_train   = np.split( Xe_y_train, self.reruns )
        y_train_norm = np.split( y_train_norm, self.reruns )

        dataset = {
            'y': {
                'train': {'X':Xe_y_train, 'target':y_train_norm},
                'test':  {'X':Xe_y_test,  'target':y_test_norm},
            },
            'c': {
                'train': {'X':Xe_c_train, 'target':c_train_norm},
                'test':  {'X':Xe_c_test,  'target':c_test_norm},
            },
            'l': {
                'train': {'X':Xe_l_train, 'target':l_train},
                'test':  {'X':Xe_l_test,  'target':l_test},
            }
        }

        return dataset

    def load_dataset(self, model_name:str):
        model_dir = self.log_dir / f'models_{model_name}'
        model_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = model_dir / 'dataset.pkl'

        print(f'\n---> Processing {model_name}\n')

        Model = getattr(Models, self.cfg[model_name]['model_name'])
        randomstate = np.random.RandomState(self.cfg[model_name]['model']['random_seed_init_dataset'])

        if self.create_dataset_from_scratch:
            print('\tstarting model...')
            model = Model(**self.cfg[model_name]['model']['kwargs'], random_generator=randomstate)
            print('\tstarting model...done!')

            print('\tgenerating dataset...')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dataset = self._generate_train_test_data(model=model)
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset,file=f)
            print('\tgenerating dataset....done!')
        else:
            print('\treading dataset...')
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(file=f)
            print('\treading dataset...done')
        
        return dataset

    def test_linreg_with_model(self):

        ''' Predict and evalaute metrics for each model
        '''
        datasets = dict()

        ''' Read / Create Dataset for each model
        '''

        with Parallel(n_jobs=self.nbr_jobs, prefer=["threads","processes"][1], backend='loky') as parallel:


            ''' Read dataset 
            ===================
                datasets is a dictionary with the following structure
                ```
                datasets = {
                    MODEL_NAME: {
                        'y': {
                            'train': {
                                'X': np.ndarray,
                                'target': np.ndarray
                            },
                            'test': {
                                'X': np.ndarray,
                                'target': np.ndarray
                            }
                        }
                    }
                }
                ```
            '''
            res = parallel(
                delayed(self.load_dataset)(model_name=model_name)
                for model_name in tqdm(self.model_names)
            )
            datasets = {k:v for k,v in zip(self.model_names, res)}


            ''' Run models 
            ===================
            
                Dictionary containing prediction results for 
                    each opt problem (MODEL_NAME)
                    each model (PREDICTION_MODEL_NAME)
                    each rerun (RERUN_NUMBER)
                    each train size (TRAIN_SIZE)
                
                are saved to `LOG_DIR / MODEL_NAME / PREDICTION_MODEL_NAME.y.RERUN_NUMBER.pkl'`
                
                Each file contains a dictionary with the following structure
                ```
                    {
                        'y': {
                            'test': {
                                TRAIN_SIZE: {
                                    pred_mean : np.ndarray,
                                    pred_std : np.ndarray,
                                    target : np.ndarray,
                                    MAE: np.ndarray
                                }
                            }
                        }
                    }
                ```
            '''
            
            def run_model(i, X_train, target_train, dataset, pm):
                
                run_preds = {'y':{'test':{}}}
                
                X_test         = dataset['y']['test']['X']
                target_test    = dataset['y']['test']['target']
                
                # for each prediction sizes
                for train_size in tqdm(self.train_sizes, desc=f'Processing {model_name} - {pm.name}'):

                    X_train_i      = X_train[:train_size]
                    target_train_i = target_train[:train_size]
                    res = pm.fun(
                        X_train=X_train_i, 
                        y_train=target_train_i, 
                        X_test=X_test, 
                        **pm.kwargs
                    )
                    run_preds['y']['test'][train_size] = dict(
                        pred_mean   = res[0],
                        pred_std    = res[1],
                        target      = target_test,
                        MAE         = np.abs(res[0] - target_test).mean() / target_test.std(),
                    )
                    
                    print(f"train_size: {train_size} MAE: {run_preds['y']['test'][train_size]['MAE']:.3f} ")
                
                MAEs = [run_preds["y"]["test"][train_size]["MAE"] for train_size in self.train_sizes]
                print(f'{pm.name} results: {MAEs}')
                
                res_file = self.log_dir / f'models_{model_name}' / f'{pm.name}.y.{i}.pkl'
                with open(res_file, 'wb') as f:
                    pickle.dump(run_preds, file=f)
            

            results = {k:dict() for k in self.model_names}
            # for each model = optimization problem
            for model_name, dataset in datasets.items():
                # for each prediction model
                for pm in self.pred_models:
                    
                    # for each rerun: train prediction model with different trainining sizes
                    if self.train_from_scratch:
                        res = parallel(
                            delayed(run_model)(
                                i=i, X_train=X_train, target_train=target_train, dataset=dataset, pm=pm
                            )
                            # for each rerun
                            for i, (X_train, target_train) in enumerate(zip(dataset['y']['train']['X'], dataset['y']['train']['target']))
                        )
                    
                    # read saved results
                    results[model_name][pm.name] = []
                    res_files = glob(str(self.log_dir / f'models_{model_name}' / f'{pm.name}.y.*.pkl'))
                    for res_file in tqdm(res_files, desc=f'Reading results {model_name} {pm.name}'):
                        with open(file=res_file, mode='rb') as f:
                            res = pickle.load(file=f) 
                        results[model_name][pm.name] += [res]


        ''' Plot prediction ERROR vs Number of training samples
        -------------------------------------------------------------------------'''

        # fig, axs = plt.subplots(1,len(self.model_names), figsize=(20,4))
        fig, axs = plt.subplots(1,len(self.model_names), figsize=(15,4))
        
        # for each model = optimization problem
        for i, model_name in enumerate(self.model_names):
            
            ax = axs[i]
            
            # for each prediction model
            for pm in self.pred_models[::-1]:

                MAE_mean = {
                    tr_size: (
                        np.array([
                            res['y']['test'][tr_size]['MAE']
                            # np.abs(res['y']['test'][tr_size]['pred_mean'] - res['y']['test'][tr_size]['target']).mean() / 
                            # res['y']['test'][tr_size]['target'].std()
                            for res in results[model_name][pm.name]
                        ]).mean()
                    )
                    for tr_size in self.train_sizes
                }

                ax.plot( MAE_mean.keys(), MAE_mean.values(), **pm.plot_kwargs )
            ax.grid(True)
            ax.legend(prop={'size':6})
            ax.set_xlabel('Number of training samples')
            ax.set_title(f'{model_name}')
        axs[0].set_ylabel(f'MAE')
        plt.suptitle(f'MAE test performance vs number of training samples')
        plt.tight_layout()
        fig.savefig( Path(__file__).parent / 'log' / f'{Path(__file__).stem}__y_MAE_x_NbrSamples.png', dpi=300 ) # f'models_{model_name}' 
        plt.close()


if __name__ == "__main__":

    t = Test_Learning_EquationDiscovery()
    t.setUp()
    t.test_linreg_with_model()

    # with (  
    #     open((Path(__file__).parent/'log'/Path(__file__).stem).with_suffix('.log'), 'w') as fstdout, 
    #     Tee2(stdout_streams=[fstdout, sys.stdout]
    #     ) as T
    # ):
    #     t = unittest.main(
    #         verbosity=2, exit=False, catchbreak=True, 
    #         # argv=['ignored', '-v', 'Test_LinReg_with_Contamination_Control'] #! @@@@@@@@@@@@@@@@@@@@@@@    argv
    #         argv=['ignored', '-v', 'Test_LinReg_with_SEIR_y']
    #     )



        # rg = self.train_sizes[-1] - self.train_sizes[0]
        # jitter = np.linspace(-0.02*rg, 0.02*rg, len(self.pred_models)).reshape(-1,1)

        # for m in self.reg_metrics:
        #     fig = plt.figure()
        #     for ipm, pm in enumerate(reversed(self.pred_models)):
        #         means = np.array([v.mean().numpy() for v in pm.metrics[m.name]])
        #         stds  = np.array([v.std().numpy()  for v in pm.metrics[m.name]])
        #         plt.plot(
        #             self.train_sizes, means, 
        #             c=pm.color, label=pm.name, ls=pm.linstyle,
        #             lw=2, ms=3, marker=MarkerStyle('s', 'none'), markeredgewidth=1, 
        #         )
        #         # plt.errorbar( 
        #             # x=self.train_sizes + jitter[ipm], y=means, yerr=stds*0,
        #             # c=pm.color, label=pm.name, ls=pm.linstyle,
        #             # lw=2, ms=6, capsize=3, capthick=3, marker=MarkerStyle('s', 'none'), markeredgewidth=2, 
        #         # )
        #     plt.grid(True)
        #     plt.legend(prop={'size':6})
        #     plt.title(f'{m.name} test performance vs number of training samples')
        #     plt.xlabel('Number of training samples')
        #     plt.ylabel(f'{m.name}')
        #     plt.tight_layout()
        #     fig.savefig( Path(__file__).parent / 'log' / f'{Path(__file__).stem}_{self.exp_name}__{m.name}_x_NbrSamples.png', dpi=300 )
        #     plt.close()


        # ''' Plot violin plot for the prediction ERROR
        # -------------------------------------------------------------------------'''

        # for m in self.reg_metrics:
        #     # df = pd.DataFrame(data={
        #     #     pm.name: pm.metrics[m.name][-1] for pm in self.pred_models
        #     # })
        #     df = pd.DataFrame(data={
        #         pm.name: pm.means[-1] - self.y_test for pm in self.pred_models
        #     })

        #     fig = plt.figure(figsize=(8,8))
        #     sns.violinplot(df, orient='h', cut=0, scale="count")
        #     plt.title(f'Violin plot for test prediction error\nTraining size = {max(self.train_sizes)}')
        #     plt.tight_layout()
        #     fig.savefig( Path(__file__).parent / 'log' / f'{Path(__file__).stem}_{self.exp_name}__{m.name}_violin.png', dpi=300 )
        #     plt.close()


        # ''' Plot prediction test predictions for training size = tr_size_plot
        # -------------------------------------------------------------------------'''
        # nsamples = min(30,len(self.y_test))
        # idx_valid = ~np.isnan(self.y_test)
        # idx_eval = np.where( idx_valid )[0][:nsamples]

        # x_vals = np.arange(nsamples) + np.linspace(-0.2,0.2,len(self.pred_models)).reshape(-1,1)
        # # pd.DataFrame(
        # #     data = {
        # #         pm.name: pm.means[-1][:nsamples]
        # #         for pm in self.pred_models
        # #     }
        # # )

        # fig = plt.figure(figsize=(14,6))
        # plt.scatter(range(nsamples), self.y_test[idx_eval], c='k', s=10**2, marker=MarkerStyle('o', 'none'), label='True values')
        # for i, pm in enumerate(reversed(self.pred_models)):
        #     plt.errorbar( 
        #         x=x_vals[i], y=pm.means[-1][idx_eval], yerr=pm.stds[-1][idx_eval],
        #         c=pm.color, label=pm.name,
        #         ls="", lw=1, ms=6, capsize=3, capthick=3, marker=MarkerStyle('o', 'full'), markeredgewidth=2, 
        #     )
        # plt.grid(True)
        # plt.legend(prop={'size':6})
        # plt.title(f'Test predictions for training size = {max(self.train_sizes)}')
        # plt.xlabel('Test sample number')
        # plt.ylabel(f'Predictions')
        # plt.tight_layout()
        # fig.savefig( Path(__file__).parent / 'log' / f'{Path(__file__).stem}_{self.exp_name}__test_preds', dpi=300 )
        # plt.close()



# class Test_LinReg_with_BQP_y(ITest_Reg_with_Model): #
#     exp_name:str        = 'BQP_y' 
#     n_vars:int          = 20
#     alpha_f             = 10**2
#     lambda_f            = 1e-3
#     train_sizes:list    = np.linspace(50,400, 16, dtype=int) # range(50, 100+1, 50) #range(50, 400+1, 50)
#     test_size:int       = 100

#     def setUp(self):
#         ''' Setup BQP and init linear regressors
#         '''
#         super().setUp()

#         self.model = Constrained_Binary_Quad_Problem(n_vars=self.n_vars, alpha_f=self.alpha_f, lambda_f=self.lambda_f, alpha_c=self.alpha_f, lambda_c=self.lambda_f)
#         encoder = OneHotEncoder(product_space=self.model.product_space)
#         # evaluate train
#         Xc_train = self.model.sample(nbr_samples=max(self.train_sizes))
#         X_train = encoder.encode(X=Xc_train).astype(int)
#         y_train,c_train,l_train = self.model.evaluate(X=X_train)
#         # evaluate test
#         Xc_test = self.model.sample(nbr_samples=self.test_size)
#         X_test = encoder.encode(X=Xc_test).astype(int)
#         y_test,c_test,l_test = self.model.evaluate(X=X_test)

#         # normalize data
#         y_train_std, y_train_mean = np.nanstd(y_train), np.nanmean(y_train)
#         # save results
#         self.X_train = X_train
#         self.y_train = (y_train.squeeze()  - y_train_mean) / y_train_std
#         self.X_test  = X_test
#         self.y_test  = (y_test.squeeze()  - y_train_mean) / y_train_std

# class Test_LinReg_with_CC_c(ITest_Reg_with_Model): # 
#     exp_name:str       = 'CC_c'
#     n_vars:int          = 20
#     train_sizes:list    = np.linspace(50, 3000, 16, dtype=int) # range(50, 100+1, 50) #range(50, 400+1, 50)
#     test_size:int       = 10000

#     pred_models = [
#         Reg_Model(fun=regression_HS,            name='HorseShoe BLR (deg=2)',                    color='tab:purple', kwargs=dict(degree=2, nbr_samples=50)),
#         Reg_Model(fun=regression_TP,            name='TProcessReg (Kpoly(deg=2))',               color='tab:blue',   kwargs=dict(degree=2, MMAP=True, kernel_type='poly')),
#         Reg_Model(fun=regression_TP,            name='TProcessReg (Kpoly(deg=3))',               color='tab:cyan',   kwargs=dict(degree=3, MMAP=True, kernel_type='poly')),
#         Reg_Model(fun=regression_TP,            name='TProcessReg (Khamm)',                      color='tab:orange', kwargs=dict(degree=2, MMAP=True, kernel_type='hamm')),
#         Reg_Model(fun=regression_TP,            name='TProcessReg (Kdiff)',                      color='tab:brown',  kwargs=dict(degree=2, MMAP=True, kernel_type='diff')),
#         Reg_Model(fun=regression_TP,            name='TProcessReg (Kcomb[Kpoly(deg=2), Khamm])', color='tab:green',  kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_hamm')),
#         Reg_Model(fun=regression_TP,            name='TProcessReg (Kcomb[Kpoly(deg=3), Khamm])', color='tab:olive',  kwargs=dict(degree=3, MMAP=True, kernel_type='comb__poly_hamm')),
#         Reg_Model(fun=regression_TP,            name='TProcessReg (Kcomb[Kpoly(deg=2), Kdiff])', color='tab:pink',   kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_diff')),
#         Reg_Model(fun=regression_TP,            name='TProcessReg (Kcomb[Kpoly(deg=3), Kdiff])', color='tab:gray',   kwargs=dict(degree=3, MMAP=True, kernel_type='comb__poly_diff')),
#     ]

#     def setUp(self):
#         ''' Setup BQP and init linear regressors
#         '''
#         super().setUp()
#         self.model = Contamination_Control_Problem(d=self.n_vars, T=10**3)
#         encoder = OneHotEncoder(product_space=self.model.product_space)

#         # evaluate train
#         Xc_train = self.model.sample(nbr_samples=max(self.train_sizes))
#         X_train = encoder.encode(X=Xc_train).astype(int)
#         _,y_train,_ = self.model.evaluate(X=X_train)

#         # evaluate test
#         Xc_test = self.model.sample(nbr_samples=self.test_size)
#         X_test = encoder.encode(X=Xc_test).astype(int)
#         _,y_test,_ = self.model.evaluate(X=X_test)

#         # normalize data
#         y_train_std, y_train_mean = np.nanstd(y_train), np.nanmean(y_train)
#         # save results
#         self.X_train = X_train
#         self.y_train = (y_train.squeeze()  - y_train_mean) / y_train_std
#         self.X_test  = X_test
#         self.y_test  = (y_test.squeeze()  - y_train_mean) / y_train_std

# class Test_LinReg_with_SEIR_y(ITest_Reg_with_Model): # 
#     exp_name:str       = 'SEIR_y'
#     # train_sizes:list    = np.linspace(100, 6000, 12, dtype=int)
#     # test_size:int       = 6000

#     train_sizes:list    = np.linspace(50, 6000, 14, dtype=int)
#     test_size:int       = 6000

#     # pred_models = [
#     #     Reg_Model(fun=regression_GP, name='GProcessReg (Kcomb[Kpoly(deg=2), Kdiff])', color='tab:pink', linstyle='--', kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_diff')),
#     #     Reg_Model(fun=regression_TP, name='TProcessReg (Kcomb[Kpoly(deg=2), Kdiff])', color='tab:pink', linstyle='-',  kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_diff')),
#     #     # Reg_Model(fun=prediction_TP, name='TProcessReg (Kcomb[Kpoly(deg=3), Kdiff])', color='tab:gray',   kwargs=dict(degree=3, MMAP=True, kernel_type='comb__poly_diff')),
        
#     #     Reg_Model(fun=regression_TP, name='TProcessReg (Kdiff)',                      color='tab:brown',  linstyle='-',  kwargs=dict(degree=2, MMAP=True, kernel_type='diff')),
#     #     Reg_Model(fun=regression_GP, name='GProcessReg (Kdiff)',                      color='tab:brown',  linstyle='--', kwargs=dict(degree=2, MMAP=True, kernel_type='diff')),
        
#     #     # Reg_Model(fun=regression_GP, name='GProcessReg (Kpoly(deg=2))',               color='tab:blue',   linstyle='--', kwargs=dict(degree=2, MMAP=True, kernel_type='poly')),
#     #     # Reg_Model(fun=regression_TP, name='TProcessReg (Kpoly(deg=2))',               color='tab:blue',   linstyle='-',  kwargs=dict(degree=2, MMAP=True, kernel_type='poly')),

#     #     # Reg_Model(fun=prediction_TP, name='TProcessReg (Kpoly(deg=3))',               color='tab:cyan',   kwargs=dict(degree=3, MMAP=True, kernel_type='poly')),
#     #     Reg_Model(fun=regression_TP, name='TProcessReg (Khamm)',                      color='tab:orange', kwargs=dict(degree=2, MMAP=True, kernel_type='hamm')),
        
#     #     # Reg_Model(fun=regression_TP, name='TProcessReg (Kcomb[Kpoly(deg=2), Khamm])', color='tab:green',  kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_hamm')),
#     #     # Reg_Model(fun=prediction_TP, name='TProcessReg (Kcomb[Kpoly(deg=3), Khamm])', color='tab:olive',  kwargs=dict(degree=3, MMAP=True, kernel_type='comb__poly_hamm')),
        
#     #     Reg_Model(fun=regression_HS, name='HorseShoe BLR (deg=2)',                    color='tab:purple', kwargs=dict(degree=2, nbr_samples=50)),
#     # ]
#     pred_models = [
#         Reg_Model(fun=regression_TP,            name='TPReg (Kcomb[Kpoly(deg=3), Kdiff])', color='tab:gray',   kwargs=dict(degree=3, MMAP=True, kernel_type='comb__poly_diff')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kcomb[Kpoly(deg=2), Kdiff])', color='tab:pink',   kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_diff')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kcomb[Kpoly(deg=3), Khamm])', color='tab:olive',  kwargs=dict(degree=3, MMAP=True, kernel_type='comb__poly_hamm')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kcomb[Kpoly(deg=2), Khamm])', color='tab:green',  kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_hamm')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kdiff)',                      color='tab:brown',  kwargs=dict(degree=2, MMAP=True, kernel_type='diff')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Khamm)',                      color='tab:orange', kwargs=dict(degree=2, MMAP=True, kernel_type='hamm')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kpoly(deg=3))',               color='tab:cyan',   kwargs=dict(degree=3, MMAP=True, kernel_type='poly')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kpoly(deg=2))',               color='tab:blue',   kwargs=dict(degree=2, MMAP=True, kernel_type='poly')),
#         Reg_Model(fun=regression_HS,            name='HorseShoe BLR (deg=2)',              color='tab:purple', kwargs=dict(degree=2, nbr_samples=50)),
#     ]

#     def setUp(self):
#         ''' Setup BQP and init linear regressors
#         '''
#         super().setUp()
#         self.model = SEIR_Model(
#             k=2, 
#             results_path = Path(__file__).parent / 'log' / 'models_SEIR',
# 			dataset_name='dataset.csv'
#         )
#         encoder = OneHotEncoder(product_space=self.model.product_space)

#         N = (max(self.train_sizes) + self.test_size)
#         X = self.model.sample(N).astype(int)
#         y, c, l = self.model.evaluate(X=X, verbose=True)

#         # encode
#         X = encoder.encode(X=X).astype(int)

#         # randomly split train and test datasets
#         idx = np.random.permutation(range(len(X)))
#         idx_train = idx[:max(self.train_sizes)]
#         idx_test  = idx[max(self.train_sizes):]
#         X_train = X[idx_train]
#         y_train = y[idx_train]
#         X_test = X[idx_test]
#         y_test = y[idx_test]

#         # normalize data
#         y_train_std, y_train_mean = np.nanstd(y_train), np.nanmean(y_train)
#         # save results
#         self.X_train = X_train
#         self.y_train = (y_train.squeeze()  - y_train_mean) / y_train_std
#         self.X_test  = X_test
#         self.y_test  = (y_test.squeeze()  - y_train_mean) / y_train_std

# class Test_LinReg_with_SEIR_c(ITest_Reg_with_Model): # 
#     exp_name:str       = 'SEIR_c'
#     train_sizes:list    = np.linspace(100, 2000, 10, dtype=int) # range(50, 100+1, 50) #range(50, 400+1, 50)
#     test_size:int       = 5000

#     pred_models = [
#         Reg_Model(fun=regression_TP,            name='TPReg (Kcomb[Kpoly(deg=3), Kdiff])', color='tab:gray',   kwargs=dict(degree=3, MMAP=True, kernel_type='comb__poly_diff')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kcomb[Kpoly(deg=2), Kdiff])', color='tab:pink',   kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_diff')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kcomb[Kpoly(deg=3), Khamm])', color='tab:olive',  kwargs=dict(degree=3, MMAP=True, kernel_type='comb__poly_hamm')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kcomb[Kpoly(deg=2), Khamm])', color='tab:green',  kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_hamm')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kdiff)',                      color='tab:brown',  kwargs=dict(degree=2, MMAP=True, kernel_type='diff')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Khamm)',                      color='tab:orange', kwargs=dict(degree=2, MMAP=True, kernel_type='hamm')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kpoly(deg=3))',               color='tab:cyan',   kwargs=dict(degree=3, MMAP=True, kernel_type='poly')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kpoly(deg=2))',               color='tab:blue',   kwargs=dict(degree=2, MMAP=True, kernel_type='poly')),
#         Reg_Model(fun=regression_HS,            name='HorseShoe BLR (deg=2)',                    color='tab:purple', kwargs=dict(degree=2, nbr_samples=50)),
#     ]

#     def setUp(self):
#         ''' Setup BQP and init linear regressors
#         '''
#         super().setUp()
#         self.model = SEIR_Model(
#             k=2, 
#             results_path = Path(__file__).parent / 'log' / 'models_SEIR',
# 			dataset_name='dataset.csv'
#         )
#         encoder = OneHotEncoder(product_space=self.model.product_space)

#         N = max(self.train_sizes) + self.test_size
#         X = self.model.sample(N).astype(int)
#         y, c, l = self.model.evaluate(X=X, verbose=True)
#         # encode
#         X = encoder.encode(X=X).astype(int)

#         # randomly split train and test datasets
#         idx = np.random.permutation(range(len(X)))
#         idx_train = idx[:max(self.train_sizes)]
#         idx_test  = idx[max(self.train_sizes):]
#         X_train   = X[idx_train]
#         y_train   = c[idx_train]
#         X_test    = X[idx_test]
#         y_test    = c[idx_test]

#         # normalize data
#         y_train_std, y_train_mean = np.nanstd(y_train), np.nanmean(y_train)
#         # save results
#         self.X_train = X_train
#         self.y_train = (y_train.squeeze()  - y_train_mean) / y_train_std
#         self.X_test  = X_test
#         self.y_test  = (y_test.squeeze()  - y_train_mean) / y_train_std

# class Test_LinReg_with_SEIR_l(ITest_Class_with_Model): # 
#     exp_name:str       = 'SEIR_l'
#     # train_sizes:list    = np.linspace(100, 6000, 12, dtype=int)
#     # test_size:int       = 6000

#     # train_sizes:list    = np.linspace(100, 1000, 4, dtype=int)
#     # train_sizes:list    = np.array([1000], dtype=int)
#     train_sizes:list    = np.linspace(100, 2000, 10, dtype=int)
#     test_size:int       = 6000

#     pred_models = [
#         Reg_Model(fun=classification_GP, name='GPClass (Kcomb[Kpoly(deg=2), Khamm])', color='tab:green',  kwargs=dict(degree=2, kernel_type='comb__poly_hamm')),
#         Reg_Model(fun=classification_GP, name='GPClass (Kcomb[Kpoly(deg=2), Kdiff])', color='tab:pink',   kwargs=dict(degree=2, kernel_type='comb__poly_diff')),
#         Reg_Model(fun=classification_GP, name='GPClass (Kdiff)',                      color='tab:brown',  kwargs=dict(degree=2, kernel_type='diff')),
#         Reg_Model(fun=classification_GP, name='GPClass (Khamm)',                      color='tab:orange', kwargs=dict(degree=2, kernel_type='hamm')),
#         # Reg_Model(fun=classification_GP, name='GProcessReg (Kpoly(deg=2))',               color='tab:blue',   kwargs=dict(degree=2, kernel_type='poly')),
#     ]

#     def setUp(self):
#         ''' Setup BQP and init linear regressors
#         '''
#         super().setUp()
#         self.model = SEIR_Model(
#             k=2, 
#             results_path = Path(__file__).parent / 'log' / 'models_SEIR',
#             dataset_name='dataset.csv'
#         )
#         encoder = OneHotEncoder(product_space=self.model.product_space)

#         N = (max(self.train_sizes) + self.test_size)
#         X = self.model.sample(N).astype(int)
#         y, c, l = self.model.evaluate(X=X, verbose=True)

#         # encode
#         X = encoder.encode(X=X).astype(int)

#         # randomly split train and test datasets
#         idx = np.random.permutation(range(len(X)))
#         idx_train = idx[:max(self.train_sizes)]
#         idx_test  = idx[max(self.train_sizes):]
#         self.X_train = X[idx_train]
#         self.y_train = l[idx_train].squeeze()
#         self.X_test  = X[idx_test]
#         self.y_test  = l[idx_test].squeeze()

# class Test_LinReg_with_Lorenz_y(ITest_Reg_with_Model): # 
#     exp_name:str       = 'Lorenz_y'
#     # train_sizes:list    = np.linspace(100, 6000, 12, dtype=int)
#     # test_size:int       = 6000

#     train_sizes:list    = np.linspace(100, 2000, 8, dtype=int)
#     test_size:int       = 6000

#     # pred_models = [
#     #     Reg_Model(fun=regression_GP, name='GProcessReg (Kcomb[Kpoly(deg=2), Kdiff])', color='tab:pink', linstyle='--', kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_diff')),
#     #     Reg_Model(fun=regression_TP, name='TProcessReg (Kcomb[Kpoly(deg=2), Kdiff])', color='tab:pink', linstyle='-',  kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_diff')),
#     #     # Reg_Model(fun=prediction_TP, name='TProcessReg (Kcomb[Kpoly(deg=3), Kdiff])', color='tab:gray',   kwargs=dict(degree=3, MMAP=True, kernel_type='comb__poly_diff')),
        
#     #     Reg_Model(fun=regression_TP, name='TProcessReg (Kdiff)',                      color='tab:brown',  linstyle='-',  kwargs=dict(degree=2, MMAP=True, kernel_type='diff')),
#     #     Reg_Model(fun=regression_GP, name='GProcessReg (Kdiff)',                      color='tab:brown',  linstyle='--', kwargs=dict(degree=2, MMAP=True, kernel_type='diff')),
        
#     #     # Reg_Model(fun=regression_GP, name='GProcessReg (Kpoly(deg=2))',               color='tab:blue',   linstyle='--', kwargs=dict(degree=2, MMAP=True, kernel_type='poly')),
#     #     # Reg_Model(fun=regression_TP, name='TProcessReg (Kpoly(deg=2))',               color='tab:blue',   linstyle='-',  kwargs=dict(degree=2, MMAP=True, kernel_type='poly')),

#     #     # Reg_Model(fun=prediction_TP, name='TProcessReg (Kpoly(deg=3))',               color='tab:cyan',   kwargs=dict(degree=3, MMAP=True, kernel_type='poly')),
#     #     Reg_Model(fun=regression_TP, name='TProcessReg (Khamm)',                      color='tab:orange', kwargs=dict(degree=2, MMAP=True, kernel_type='hamm')),
        
#     #     # Reg_Model(fun=regression_TP, name='TProcessReg (Kcomb[Kpoly(deg=2), Khamm])', color='tab:green',  kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_hamm')),
#     #     # Reg_Model(fun=prediction_TP, name='TProcessReg (Kcomb[Kpoly(deg=3), Khamm])', color='tab:olive',  kwargs=dict(degree=3, MMAP=True, kernel_type='comb__poly_hamm')),
        
#     #     Reg_Model(fun=regression_HS, name='HorseShoe BLR (deg=2)',                    color='tab:purple', kwargs=dict(degree=2, nbr_samples=50)),
#     # ]
#     pred_models = [
#         # Reg_Model(fun=regression_TP,            name='TPReg (Kcomb[Kpoly(deg=3), Kdiff])', color='tab:gray',   kwargs=dict(degree=3, MMAP=True, kernel_type='comb__poly_diff')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kcomb[Kpoly(deg=2), Kdiff])', color='tab:pink',   kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_diff')),
#         # Reg_Model(fun=regression_TP,            name='TPReg (Kcomb[Kpoly(deg=3), Khamm])', color='tab:olive',  kwargs=dict(degree=3, MMAP=True, kernel_type='comb__poly_hamm')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kcomb[Kpoly(deg=2), Khamm])', color='tab:green',  kwargs=dict(degree=2, MMAP=True, kernel_type='comb__poly_hamm')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kdiff)',                      color='tab:brown',  kwargs=dict(degree=2, MMAP=True, kernel_type='diff')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Khamm)',                      color='tab:orange', kwargs=dict(degree=2, MMAP=True, kernel_type='hamm')),
#         # Reg_Model(fun=regression_TP,            name='TPReg (Kpoly(deg=3))',               color='tab:cyan',   kwargs=dict(degree=3, MMAP=True, kernel_type='poly')),
#         Reg_Model(fun=regression_TP,            name='TPReg (Kpoly(deg=2))',               color='tab:blue',   kwargs=dict(degree=2, MMAP=True, kernel_type='poly')),
#         Reg_Model(fun=regression_HS,            name='HorseShoe BLR (deg=2)',              color='tab:purple', kwargs=dict(degree=2, nbr_samples=50)),
#     ]

#     def setUp(self):
#         ''' Setup BQP and init linear regressors
#         '''
#         super().setUp()
#         self.model = Lorenz_Model(
#             k=2, 
#             results_path = Path(__file__).parent / 'log' / 'models_Lorenz',
# 			dataset_name='dataset.csv'
#         )
#         encoder = OneHotEncoder(product_space=self.model.product_space)

#         N = (max(self.train_sizes) + self.test_size)
#         X = self.model.sample(N).astype(int)
#         y, c, l = self.model.evaluate(X=X, verbose=True)

#         # encode
#         X = encoder.encode(X=X).astype(int)

#         # randomly split train and test datasets
#         idx = np.random.permutation(range(len(X)))
#         idx_train = idx[:max(self.train_sizes)]
#         idx_test  = idx[max(self.train_sizes):]
#         X_train = X[idx_train]
#         y_train = y[idx_train]
#         X_test = X[idx_test]
#         y_test = y[idx_test]

#         # normalize data
#         y_train_std, y_train_mean = np.nanstd(y_train), np.nanmean(y_train)
#         # save results
#         self.X_train = X_train
#         self.y_train = (y_train.squeeze()  - y_train_mean) / y_train_std
#         self.X_test  = X_test
#         self.y_test  = (y_test.squeeze()  - y_train_mean) / y_train_std



