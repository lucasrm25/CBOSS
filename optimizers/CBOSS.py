'''
Author: Lucas Rath
'''
from copy import copy, deepcopy
from operator import itemgetter
from typing import Callable, List, Tuple, Union
import numpy as np
from joblib import Parallel, delayed
import torch
import time
import textwrap
from pyro.distributions.multivariate_studentt import MultivariateStudentT
from torch.distributions.multivariate_normal import MultivariateNormal
from CBOSS.bayesian_models.acquisitions import FRCHEI
from CBOSS.optimizers.simulated_annealing import _simulated_annealing
from CBOSS.models.search_space import ProductSpace, NominalSpace, BinarySpace
from CBOSS.bayesian_models.means import ZeroMean
from CBOSS.bayesian_models.kernels import CombKernel, PolynomialKernel, HammingKernel, DiscreteDiffusionKernel, BinaryDiscreteDiffusionKernel, OneHotEncodedKernel
from CBOSS.bayesian_models.regression import GaussianProcessRegression, StudentTProcessRegression, BayesianRegression
from CBOSS.bayesian_models.classification import LaplaceGaussianProcessClassification
from CBOSS.bayesian_models.svgp import SVGP, BernoulliSigmoidLikelihood
from CBOSS.utils.torch_utils import ModuleX
from CBOSS.utils.numpy_utils import argminN, argmaxN, arg_new_unique


class InverseScale(ModuleX):
	''' Scale back the output of a BayesianRegression model: y = (y_scaled * scale) + shift
	'''
	def __init__(self, reg:BayesianRegression, scale:float, shift:float) -> None:
		super().__init__()
		assert isinstance(reg, BayesianRegression)
		self.reg = reg
		self.scale = scale
		self.shift = shift

	def forward(self, *args, **kwargs) -> torch.Tensor:
		dist = self.reg(*args, **kwargs)
		if isinstance(dist, MultivariateStudentT):
			return MultivariateStudentT(
				df  = dist.df,
				loc = dist.loc * self.scale + self.shift,
				scale_tril = dist.scale_tril * self.scale
			)
		elif isinstance(dist, MultivariateNormal):
			return MultivariateNormal(
				loc = dist.loc * self.scale + self.shift,
				scale_tril = dist.scale_tril * self.scale
			)
		else:
			raise NotImplementedError

def train_surrogates(
    X:torch.Tensor, y:torch.Tensor, c:torch.Tensor, l:torch.Tensor,
    product_space:ProductSpace,
    n_evals:int, 
    evalBudget:int,
    reg_mode:str, 
    kernels:List[str] = ['POLY','DIFF'], 
    handle_failures:bool = True, 
    train_sigma2:bool = True, 
    verbose:bool = False, 
    dtype = torch.float64,
	beta_feas_fac:float = 20.,
	beta_succ_fac:float = 5.,
 	clas_type:str = ['LGP','SVGP'][1],
):
	'''
	Trains surrogate models for the objective function, constraints, and success/failure classification.

	Args:
	- X: <N,nx>
	- y: <N,1>
	- c: <N,nc>
	- l: <N,1>

	Returns:
	- FRCHEI_acq: FRCHEI acquisition function
	- reg_f: BayesianRegression model for the objective function
	- reg_g: List of BayesianRegression models for the constraints
	- class_h: BayesianRegression model for the success/failure classification
 	'''
	allowed_kernels = ['POLY', 'DIFF', 'BDIFF', 'HAM']

	''' Normalize data
	=========================='''
	y_mean = np.nanmean(y, axis=0)
	y_std  = np.nanstd(y, axis=0)
	y_s = (y - y_mean) / y_std
	c_mean = np.nanmean(c, axis=0)
	c_std  = np.nanstd(c, axis=0)
	c_s = (c - c_mean) / c_std

	''' Parse inputs
	=========================='''
	RegCls = StudentTProcessRegression if reg_mode == 'TP' else GaussianProcessRegression

	kernels = [k.upper() for k in kernels]
	assert len(kernels) > 0 and all([k in allowed_kernels for k in kernels]), f'kernel list must be contained in {allowed_kernels}'

	Klist = []
	if 'POLY' in kernels:
		Klist += [ OneHotEncodedKernel(kernel=PolynomialKernel(variance_prior=0.5, degree=2), product_space=product_space) ]
	if 'BDIFF' in kernels:
		# NOTE: this only works if all variables are Bonary or Nominal
		n_oh = OneHotEncodedKernel.onehot_len(product_space=product_space)
		Klist += [ OneHotEncodedKernel(kernel=BinaryDiscreteDiffusionKernel(length_scales=[1.]*n_oh), product_space=product_space) ]
	if 'DIFF' in kernels:
		nx = X.shape[1]
		Klist += [ DiscreteDiffusionKernel(length_scales=[1.]*nx, product_space=product_space) ]	# torch.abs(Kdiff(Xt,Xt) - Kbdiff(Xt,Xt)).max()
	if 'HAM' in kernels:
		nx = X.shape[1]
		Klist += [ HammingKernel(length_scales=[1.]*nx) ]
	
	if len(Klist) > 1:
		K = CombKernel(kernelList=Klist)
	else:
		K = Klist[0]
  
	assert clas_type in ['LGP','SVGP'], f'clas_type={clas_type} must be one of ["LGP","SVGP"]'

	''' Train models
	=========================='''
	# train regression surrogate for objective function
	reg_f_scaled = RegCls(X=X, y=y_s[:,0], mean_fun=ZeroMean(), kernel=deepcopy(K), train_sigma2=train_sigma2).to(dtype)
	reg_f_scaled.train()
	reg_f_scaled.fit(lr=1.0, maxiter=500, disp=verbose)
	reg_f_scaled.eval()
	reg_f = InverseScale(reg=reg_f_scaled, scale=y_std, shift=y_mean)

	# train regression surrogates for constraints
	reg_g = []
	for i in range(c.shape[1]):
		reg_g_scaled = RegCls(X=X, y=c_s[:,i], mean_fun=ZeroMean(), kernel=deepcopy(K), train_sigma2=train_sigma2).to(dtype)
		reg_g_scaled.train()
		reg_g_scaled.fit(lr=1.0, maxiter=500, disp=verbose)
		reg_g_scaled.eval()
		reg_g.append(InverseScale(reg=reg_g_scaled, scale=c_std[i], shift=c_mean[i]))

	# train classification surrogate for success/failures
	class_h = None
	if handle_failures:
		if clas_type == 'LGP':
			class_h = LaplaceGaussianProcessClassification(X=X, y=l[:,0], kernel=deepcopy(K)).to(dtype)
			class_h.train()
			opt_state = class_h.fit(lr=1.0, maxiter=1, maxevals_per_iter=int(1e3), disp=verbose)
			class_h.eval()
			if not opt_state['has_converged']:
				print(f'WARNING: GP-Classifier did not converge')
		elif clas_type == 'SVGP':
			N = len(X)
			Ns = N
			s = torch.rand(N).argsort()[:Ns]
			class_h = SVGP(
				X=X, y=l[:,0], s=s,
				kernel = deepcopy(K),
				likelihood = BernoulliSigmoidLikelihood(),
				batch_size = N,
				optimizer='lbfgs'
			).double()
			class_h.fit(lr=1.0, maxiter=500, disp=verbose)
			class_h.eval()

	''' Init acquisition function
	=========================='''
	beta_feas = (beta_feas_fac*n_evals/evalBudget)
	beta_succ = (beta_succ_fac*n_evals/evalBudget)
	# select best feasible sample. If no feasible sample exists, select sample that is closest to feasibility
	idx_succ = (l==1).all(1)
	idx_feas = (c<=0).all(1) & idx_succ
	y_s_best_feas = y_s[idx_feas].min() if idx_feas.any() else y_s[idx_succ][np.nanargmin(np.nanmax(c[idx_succ],1))]
	FRCHEI_acq = FRCHEI(reg_f=reg_f_scaled, reg_g=reg_g, class_h=class_h, beta_feas=beta_feas, beta_succ=beta_succ, y_best=y_s_best_feas) # np.nanmin(y_s)
	return FRCHEI_acq, reg_f, reg_g, class_h

def CBOSS( 
	eval_fun:Callable[[np.ndarray],Tuple[np.ndarray,np.ndarray,np.ndarray]],
	X:np.ndarray,
	y:np.ndarray, 
	c:np.ndarray,
	l:np.ndarray,
	product_space:ProductSpace, 
	# 
	evalBudget:int=500, 
	maxIter=np.inf,  
	n_jobs:int = 1,
	batchsize:int = 2,
	double_precision:bool = True,
	acq_fun:str = 'FRCHEI',
	kernels:List[str] = ['POLY','DIFF'],
	clas_type:str = ['LGP','SVGP'][1],
	#
	beta_feas_fac:float = 20.,
	beta_succ_fac:float = 5.,
	#
	SA_reruns_per_batch:int = 2,
	SA_kwargs:dict = dict(
		evalBudget = 100,
		T_point	= (0.5, 0.005),
		stop_criteria_factor = 2
	),
	#
	verbose:bool = False,
	log_fun:Callable[[dict],None] = None
) -> dict:
	''' Combinatorial Bayesian Optimization for Structure Selection (CBOSS) algorithm.

	Optimization problem setup:
	```
	x* = arg min 	f(x) 
			s.t.	g(x) <= 0
					h(x) = 0
	```
 
	where:
 	- x in X is a vector of binary or categorical variables.
	- f: X -> R is the objective function.
	- g: X -> R^Nc is the vector of inequality constraints.
	- h: X -> {0,1} is the vector of binary equality constraints.


	This method uses surrogate models specialized for categorical and binary spaces.
	
	It uses the Failure-Robust Constrained Hierarchical Expected Improvement (FRCHEI) acquisition function,
	which can handle inequality and binary equality constraints.


	Args:
	- eval_fun: A function that takes X as input and returns a tuple of three arrays: y, c, and l.
	- X: An array of shape (N, Nx) representing the input data.
	- y: An array of shape (N, 1) representing the objective evaluations.
	- c: An array of shape (N, Nc) representing the constraint evaluations.
	- l: An array of shape (N, 1) representing the success flag evaluations.
	- product_space: A ProductSpace object representing the search space.
	- evalBudget: The maximum number of function evaluations.
	- maxIter: The maximum number of iterations.
	- n_jobs: The number of parallel jobs to run.
	- batchsize: The number of samples to optimize in parallel.
	- double_precision: Whether to use double precision.
	- acq_fun: The acquisition function to use from `['FRCHEI','CHEI','FRCEI','CEI']`.
	- kernels: The kernels to use from `['POLY', 'DIFF', 'BDIFF', 'HAM']`.
	- clas_type: The classifier type to use in case there are evaluation failures/crashes.
	- beta_feas_fac: The feasibility factor to use.
	- beta_succ_fac: The success factor to use.
	- SA_reruns_per_batch: The number of times to rerun simulated annealing per batch.
	- SA_kwargs: A dictionary of arguments to pass to the simulated annealing function.
	- verbose: Whether to print verbose output.
	- log_fun: A function to call with logging information.

	Returns:
	- A dictionary containing the optimization results.
	'''
	walltime_start = time.time()
	walltime_last  = time.time() - walltime_start

	''' Parse arguments
	======================'''
	# copy this object because we might modify this during the optimization
	SA_kwargs = deepcopy(SA_kwargs)

	N,Nx = X.shape
	dtype = torch.float64 if double_precision else torch.float32

	assert X.ndim==y.ndim==l.ndim==c.ndim==2 and len(X)==len(y)==len(l)==len(c)
	assert y.shape[1]==1 and l.shape[1]==1, f'y.shape={y.shape}, l.shape={l.shape} must be (N,1)'

	allowed_acq = ['FRCHEI','CHEI','FRCEI','CEI']
	assert acq_fun.upper() in allowed_acq, f'acq_fun={acq_fun} must be one of {allowed_acq}'

	reg_mode = 'TP' if acq_fun in ['FRCHEI','CHEI'] else 'GP' if acq_fun in ['FRCEI','CEI'] else None
	handle_failures = acq_fun in ['FRCHEI','FRCEI']

	if verbose:
		print(
      		f'Starting CBO with the following settings:\n'
        	f'\tacquisition function: {acq_fun}\n'
         	f'\tregression model: {reg_mode}\n'
         	f'\tkernels: {kernels}\n'
         	f'\thandle failures: {handle_failures}\n'
          	f'\tbatch size: {batchsize}'
        )

	# assert that all categorical inputs are defined as numbers and not strings
	assert all([isinstance(s,NominalSpace) or isinstance(s,BinarySpace) for s in product_space.subspaces]),\
		'This code only currently only supports binary and categorical variables'
	assert all([ 
        all([ 
            isinstance(cat_opt,int) or isinstance(cat_opt,float) 
            for cat_opt in product_space.subspaces[i].bounds
        ])  
        for i in product_space.id_N
    ]), 'This code currently only supports categorical variables, whose options are all numbers, not strings'

	''' Start Optimization
	=========================='''
	# init some logging arrays
	y_pred_mean   = y.copy()
	y_pred_std    = y * 0		# initial given data is known
	c_pred_mean   = c.copy()
	c_pred_std    = c * 0		# initial given data is known
	l_pred_prob   = l.copy()
	walltime 	  = np.zeros(len(X))
	with Parallel(n_jobs=n_jobs, prefer="threads", backend='threading') as parallel: # https://joblib.readthedocs.io/en/latest/parallel.html

		n_iter = 0
		n_evals = len(X)  # we count the given dataset as evaluations
		while n_evals < evalBudget and n_iter < maxIter:

			if verbose: print(f'\nIteration: {n_iter}/{maxIter} - nbr_evals: {n_evals}/{evalBudget}\n')

			''' 1. Retrain surrogates
			=========================='''
			time_training_start = time.time()	
			FRCHEI_acq, reg_f, reg_g, class_l = train_surrogates(
				X = torch.as_tensor(X.astype(float)),
				y = torch.as_tensor(y),
				c = torch.as_tensor(c),
				l = torch.as_tensor(l),
				n_evals=n_evals, evalBudget=evalBudget,
				product_space=product_space,
				reg_mode=reg_mode, kernels=kernels, 
    			train_sigma2=True,
				handle_failures=handle_failures, dtype=dtype, verbose=verbose,
				beta_feas_fac=beta_feas_fac,
				beta_succ_fac=beta_succ_fac,
				clas_type=clas_type,
			)
			time_training_end = time.time()
			if verbose: 
				print(f'Training time: {time_training_end - time_training_start:.2f} s\n')

			''' 2. Optimize acquisition function
			======================================'''
			with torch.no_grad():

				X_AcqOpt_bests = np.zeros((0,Nx))
				while len(X_AcqOpt_bests) < batchsize:
					
					''' 2.1. select starting point for the optimizer. 
     				Selects best feasible locations. Otherwise, select locations that are closest to feasibility'''
					idx_succ = (l==1).all(1)
					idx_feas = (c<=0).all(1) & idx_succ
					idx_unfeas = (c>0).all(1) & idx_succ
					# select the `batchsize*SA_reruns_per_batch` best feasible samples. If there are not enough feasible, take the samples that are closest to feasibility
					nbr_select = batchsize*SA_reruns_per_batch
					nbr_feas = min(nbr_select,idx_feas.sum())
					nbr_unfeas = nbr_select - nbr_feas
					# select
					X_best_feas = X[idx_feas][argminN(y[idx_feas,0],nbr_feas)] if nbr_feas > 0 else np.zeros((0,Nx),dtype=int)
					X_best_unfeas = X[idx_unfeas][argminN(c[idx_unfeas].max(1),nbr_unfeas)] if nbr_unfeas > 0 else np.zeros((0,Nx),dtype=int)
					X_best = np.vstack([X_best_feas, X_best_unfeas])

					''' 2.2 optimize acquisition function 
     				'''
					time_acqopt_start = time.time()
					acqOpt_results = parallel(
						delayed(_simulated_annealing)(
							objective     = lambda X_pred: - FRCHEI_acq( torch.as_tensor(X_pred.astype(float)), log=True),
							product_space = product_space,
							X_init = X_best[None,i],
							**SA_kwargs
						) 
						for i in range(batchsize * SA_reruns_per_batch)
					)
					time_acqopt_end = time.time()
					print(f'SA time: {time_acqopt_end - time_acqopt_start:.2f} s')

					''' 2.3 select the best unique solutions for each batch
     				'''
					# zip results into X and acq arrays
					X_AcqOpt_runs, acq_AcqOpt_runs = list(zip(*acqOpt_results))
					acq_AcqOpt_runs = [-aq for aq in acq_AcqOpt_runs]		# NOTE: SA minimizes while the FRCHEI is to be maximized
					# init lists for storing best samples
					X_AcqOpt_bests   = np.empty((0,Nx)) # np.zeros((batch_size,nx))
					acq_AcqOpt_bests = np.empty((0)) # np.zeros((batch_size))
					for idxs in np.split(np.arange(nbr_select), batchsize):
						X_AcqOpt   = np.vstack( itemgetter(*idxs)(X_AcqOpt_runs) )
						acq_AcqOpt = np.hstack( itemgetter(*idxs)(acq_AcqOpt_runs) )
						# select best unique solutions for each batch
						idx_unique = arg_new_unique(X_new=X_AcqOpt, X=X)
						idx_unique = idx_unique[arg_new_unique(X_new=X_AcqOpt[idx_unique], X=X_AcqOpt_bests)]
						idx_max_acq = argmaxN(y=acq_AcqOpt[idx_unique], N=1)
						# append to the list of samples to be evaluated
						X_AcqOpt_bests   = np.vstack((X_AcqOpt_bests,   X_AcqOpt[idx_unique][idx_max_acq]))
						acq_AcqOpt_bests = np.hstack((acq_AcqOpt_bests, acq_AcqOpt[idx_unique][idx_max_acq]))
					if len(X_AcqOpt_bests) < batchsize:
						SA_kwargs['evalBudget'] = int(1.2*SA_kwargs['evalBudget'])
						SA_reruns_per_batch += 1
						print(f'Increasing Simulated Annealing evalBudget to {SA_kwargs["evalBudget"]}, '\
            				f'and number of reruns to {SA_reruns_per_batch} because no new solutions have been found with SA')
					if verbose:
						print(f'Acq Opt run in {time_acqopt_end-time_acqopt_start:.1f} s\n')

				''' 3. Evaluate expensive model
				==================================='''
				# evaluate model objective at new evaluation points
				X_eval = X_AcqOpt_bests
				assert len(set((tuple(x) for x in X_eval))) == len(X_eval), 'repeated X candidates for evaluation'
				assert len(X_eval) == batchsize

				time_eval_start = time.time()
				y_eval, c_eval, l_eval = eval_fun(X_eval)
				time_eval_end = time.time()

				if verbose:
					with np.printoptions(edgeitems=50, linewidth=100000):
						print(f'Model evaluated in {time_eval_end-time_eval_start:.1f} s\n')
						print(f'New configurations evaluated:')
						for i in range(len(X_eval)):
							print(textwrap.indent( f'X: {"".join([str(x) for x in X_eval[i]])}  y: {y_eval[i,0]:.2f}  c: {np.nanmax(c_eval[i]):.2f}  l: {l_eval[i,0]}', "\t" ))
						print()

				n_evals += len(y_eval)

				''' 4. Update dataset
				========================='''
				# Update dataset
				X   = np.concatenate((X, X_eval), axis=0)
				y   = np.concatenate((y, y_eval), axis=0)
				c   = np.concatenate((c, c_eval), axis=0)
				l   = np.concatenate((l, l_eval), axis=0)

				''' Log
				==========='''
				X_eval = torch.as_tensor(X_eval.astype(float)).to(dtype)

				y_pred_dist_new   = reg_f(X_pred=X_eval, diagonal=True, include_noise=False)
				y_pred_mean_new	  = y_pred_dist_new.mean.numpy()
				y_pred_std_new 	  = y_pred_dist_new.variance.numpy() ** 0.5
				y_pred_mean   	  = np.concatenate((y_pred_mean, y_pred_mean_new[:,None]))
				y_pred_std    	  = np.concatenate((y_pred_std,  y_pred_std_new[:,None]))

				c_pred_dist_new   = [reg_gi(X_pred=X_eval, diagonal=True, include_noise=False) for reg_gi in reg_g]
				c_pred_mean_new   = np.vstack([ci_dist.mean for ci_dist in c_pred_dist_new]).T
				c_pred_std_new    = np.vstack([ci_dist.variance ** 0.5 for ci_dist in c_pred_dist_new]).T
				c_pred_mean 	  = np.vstack((c_pred_mean, c_pred_mean_new))
				c_pred_std  	  = np.vstack((c_pred_std,  c_pred_std_new))

				l_pred_prob_new = class_l(X_pred=X_eval).mean.cpu().numpy()[:,None] if handle_failures else np.ones_like(l_eval)
				l_pred_prob	    = np.concatenate((l_pred_prob, l_pred_prob_new))

				walltime_curr = time.time() - walltime_start
				walltime = np.concatenate((walltime, [walltime_curr]*batchsize))

				optimizer_log = dict(
					X=X, y=y, c=c, l=l, 
					walltime = walltime,
					y_pred_mean = y_pred_mean, y_pred_std = y_pred_std,
					c_pred_mean = c_pred_mean, c_pred_std = c_pred_std, 
					l_pred_prob = l_pred_prob,
				)

				if verbose:
					print(f'nevals: {len(y)}/{evalBudget}  time: {walltime[-1]/60.:.1f}min  t_iter: {(walltime_curr-walltime_last)/60.:.1f}min  y_feas_min: {np.nanmin(y[(c<=0).all(1)],initial=np.inf):.3f}\n')

				if log_fun is not None:
					log_fun(log=optimizer_log)

				walltime_last = walltime_curr
				n_iter  += 1

	return optimizer_log
