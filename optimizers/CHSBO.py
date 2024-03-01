# '''
# Author: Lucas Rath
# '''

# import sys
# from copy import copy, deepcopy
# from dataclasses import dataclass, field
# from operator import itemgetter
# from typing import Callable, List, Tuple, Union
# import numpy as np
# from joblib import Parallel, delayed
# import torch
# from torch.distributions.normal import Normal
# import scipy as sp
# import time
# import textwrap

# from CBOSS.bayesian_models.acquisitions import hierarchical_expected_improvement, expected_improvement
# from CBOSS.optimizers.simulated_annealing import _simulated_annealing
# from CBOSS.models.search_space import ProductSpace, OneHotEncoder
# from CBOSS.bayesian_models.means import ZeroMean
# from CBOSS.bayesian_models.kernels import CombKernel, PolynomialKernel, HammingKernel, DiscreteDiffusionKernel
# from CBOSS.bayesian_models.regression import GaussianProcessRegression, StudentTProcessRegression, HorseShoeBayesianLinearRegression
# from CBOSS.bayesian_models.classification import LaplaceGaussianProcessClassification
# from CBOSS.bayesian_models.features import PolynomialFeatures
# from CBOSS.utils.numpy_utils import argminN, argmaxN, arg_new_unique
# from CBOSS.utils.torch_utils import Scaler
# def infill_SCBO(X_new:np.ndarray, y_new:np.ndarray, c_new:np.ndarray, N:int=1):
# 	''' Selection strategy based on `Eriksson et al (2021) - Scalable Constrained BO` - SCBO
#  	which is an extension of Thompson sampling to constraints
	
# 	Let   F = { i | c_l(x_i) <= 0 for 1<=l<=m }

# 	if F is not empty, then select   
# 		x_i = argmin_{i in F} y_new_i
# 	otherwise, select
# 		x_i = argmin_{x} sum_{l=1}^m  max{ c_l(x), 0 }

# 	Returns (X_new,y_new,c_new)
# 	'''
# 	assert X_new.shape[0] == y_new.shape[0] == c_new.shape[0] and X_new.ndim == c_new.ndim == 2 and y_new.ndim == 1, 'Wrong dimensions!'
# 	Nx = len(X_new)
# 	if N > Nx:
# 		raise RuntimeError(f'There are not enough {N} unique values of X_SA that are not in X')

# 	# select the indices of feasible candidates
# 	F = np.array([i for i in range(Nx) if (c_new[i,:] <= 0).all() ], dtype=int)

# 	# number of feasible (and unfeasible) points we have to select
# 	N_feas = min( len(F), N )
# 	N_unfeas = N - N_feas
	
# 	# select N_feas points with best objective
# 	idx_feas = F[argminN(y=y_new[F], N=N_feas).tolist()].tolist()
# 	# select N_unfeas points with minimum total violation
# 	idx_unfeas = argminN( y = np.sum( c_new.clip(0) , axis=1), N=N_unfeas ).tolist()

# 	idx_best = idx_feas + idx_unfeas
# 	return X_new[idx_best], y_new[idx_best], c_new[idx_best]

# def BOCS_SCBHS( 
#         eval_fun:Callable[[np.ndarray],Tuple[np.ndarray,np.ndarray]],
# 		X:np.ndarray, 
# 		y:np.ndarray, 
# 		c:np.ndarray, 
# 		l:np.ndarray,
# 		product_space:ProductSpace, 
#   		SA_kwargs:dict,
# 		SA_reruns:int=5, 
# 		evalBudget:int=100, 
# 		maxIter=np.inf,  
# 		order_f:int=2, 
# 		order_g:int=2, 
# 		n_jobs:int=1, 
# 		batch_size:int=4, 
# 		log_fun:Callable[[dict],None] = None
# ) -> dict:
# 	''' BOCS: Scalable Constrained Bayesian Optimization for Combinatorial Structures using a BHS (Bayesian Horseshoe prior)
# 	for the objective function and for the constraints.

# 	Infill criterion is the Thompson Sampling with Constraints (TSC) 
# 	Eriksson and Poloczek, 2021 - Scalable Constrained Bayesian Optimization
	
# 	 Inputs: 
# 		- objective_constraints_fun: returns (f(X),c(X)): <N,nx> --> (<N,>,<N,nc>)
# 		- evalBudget: max number of function evaluations
# 		- X: <N,nx>
# 		- y: <N,>
# 		- order (int): statistical model order
# 	'''
# 	n_c = c.shape[1]

# 	''' Scale data
# 	=========================='''
# 	# rescale data and update regression model
# 	y_scaler = Scaler()  # StandardScaler()
# 	y_s = y_scaler.fit(y[:,None]).transform(y[:,None])[:,0]
# 	c_scaler = Scaler()  # StandardScaler()
# 	c_s = c_scaler.fit(c).transform(c)

# 	''' Start Optimization
# 	=========================='''
# 	y_pred_sample = y.copy()
# 	c_pred_sample = c.copy()
# 	y_pred_mean   = y.copy()
# 	# y_pred_std    = y * 0
# 	c_pred_mean   = c.copy()
# 	# c_pred_std    = c * 0
# 	with Parallel(n_jobs=n_jobs, prefer="threads") as parallel: # https://joblib.readthedocs.io/en/latest/parallel.html

# 		n_iter = 0
# 		n_evals = X.shape[0]  # we count the given dataset as evaluations
# 		log = {}
# 		while n_evals < evalBudget and n_iter < maxIter:

# 			''' Retrain surrogate
# 			=========================================='''

# 			# update linear regression model
# 			linreg_f = HorseShoeBayesianLinearRegression(X=X, y=y_s, features=PolynomialFeatures(degree=order_f))
# 			linreg_g = [
# 				HorseShoeBayesianLinearRegression(X=X, y=c_s_i, features=PolynomialFeatures(degree=order_g))
# 				for c_s_i in c_s.T
# 			]
# 			alpha_post_samples_f = linreg_f.posterior_sample_alpha(nsamples=SA_reruns, burnin=100, thin=10)
# 			alpha_post_samples_g = [
#        			linreg_g_i.posterior_sample_alpha(nsamples=SA_reruns, burnin=100, thin=10)
# 				for linreg_g_i in linreg_g
# 			]

# 			''' Optimize acquisition function using Infill Criterion
# 			==========================
#    			'''

# 			# Run SA optimization - NOTE: for each rerun, sample another parameters from the posterior
# 			SA_results = parallel(
# 				delayed(_simulated_annealing)(
# 					objective = lambda X: linreg_f.y_mean(X=X, alpha_mean=alpha_post_samples_f[-i]),  # get one sample from alpha posterior
# 					product_space=product_space,
# 					**SA_kwargs
# 				)
# 				for i in range(SA_reruns)
# 			)
# 			X_SA_all = np.vstack([ X_SA for X_SA, _ in SA_results ])
# 			# scaled predictions
# 			y_SA_all_s = np.hstack([ y_SA for _, y_SA in SA_results ])
# 			c_SA_all_s = np.vstack([	# NOTE: here we also sample the posterior for predicting the constraints
# 				linreg_g_i.y_mean(X=X_SA_all, alpha_mean=alpha_post_samples_g_i[-1])		
# 				for linreg_g_i, alpha_post_samples_g_i in zip(linreg_g, alpha_post_samples_g)
# 			]).T
# 			# rescale back to original space
# 			y_SA_all = y_scaler.inv_transform(y_SA_all_s)
# 			c_SA_all = c_scaler.inv_transform(c_SA_all_s)

# 			try:
# 				# ignore visited inputs
# 				idx_unique = arg_new_unique(X_new=X_SA_all, X=X)
# 				X_SA_all, y_SA_all, c_SA_all = X_SA_all[idx_unique], y_SA_all[idx_unique], c_SA_all[idx_unique]

# 				# select based on **SCBO infill criterion**
# 				X_SA_bests, y_SA_bests, c_SA_bests = infill_SCBO(X_new=X_SA_all, y_new=y_SA_all, c_new=c_SA_all, N=batch_size)

# 			except Exception as e:
# 				print(e, file=sys.stderr)
# 				# print(f'Warning... SA did not find enough new unique {parallel_evals} candidates, that have not been evaluated before')
# 				continue

# 			''' Evaluate model, and update surrogate
# 			=========================================='''

# 			# evaluate model objective at new evaluation points
# 			X_eval = X_SA_bests
# 			y_eval, c_eval = eval_fun(X_eval)

# 			n_evals += len(y_eval)

# 			# Update dataset
# 			X = np.vstack((X, X_eval))
# 			y = np.hstack((y, y_eval))
# 			c = np.vstack((c, c_eval))
# 			# scale
# 			y_s = y_scaler.fit(y[:,None]).transform(y[:,None])[:,0]
# 			c_s = c_scaler.fit(c).transform(c)

# 			''' Log
# 			==========='''

# 			r2_train_f = linreg_f.score(X=linreg_f.X, y_true=linreg_f.y, alpha_mean=alpha_post_samples_f.mean(0))
# 			r2_train_g = np.array([
#        			linreg_g_i.score(X=linreg_g_i.X, y_true=linreg_g_i.y, alpha_mean=alpha_post_samples_g_i.mean(0))
# 				for linreg_g_i, alpha_post_samples_g_i in zip(linreg_g, alpha_post_samples_g)
#            ])

# 			y_pred_sample = np.hstack((y_pred_sample, y_SA_bests))
# 			c_pred_sample = np.vstack((c_pred_sample, c_SA_bests))

# 			y_pred_mean_new =  y_scaler.inv_transform(
# 				linreg_f.y_mean(X=X_SA_bests, alpha_mean=alpha_post_samples_f.mean(0))
# 			)
# 			c_pred_mean_new =  c_scaler.inv_transform(
# 				np.vstack([
# 					linreg_g_i.y_mean(X=X_SA_bests, alpha_mean=alpha_post_samples_g_i.mean(0))
# 					for linreg_g_i, alpha_post_samples_g_i in zip(linreg_g, alpha_post_samples_g)
# 				]).T
# 			)
# 			y_pred_mean = np.hstack((y_pred_mean, y_pred_mean_new))
# 			c_pred_mean = np.vstack((c_pred_mean, c_pred_mean_new))

# 			optimizer_log = dict(
# 				X=X, y=y, c=c, l=l, 
# 				y_pred_mean = y_pred_mean, y_pred_std = 0*y_pred_mean,
# 				c_pred_mean = c_pred_mean, c_pred_std = 0*c_pred_mean, 
# 			)

# 			if log_fun is not None:
# 				log_fun(optimizer_log)

# 			n_iter  += 1

# 	return optimizer_log

# def BOCS_BHS( 
# 		objective_fun:Callable[[np.ndarray], np.ndarray], 
# 		X:np.ndarray, 
# 		y:np.ndarray, 
# 		product_space:ProductSpace, 
# 		SA_kwargs:dict,
# 		SA_reruns:int=5, 
# 		evalBudget:int=100, 
# 		maxIter=np.inf,  
# 		order:int=2, 
# 		n_jobs:int=1, 
# 		batch_size:int=4, 
# 		log_fun:Callable[[dict],None] = None
# ):
# 	''' BOCS: Function runs binary optimization using simulated annealing on
# 	 the model drawn from the distribution over beta parameters
	
# 	 Inputs: 
# 		- objective_fun: returns f(X): <N,nx> --> <N,>
# 		- evalBudget: max number of function evaluations
# 		- X: <N,nx>
# 		- y: <N,>
# 		- order (int): statistical model order
# 	'''
# 	# Train initial statistical model
# 	linreg = HorseShoeBayesianLinearRegression(X=X, y=y, features=PolynomialFeatures(degree=order))

# 	with Parallel(n_jobs=n_jobs, prefer="threads") as parallel: # https://joblib.readthedocs.io/en/latest/parallel.html

# 		n_iter = 0
# 		n_evals = X.shape[0]  # we count the given dataset as evaluations
# 		log = {}
# 		while n_evals < evalBudget and n_iter < maxIter:


# 			''' Retrain surrogate
# 			=========================================='''

# 			alpha_post_samples = linreg.posterior_sample_alpha(nsamples=SA_reruns, burnin=100, thin=10)

# 			if n_iter == 0:
# 				y_pred_mean = linreg.y_mean(X=X, alpha_mean=alpha_post_samples.mean(0))

# 			''' Optimize acquisition function
# 			=========================='''

# 			# Run SA optimization - for each rerun, sample another parameters from the posterior
# 			SA_results = parallel(
# 				delayed(_simulated_annealing)(
# 					objective = lambda X: linreg.y_mean(X=X, alpha_mean=alpha_post_samples[-i]),  # get one drawn sample from alpha posterior
# 					product_space=product_space,
# 					**SA_kwargs
# 				) 
# 				for i in range(SA_reruns)
# 			)
# 			X_SA_all = np.vstack([ X_SA for X_SA, _ in SA_results ])
# 			y_SA_all = np.hstack([ y_SA for _, y_SA in SA_results ])

# 			try:
# 				# select best unique solutions generated by SA
# 				idx_unique = arg_new_unique(X_new=X_SA_all, X=X)
# 				idx_min_y  = idx_unique[ argminN(y=y_SA_all[idx_unique], N=batch_size) ]
# 				X_SA_bests = X_SA_all[idx_min_y]
# 				y_SA_bests = y_SA_all[idx_min_y]

# 			except Exception as e:
# 				print(e, file=sys.stderr)
# 				continue

# 			''' Evaluate model, and update surrogate
# 			=========================================='''

# 			# evaluate model objective at new evaluation points
# 			X_eval = X_SA_bests
# 			y_eval = objective_fun(X_eval)
# 			n_evals += len(y_eval)

# 			# Update dataset
# 			X = np.vstack((X, X_eval))
# 			y = np.hstack((y, y_eval))
# 			y_pred_mean = np.hstack((y_pred_mean, y_SA_bests))

# 			# update linear regression model
# 			linreg = HorseShoeBayesianLinearRegression(X=X, y=y, features=PolynomialFeatures(degree=order))
			

# 			''' Log
# 			==========='''

# 			optimizer_log = dict(X=X, y=y, y_pred_mean = y_pred_mean)
# 			if log_fun is not None:
# 				log_fun(optimizer_log)
    
# 			n_iter  += 1

# 	return optimizer_log

