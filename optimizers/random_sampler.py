'''
Author: Lucas Rath
'''
from typing import Callable, List, Tuple, Union
import numpy as np
import time
from CBOSS.models.search_space import ProductSpace

def random_sampler( 
	eval_fun:Callable[[np.ndarray],Tuple[np.ndarray,np.ndarray,np.ndarray]],
	X:np.ndarray,
	y:np.ndarray, 
	c:np.ndarray,
	l:np.ndarray,
	product_space:ProductSpace,
	evalBudget:int=100, 
	log_fun:Callable[[dict],None] = None
):
	''' Runs random sampler for MINIMIZING mixed (binary and categorical) functions
	'''
	assert X.ndim==y.ndim==l.ndim==c.ndim==2 and len(X)==len(y)==len(l)==len(c)
	assert y.shape[1]==1 and l.shape[1]==1, f'y.shape={y.shape}, l.shape={l.shape} must be (N,1)'

	n_evals = len(X)  # we count the given dataset as evaluations
	nbr_evals_left = evalBudget - n_evals

	time_start = time.time()
	# sample model
	X_eval = product_space.sample(N=nbr_evals_left)

	# Evaluate expensive model
	y_eval, c_eval, l_eval = eval_fun(X_eval)

	time_end = time.time()

	# Update dataset
	X   = np.vstack((X, X_eval))
	y   = np.vstack((y, y_eval))
	c   = np.vstack((c, c_eval))
	l   = np.vstack((l, l_eval))

	optimizer_log = dict(
		X=X, y=y, c=c, l=l,  
		walltime = np.arange(len(y)) / len(y) * (time_end - time_start)
	)

	if log_fun is not None:
		log_fun(log=optimizer_log)

	return optimizer_log