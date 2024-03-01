'''
Author: Lucas Rath
'''

from typing import Callable, Union, Tuple
import numpy as np
from scipy.special import softmax
import time
from CBOSS.models.search_space import BinarySpace, NominalSpace, ProductSpace

def simulatedAnnealing( 
	eval_fun:Callable[[np.ndarray],Tuple[np.ndarray,np.ndarray,np.ndarray]],
	X:np.ndarray,
	y:np.ndarray, 
	c:np.ndarray,
	l:np.ndarray,
	product_space:ProductSpace,
	evalBudget:int = 100, 
	T_point:tuple = (0.5, 0.005),
	log_fun:Callable[[dict],None] = None
) -> dict:
    ''' Runs simulated annealing algorithm for MINIMIZING mixed (binary and categorical) functions
    '''
    assert X.ndim==y.ndim==l.ndim==c.ndim==2 and len(X)==len(y)==len(l)==len(c)
    assert y.shape[1]==1 and l.shape[1]==1, f'y.shape={y.shape}, l.shape={l.shape} must be (N,1)'

    def constrained_objective(X_pred):
        y,c,l = eval_fun(X_pred)
        return np.where((c <= 0).all(1) & l==1, y, 1e50)[:,0]

    time_start = time.time()
    X, _ = _simulated_annealing(
        objective     		 = constrained_objective,
        product_space 		 = product_space,
        evalBudget 		     = evalBudget, 
        X_init 				 = X,
        y_init 				 = y[:,0],
        T_point 			 = T_point,
        stop_criteria_factor = np.inf
    )
    time_end = time.time()

    # Update dataset
    y, c, l = eval_fun(X)

    optimizer_log = dict(
        X=X, y=y, c=c, l=l,
        walltime = np.arange(len(y)) / len(y) * (time_end - time_start)
    )

    if log_fun is not None:
        log_fun(log=optimizer_log)

    return optimizer_log


def _simulated_annealing(
        objective:Callable[[np.ndarray],np.ndarray], 
        product_space:ProductSpace,
        evalBudget:int,
        n_init_samples:int = None,
        X_init:np.ndarray = None,
        y_init:np.ndarray = None,
        T_point:Tuple[int,int] = (0.5, 0.005),
        stop_criteria_factor:int = 2,
        random_generator:np.random.RandomState = np.random
):
    ''' Runs simulated annealing algorithm for MINIMIZING mixed (binary and categorical) functions
        f(X) = objective(X)

    Arguments:
        - evalBudget: number of total evaluations allowed. The initial samples count as budget
        - n_init_samples: number of random initial samples to be drawn using `objective`
        - X_init
        - y_init
        - stop_criteria_factor: the algorithm will stop early if it does make make any progress for
        <stop_criteria_factor * nbr_dimensions> iterations
    Returns: 
        - (X, y)
    '''
    assert np.all([isinstance(sp, BinarySpace) or isinstance(sp, NominalSpace) for sp in product_space.subspaces]),\
        'This optimizer only supports product spaces of Binary and Nominal variables'
    assert (n_init_samples is None) ^ (X_init is None),\
        'either `n_init_samples` or `X_init` must be provided'
    assert X_init is None or X_init.ndim == 2, 'X.ndim must be equal 2'
    assert y_init is None or y_init.ndim == 1, 'y.ndim must be equal 1'

    # estimate the number of iterations
    n_iter_mean = int(evalBudget // np.mean([len(sp.bounds) for sp in product_space.subspaces]))
    # Calculate the decay needed such that at iteration number `n_iter`: the temperature T==T_point[1]
    T = T_point[0] # Set initial temperature
    decay = (T_point[1]/T_point[0]) ** (1/(n_iter_mean))  # temperature decay
    cool = lambda T: decay*T # cooling scheduler

    # Set initial condition and evaluate objective
    if n_init_samples is not None:
        X_init = product_space.sample(N=n_init_samples, random_generator=random_generator)
        y_init = objective(X_init)
    if X_init is not None and y_init is None:
        y_init = objective(X_init)

    n_evals, n_vars = X_init.shape

    # Declare arrays to save solutions
    X = np.zeros((evalBudget, n_vars), dtype=object) * np.nan
    y = np.zeros(evalBudget) * np.nan
    X[:n_evals] = X_init
    y[:n_evals] = y_init

    # start with the best sample
    best_idx = np.nanargmin(y)
    curr_x = X[best_idx]
    curr_y = y[best_idx]

    iter_no_progress = 0

    # Run simulated annealing
    while n_evals < evalBudget:

        # Decrease T according to cooling schedule
        T = cool(T)

        # sample variable to switch
        flip_var = random_generator.randint(len(product_space))

        categories = product_space.subspaces[flip_var].bounds
        nbr_cat = len(categories)
        n_new_evals = nbr_cat - 1  # we have to evaluate all the other variants but the current one

        if n_evals + n_new_evals > evalBudget:
            break

        X_candidates = np.tile(curr_x, reps=[nbr_cat,1])
        X_candidates[:,flip_var] = categories

        # evaluate X_candidates except for curr_x
        mask_curr = np.all(curr_x == X_candidates, axis=1)
        y_candidates = np.zeros(nbr_cat)
        y_candidates[mask_curr]   = curr_y
        # y_candidates[~ mask_curr] = objective(X_candidates[~ mask_curr])
        y_candidates[~ mask_curr] = np.concatenate([
            objective(X_candidates[None,idx])
            for idx in np.where(~mask_curr)[0]
        ])

        # store new evaluations
        X[n_evals:n_evals+n_new_evals] = X_candidates[~ mask_curr]
        y[n_evals:n_evals+n_new_evals] = y_candidates[~ mask_curr]

        n_evals += n_new_evals

        idx_select = random_generator.choice(
            nbr_cat, 1,
            p = softmax( - np.where(np.isnan(y_candidates),1e10,y_candidates) / T )
        )[0]

        curr_x = X_candidates[idx_select]
        curr_y = y_candidates[idx_select]

        iter_no_progress = iter_no_progress+1 if mask_curr[idx_select] else 0
        if iter_no_progress >= stop_criteria_factor * n_vars:
            break

    X = X[:n_evals]
    y = y[:n_evals]

    return (X, y)