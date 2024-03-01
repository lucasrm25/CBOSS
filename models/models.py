''' Author: Lucas Rath
'''

from typing import List, Dict, Callable, Union, Tuple, Any, TypeVar
from pathlib import Path
import math
from functools import cached_property
from scipy.integrate import solve_ivp
import warnings
import numpy as np
import pandas as pd
import json
from scipy import stats
from scipy.signal import correlation_lags, correlate
with warnings.catch_warnings():
	import pynumdiff
import yaml
from tqdm import tqdm
import subprocess
import matplotlib.pyplot as plt 
from matplotlib import gridspec
from sklearn.preprocessing import PolynomialFeatures

from CBOSS.models.search_space import ProductSpace, BinarySpace, NominalSpace
from CBOSS.models import batch_ode


class IModel():
	''' Abstract/Interface class for general constrained optimization problems
	'''
	
	def __init__(self, product_space:ProductSpace):
		self.product_space = product_space

	def evaluate_objective(self, X:np.ndarray) -> np.ndarray:
		''' Objective function f(X) to be MINIMIZED
		Args:
			- X: <N,nx>
		Returns:
			- y: <N,1>
		'''
		raise NotImplementedError

	def evaluate_constraints(self, X:np.ndarray) -> np.ndarray:
		''' Constraints: c(X) <= 0
		Args:
			- X: <N,nx>
		Returns:
			- c: <N,nc>
		'''
		raise NotImplementedError

	def evaluate_flag(self, X:np.ndarray) -> np.ndarray:
		''' Constraints: c(X) <= 0
		Args:
			- X: <N,nx>
		Returns:
			- l: <N,1>
		'''
		raise NotImplementedError

	def evaluate(self, X:np.ndarray, extra_info:bool=False) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
		''' Evaluate objective, constraint and success functions
		Args:
			- X: <N,nx>
		Returns:
			- y: <N,1>   objective function
			- c: <N,nc>	 constriants
			- l: <N,1>	 success/failure flag
		'''
		# raise NotImplementedError
		y = self.evaluate_objective(X=X)
		c = self.evaluate_constraints(X=X)
		l = self.evaluate_flag(X=X)
		return y, c, l

	@cached_property
	def feature_names(self) -> List[str]:
		''' 
		Returns:
			- Array[str,nx]
		'''
		return [sp.var_name for sp in self.product_space.subspaces]

	def __len__(self) -> int:
		return len(self.product_space)

	def sample(self, nbr_samples:int, random_generator=np.random):
		''' generate nbr_samples random samples considering binary and categorical variable domains
		Returns:
			- X: <N, nx>
		'''
		X = self.product_space.sample(N=nbr_samples, random_generator=random_generator)
		return X

	def __str__(self) -> str:
		return f'Model defined over a {self.product_space}'

	def __repr__(self) -> str:
		return self.__str__()

class ICachedModel(IModel):
	''' Interface for models that store previous model evaluations in a .csv dataset file
	'''
	y_keys:List[str] = None
	c_keys:List[str] = None
	l_keys:List[str] = None

	def __init__(self, 
			product_space:ProductSpace, 
			results_path:str = None, 
			dataset_name:str = 'dataset.csv',
			restart_dataset:bool = False,
			drop_duplicates:bool = True
	):
		'''
		NOTE: if results_path = None, then no dataset will be saved or loaded
		'''
		super().__init__(product_space=product_space)
		self.product_space = product_space
		self.results_path = Path(results_path)
		self.dataset_name = dataset_name
		self.drop_duplicates = drop_duplicates
		# index for the dataframe
		self.dataset_index = [ss.var_name for ss in self.product_space.subspaces]
		# mapping var_name:index
		self.idx_map = {space.var_name:idx for idx,space in enumerate(self.product_space.subspaces)}

		self.dataset = None
		if not restart_dataset:
			self.load()	# load dataset

	def append(self, X:np.ndarray, **kwargs):
		newdata = pd.DataFrame( 
			data = dict( 
				**{k:Xc for k,Xc in zip(self.dataset_index, X.T)},
				**kwargs
			)
		).set_index(self.dataset_index)

		self.dataset = pd.concat((
			self.dataset,
			newdata,
		))

		# drop duplicate indices
		if self.drop_duplicates:
			self.dataset = self.dataset.groupby(level=self.dataset.index.names).last()

	def _evaluate(self, X:np.ndarray, verbose:bool=False) -> Dict[str,np.ndarray]:
		''' Evaluates model
		Returns:
			- Dict[str,np.ndarray] of metrics which will be appended to the dataset
		'''
		raise NotImplementedError
	
	def evaluate(self, X:np.ndarray, save:bool=False, verbose:bool=False) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
		''' Evaluate objective, constraint and success functions
		Args:
			- X: <N,nx>
		Returns:
			- y: <N,1>   objective function
			- c: <N,nc>	 constriants
			- l: <N,1>	 success/failure flag
		'''
		if self.dataset is not None and len(self.dataset) > 0:
			X_new = X[[tuple(x) not in self.dataset.index for x in X]]
		else:
			X_new = X

		if len(X_new) > 0:
			metrics:dict = self._evaluate(X=X_new, verbose=verbose)
			self.append( X=X_new, **metrics)
		if save:
			self.save()

		dataset_X = self.dataset.loc[X.tolist()]

		y = dataset_X[self.y_keys].to_numpy()
		c = dataset_X[self.c_keys].to_numpy()
		l = dataset_X[self.l_keys].to_numpy()

		return y, c, l

	def save(self):
		if self.results_path is not None:
			self.dataset.to_csv(self.results_path / self.dataset_name, index=True)

	def load(self):
		if self.results_path is not None:
			path = self.results_path / self.dataset_name
			if path.exists():
				self.dataset = pd.read_csv(path).set_index(self.dataset_index)
				# self.dataset = self.dataset.rename( {old:new for old,new in zip(self.dataset.columns[:-3],self.dataset_index)}, axis='columns' )


'''-------------------------------'''

class Constrained_Binary_Quad_Problem(IModel):
	'''
	Defines the Constrained Binary Quadratic Programming Problem:

		max   x^T Q_f x - lambda_f*|x|_1
		s.t.  x^T Q_c x + lambda_c*|x|_1 <= 0

	Note: as the correlation length `alpha` increases, Q changes from a 
	nearly diagonal to a denser matrix, making the optimization more challenging
	'''

	@staticmethod
	def calculate_quadratic_matrix(n_vars:int, alpha:float):
		# evaluate decay function
		i = np.linspace(1,n_vars,n_vars)
		j = np.linspace(1,n_vars,n_vars)
		K = lambda s,t: np.exp(-1*(s-t)**2/alpha)
		decay = K(i[:,None], j[None,:])
		# Generate random quadratic model and apply exponential decay to Q
		Q  = np.random.randn(n_vars, n_vars)
		return Q * decay

	def __init__(self, n_vars:int, alpha_f:float, lambda_f:float, alpha_c:float, lambda_c:float):
		super().__init__(product_space = BinarySpace() * n_vars )
		self.lambda_f = lambda_f	
		self.Q_f = self.calculate_quadratic_matrix(n_vars=n_vars, alpha=alpha_f)
		self.lambda_c = lambda_c
		self.Q_c = self.calculate_quadratic_matrix(n_vars=n_vars, alpha=alpha_c)

	def evaluate_objective(self, X:np.ndarray) -> np.ndarray:
		''' Objective function to be MINIMIZED
		Args:
			- Xoh: <N,n_vars>
		Returns:
			- y: <N,1>
		'''
		y = (X.dot(self.Q_f)*X).sum(axis=1) - self.lambda_f * np.sum(X,axis=1)
		return -y[:,None]  # here a minus becouse the original function is to be maximized

	def evaluate_constraints(self, X:np.ndarray) -> np.ndarray:
		''' Constraints: c(X) <= 0
		Args:
			- Xoh: <N,n_vars>
		Returns:
			- c: <N,1>
		'''
		c = (X.dot(self.Q_c)*X).sum(axis=1) + self.lambda_c * np.sum(X,axis=1)
		return c[:,None]

	def evaluate(self, X):
		''' Evaluate objective function to be MINIMIZED and constraints
		Args:
			- X: <N,d>	binary variable representing preventing effort at each stage
		Returns:
			- y: <N,1>  cost
			- c: <N,d>  constraint
			- l: <N,1>	success/failure flag
		'''
		y = self.evaluate_objective(X=X)
		c = self.evaluate_constraints(X=X)
		l = np.ones_like(y)		# all simulations are successful
		return y, c, l

'''-------------------------------'''

class Contamination_Control_Problem(IModel):
	'''
	Defines the Contamination Control Problem:
		Hu, Y., Hu, J., Xu, Y., Wang, F., and Cao, R. Z. (2010) Contamination control in food supply chain.

		Z_i - fraction of contaminated food at stage i,  1<=i<=d
		c_i - cost for prevention effort at stage i
		Λ_i - contamination spread rate when prevention NO effort is taken
		Γ_i - contamination decrease rate when prevention effort is taken
		x_i ∈ {0, 1} - decision variable associated with the prevention effort at stage i. 

		Dynamics for food contamination:
			Z_i = Λ_i(1 - xi)(1 - Z_{i-1}) + (1 - Γ_i x_i) Z_{i-1}

		U - upper limit for the fraction of contaminated food
		epsilon - upper limit for the probability of the fraction of contaminated food exceeding 

		Thus, the goal is to decide for each stage whether to implement a prevention 
  		effort in order to minimize the cost while ensuring the fraction of contaminated food 
    	does not exceed an upper limit U_i with probability at least 1 - epsilon. 
  		The random variables Λ_i, Γ_i and Z_1 follow beta-distributions.
		
		min   X c + alpha_f |x|_1
		s.t.  sum_{i=1}^{d} 1/T sum_{k=1}^{T} 1_{Z_k > U_i} >= 1 - epsilon

		where 
  			- X = [x]^{i=1...d}
			- c = [x]^{i=1...d}
	'''
 
	def __init__(self, 
		d=25, c=1.0, T=10**2, U=0.1, epsilon=0.05, 
		Z0_dist = stats.beta(a=1., b=30.),
		L_dist  = stats.beta(a=1., b=17./3),
		G_dist  = stats.beta(a=1., b=3./7),
		):

		super().__init__(product_space = BinarySpace() * d )
		self.d = d
		self.T = T
		self.U = U
		self.epsilon = epsilon
		self.c = c
		self.L_dist = L_dist
		self.G_dist = G_dist
		self.Z0_dist = Z0_dist

	def _simulate(self, X:np.ndarray) -> np.ndarray:
		''' Evaluate objective function to be MINIMIZED and constraints
		Args:
			- X: <N,d>		binary variable representing preventing effort at each stage
		Returns:
			- Z: <N,d+1,T>  contamination
			- cost: <1,>	cost
		'''
		N, d = X.shape
		assert d == self.d, f'X:<{N},{d}> expected <{N},{self.d}>'

		Z = np.zeros((N, d+1, self.T))

		# generate initial fraction of contamination
		Z[:,0,:] = self.Z0_dist.rvs(size=(N,self.T))
		# generate contamination spread rates
		L = self.L_dist.rvs(size=(N,d,self.T))
		# generate contamination decrease rates 
		G = self.G_dist.rvs(size=(N,d,self.T))

		# Determinating fraction of contamination at each stage
		for di in range(d):
			Z[:,di+1,:] = L[:,di,:] * (1 - X[:,di,None]) * (1 - Z[:,di,:]) + (1 - G[:,di,:]*X[:,di,None]) * Z[:,di,:]

		# calculate cost
		cost = (self.c * X).sum(1,keepdims=True)
		
		return Z, cost

	def evaluate(self, X):
		''' Evaluate objective function to be MINIMIZED and constraints
		Args:
			- X: <N,d>	binary variable representing preventing effort at each stage
		Returns:
			- y: <N,1>  cost
			- c: <N,d>  constraint violation at each stage d
			- l: <N,1>	success/failure flag
		'''
		Z, cost = self._simulate(X=X)

		# probability that the contamination is below threshold U must be greater than 1-epsilon
		# p(Z <= self.U) >= (1-self.epsilon)  ->  (1-self.epsilon) - p(Z <= self.U) <= 0
		constraint = (1 - self.epsilon) - (Z <= self.U).mean(-1)
		constraint = constraint.max(1,keepdims=True)

		y = cost
		c = constraint
		l = np.ones_like(y)		# all simulations are successful
		return y, c, l


''' 	 	Dynamical Systems
-------------------------------'''

class Sparse_Identification_Nonlinear_Dynamical_Systems(IModel):

	t_eval:np.ndarray 		 = np.arange(0,1,0.01)		# simulation time
	n:int 					 = 1 						# number of ODE dimensions
	k:int 					 = 2
	x_0 					 = np.array([0,0,0])
	Xi_true 				 = np.ones((10,1))			# <nm,n>
	coeff_eps_count:float    = 1e-5
	coeff_L1norm_thres:float = 2e1
	weight_complexity:float	 = 1e-2
	z_meas_std:float 		 = 0.

	TVR_filt_nbriter:int 	 = 10
	TVR_filt_reg:float 	 	 = 1e-2

	y_keys:List[str]	= ['AICc']
	c_keys:List[str] 	= ['log_coeff_L1norm_over_thres']
	l_keys:List[str]	= ['success']

	def __init__(
		self, 
		# k:int,
		# results_path:str = None, 
		# dataset_name:str = 'dataset.csv',
		# restart_dataset:bool = True,
		random_generator=np.random
	):
		'''
		Args:
			- k: max polynomial order of each dimension of the differential equation 
		'''
		# self.k = k
		# self.features = PolynomialFeatures(degree=self.k, interaction_only=False, include_bias=True, order='C')
		# self.features.fit_transform(np.ones((1,self.n)))	# fake inputs to init boring sklearn

		# assert measurements are equaly spaces
		dt = (self.t_eval[1:] - self.t_eval[:-1]).mean()
		assert (self.t_eval[1:] - self.t_eval[:-1]).std() < 1e-8

		''' Simulate true system '''
		self.Z_true, success = self.simulate_true()
		assert success == True, 'True simulation failed, please choose other true ODE coefficients'
		
		# add noise
		self.Z_meas  = self.Z_true  + random_generator.normal(loc=0., scale=self.z_meas_std, size=self.Z_true.shape)

		# calculate true derivative
		self.dZ_true = np.gradient(self.Z_true, self.t_eval, axis=0)
		# calculate apparent derivative (finite-differences) - (without filtering)
		self.dZ_meas = np.gradient(self.Z_meas, self.t_eval, axis=0)

		# (self.TVR_filt_nbriter, self.TVR_filt_reg) = (10, 1e-2)

		# use Total Variation Regularization to remove noise from signal and its derivative
		self.Z_meas_filt  = np.zeros_like(self.Z_meas)
		self.dZ_meas_filt = np.zeros_like(self.dZ_meas)
		for ni in range(self.n):
			self.Z_meas_filt[:,ni], self.dZ_meas_filt[:,ni] = pynumdiff.total_variation_regularization.iterative_velocity(
				x=self.Z_meas[:,ni], 
				dt=dt,
				params = (self.TVR_filt_nbriter, self.TVR_filt_reg)
			)

		# fig = self.plot_measurements_and_filter()

		# store feature matrix
		self.Theta_Z_meas_filt = self.features.fit_transform(X=self.Z_meas_filt)

		var_names = np.concatenate([ f'd{self.n_names[ni]}:' + np.asarray(self.monomial_names) for ni in range(self.n) ])
		assert len(var_names) == self.nx, 'Error naming variables, please check class definition.'
		super().__init__(
			product_space 	= math.prod([BinarySpace(var_name=vn) for vn in var_names]),  # BinarySpace() * self.nx, 
			# results_path	= results_path, 
			# dataset_name	= dataset_name,
			# restart_dataset	= restart_dataset
		)

	@cached_property
	def nx(self) -> int:
		return self.nm * self.n
		# nbr_coefficients = 0
		# for l in range(self.k+1):
		# 	nbr_coefficients += len(list(combinations_with_replacement( range(self.n), l )))
		# nbr_coefficients *= self.n
		# return nbr_coefficients

	@cached_property
	def features(self):
		self._features = PolynomialFeatures(degree=self.k, interaction_only=False, include_bias=True, order='C')
		self._features.fit_transform(np.ones((1,self.n)))	# fake inputs to init boring sklearn
		return self._features

	@cached_property
	def nm(self) -> int:
		''' max number of monomials per ODE dimension
		'''
		return self.features.n_output_features_

	@cached_property
	def n_names(self) -> List[str]:
		''' Returns the name of each ODE dimension as a <n> string array.
		This can and should be overriden by the inherited child class
		'''
		return [f'x{ni}' for ni in range(self.n)]

	@cached_property
	def monomial_names(self) -> List[str]:
		''' 
		Returns:
			- str <n,pn>: array of strings indicating the name of the monomials per dimension
			such as ['1', 'x0', 'x1', 'x2', 'x0x0', 'x0x1', ...]
		'''
		return self.features.get_feature_names_out(self.n_names)

	def print_equation(self, Xi) -> List[str]:
		'''
		Args:
			- Xi: <N,nm,n> batch array of coefficients
		Returns:
			- equations: <N,[str]>
		'''
		equations = []
		for xi in Xi:
			header = ('').join([' '*10] + ['{:>10s}'.format(f"'{n}dot'") for n in self.n_names])
			xi_str = '\n'.join([
				''.join(
					['{:>10s}'.format(f"'{self.monomial_names[i]}'")] +
					[
						f'{xi_ij:10.3f}' if np.abs(xi_ij)>0. else ' '*10  for xi_ij in xi_i
					]
				)
				for i, xi_i in enumerate(xi)
			])
			equations += [f'{header}\n{xi_str}']
		return equations

	def print_samples(self, X:np.ndarray) -> List[str]:
		Xext = X.reshape((-1, self.n, self.nx//self.n)).astype(bool)	# <N,n,np>
		Xi = self.fit_coefficients(X=X)
		equations = []
		for Xi_i, Xext_i in zip(Xi,Xext):
			# join coefficient values and monomials
			equation_strarray = [		# str<n,np>
				[
					f'{Xi_i[npi,ni]: 0.1e} {self.monomial_names[npi] if npi > 0 else "":3s}' # if Xext_i[ni,npi] else ''
					for npi in range(self.nm) if Xext_i[ni,npi]
				]
				for ni in range(self.n)
			]
			equations += ['\n'.join([
				' + '.join(equation_strarray[ni])
				for ni in range(self.n) 
			])]
		# print(equations[0])
		return equations 

	def Theta(self, Z:np.ndarray) -> np.ndarray:
		''' Calculates the features Theta(Z) such that  dZ/dt[i] = Theta(Z)[i] @ Xi[i]
		where Xi are the coefficient of the monomials in Theta(Z)
		Args:
			- Z: <N,n>
		Returns:
			- Theta(Z): <N,np>
		'''
		return self.features.fit_transform(X=Z)

	def fit_coefficients(self, X:np.ndarray) -> np.ndarray:
		'''
		Args:
			- X: <N,nx>
		Returns:
			- Xi: <N,nm>
		'''
		N = len(X)
		Xext = X.reshape((-1, self.n, self.nx//self.n)).astype(bool)
		Xi = np.zeros((N, self.nx//self.n, self.n))
		for Ni in range(N):
			for ni in range(self.n):
				res = np.linalg.lstsq(self.Theta_Z_meas_filt[:,Xext[Ni,ni,:]], self.dZ_meas_filt[:,ni], rcond=None)
				# res = sp.linalg.lstsq(self.Theta_Z_meas_filt[:,Xext[Ni,ni,:]], self.dZ_meas_filt[:,ni], check_finite=True, lapack_driver='gelsy')

				idx_coeffs = Xext[Ni,ni,:]
				Xi[Ni,idx_coeffs,ni] = res[0]
		return Xi

	def dz_dt_fun(self, z:np.ndarray, Xi:np.ndarray):
		return np.einsum('...f,...fn->...n', self.Theta(Z=z), Xi)

	def simulate_true(self) -> Tuple[np.ndarray, np.ndarray]:
		'''
		Returns:
			- Z: <nt,n>  batch solution of the ODE
			- success: bool
		'''
		Z, success = self.simulate(Xi=self.Xi_true[None])
		return Z[0], success[0]

	def simulate(self, Xi:np.ndarray, method=['batchode','scipy'][0]) -> Tuple[np.ndarray, np.ndarray]:
		''' Fixed step size integration of the dynamical system
		Args:
			- Xi: <N,nm,n> batch array of coefficients
		Returns:
			- Z: <N,nt,n>  batch solution of the ODE
			- success: <N,>
		'''
		assert Xi.ndim==3 and Xi.shape[-2:] == (self.nm, self.n)
		N = Xi.shape[0]
		nt = len(self.t_eval)

		if method == 'batchode':
			res = batch_ode.eRK45_eODE.solve(
				f  = lambda p,t,z: self.dz_dt_fun(z=z,Xi=p), 
				p  = Xi, 
				x0 = np.tile(self.x_0, (N,1)), 
				t  = np.tile(self.t_eval,(N,1))
			)
			Z = res.x
			success = res.success

		elif method == 'scipy':
			Z = np.zeros((N,nt,self.n)) * np.nan
			success = np.zeros(N)

			with warnings.catch_warnings(record=True) as w:
				for i, Xi_i in enumerate(Xi):
					res_scipy = solve_ivp( 
						fun = lambda t, z, args: self.dz_dt_fun(z=z[None], Xi=args[None])[0],
						args = (Xi_i,),
						t_span = (self.t_eval.min(), self.t_eval.max()),
						t_eval = self.t_eval,
						y0 = self.x_0,
						method = 'RK45',
						vectorized = False,
						max_step = 1.,
						atol = 1e1, rtol = 1e1, # force fixed time step size
						dense_output=False
						# min_step = 1.,
					)
					Z[i] = res_scipy.y.T
					success[i] = res_scipy.success == 1.
					# assert np.all(self.t_eval == res.t), 'true simulation has not been integrated properly... recheck ODE and simulation parameters'
		else:
			raise NotImplementedError(f'Method: {method} not implemented')

		assert Z.shape == (N,nt,self.n)
		assert success.shape == (N,)
		return Z, success

	def calculate_metrics(self, Xi, Z_sim, success):
		''' Calculate model metrics
		'''
		
		''' Calculate prediction metrics
		'''
		prediction_error = np.einsum('tm,Nmn->Ntn',self.Theta_Z_meas_filt,Xi) - self.dZ_meas_filt	# <N,nt,n>

		norm_factor_pred = np.abs(self.dZ_meas_filt).std(0)			# <n,>
		MAE_pred_n  = np.abs(prediction_error).mean(1)				# <N,n>
		RMSE_pred_n = np.sqrt(np.square(prediction_error).mean(1))	# <N,n>
	
		metrics_pred = dict(
			# Xi 		      = Xi,
			MAE_pred      = MAE_pred_n.mean(-1),
			NMAE_pred     = (MAE_pred_n / norm_factor_pred).mean(-1),
			RMSE_pred     = RMSE_pred_n.mean(-1),
			NRMSE_pred    = (RMSE_pred_n / norm_factor_pred).mean(-1),
			coeff_L1norm  = np.abs(Xi).sum((1,2)),
			coeff_nbr	  = (np.abs(Xi) >= 1e-10).sum((1,2)), 		# == X.sum(1)
			coeff_nbr_thrs= (np.abs(Xi) >= self.coeff_eps_count).sum((1,2))
		)

		''' Calculate simulation metrics
		'''
		# norm_factor_sim = self.Z_meas_filt.max(0) - self.Z_meas_filt.min(0)
		norm_factor_sim = self.Z_meas_filt.std(0)
		
		simulation_error = Z_sim - self.Z_meas_filt
		# simulation_error = self.Z_meas_filt[None,:]

		# remove aparently stable simulations whose error is too large:
		# 	NMAE > 1e1
		success = success & ((np.abs(simulation_error).mean(1)/norm_factor_sim).mean(-1) < 1e1)
		#	max range sim > 1e1 * max range meas
		success = success & ( ((Z_sim.max(1)-Z_sim.min(1))/(self.Z_meas_filt.max(0)-self.Z_meas_filt.min(0))).mean(-1) <= 1e1 )

		simulation_error[~success] *= np.nan

		MAE_sim_n  = np.abs(simulation_error).mean(1)							# <N,n>
		RMSE_sim_n = np.sqrt(np.square(simulation_error).mean(1))				# <N,n>

		metrics_sim = dict(
			MAE_sim		  = MAE_sim_n.mean(-1),
			NMAE_sim 	  = (MAE_sim_n / norm_factor_sim).mean(-1),
			RMSE_sim 	  = RMSE_sim_n.mean(-1),
			NRMSE_sim 	  = (RMSE_sim_n / norm_factor_sim).mean(-1),
			success		  = success.astype(int),
		)

		''' Calculate additonal metrics used for objective function
		'''
		# nt = Z_sim.shape[1] # number of time steps
		# nt = 1  # number of test observations - in this case 1 single measurement
		k = np.maximum(metrics_pred['coeff_nbr'], 1)

		eps = 1e-50
		metrics_objectives = dict(
			log10_NMAE_sim = np.log10(metrics_sim['NMAE_sim'] + eps),
			log_coeff_L1norm_over_thres = np.log(metrics_pred['coeff_L1norm'] + eps) - np.log(self.coeff_L1norm_thres + eps),
			AICc = np.log10(metrics_sim['NMAE_sim'] + eps) + self.weight_complexity*np.log2(k + eps)
		)

		metrics = metrics_pred | metrics_sim | metrics_objectives
		return metrics

	def evaluate(self, X:np.ndarray, verbose:bool=False, extra_info:bool=False) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
		''' Calculates objective, constraint and success functions

		Based on the ODE terms chosen by X, this function:
			- finds the coefficients using least squares
			- calculates a cost function by comparing the fitted ODE results with the true simulation results
			- evaluates the constraint (number of coefficients larger than a threshold)

		Args:
			- X: <N,nx>		selector indicating which coefficients to include in the ODE
		Returns:
		Returns:
			- y: <N,1>   objective function
			- c: <N,nc>	 constriants
			- l: <N,1>	 success/failure flag
			- Xi: <N,nm,n> batch array of coefficients
			- Z_sim: <N,nt,n> batch solution of the ODE
		'''
		# Fit coefficients
		Xi = self.fit_coefficients(X=X)	# <N,nm,n>
		# Simulate and evaluate
		Z_sim, success = self.simulate(Xi=Xi, method='batchode')
		# calculate metrics
		metrics = self.calculate_metrics(Xi=Xi, Z_sim=Z_sim, success=success)

		y = np.vstack([metrics[k] for k in self.y_keys]).T
		c = np.vstack([metrics[k] for k in self.c_keys]).T
		l = np.vstack([metrics[k] for k in self.l_keys]).T

		return [y,c,l] + [{'metrics':metrics, 'Xi':Xi, 'Z_sim':Z_sim, 'success':success}] * extra_info

	def plot(self, X=None, Xi=None, Z_sim=None, success=None) -> List[plt.Figure]:
		assert X is not None
		if (Xi is None or Z_sim is None or success is None):
			# Fit coefficients
			Xi = self.fit_coefficients(X=X)	# <N,nm,n>
			# Simulate and evaluate
			Z_sim, success = self.simulate(Xi=Xi, method='batchode')
		return self._plot(X=X, Xi=Xi, Z_sim=Z_sim, success=success)

	def _plot(self, X, Xi, Z_sim, success) -> List[plt.Figure]:
		...

	def plot_measurements_and_filter(self, xlim=None) -> plt.Figure:
		fig, axs = plt.subplots(2,self.n, figsize=(20,6), sharex=True)
		for ni in range(self.n):
			axs[0,ni].plot(self.t_eval, self.Z_meas[:,ni], alpha=0.5, label=f'{self.n_names[ni]}_meas')
			axs[0,ni].plot(self.t_eval, self.Z_true[:,ni], '--', label=f'{self.n_names[ni]}_true')
			axs[0,ni].plot(self.t_eval, self.Z_meas_filt[:,ni], '-.', label=f'{self.n_names[ni]}_meas_filt')
			axs[0,ni].set_xlabel('time [s]')
			axs[0,ni].legend()
		for ni in range(self.n):
			axs[1,ni].plot(self.t_eval, self.dZ_meas[:,ni], alpha=0.5, label=f'd{self.n_names[ni]}_meas')
			axs[1,ni].plot(self.t_eval, self.dZ_true[:,ni], '--', label=f'd{self.n_names[ni]}_true')
			axs[1,ni].plot(self.t_eval, self.dZ_meas_filt[:,ni], '-.', label=f'd{self.n_names[ni]}_meas_filt')
			axs[1,ni].set_xlabel('time [s]')
			axs[1,ni].legend()
		if xlim is not None:
			axs[0,0].set_xlim(xlim)
		plt.suptitle(f'{self.__class__.__name__} - Measurements')
		plt.tight_layout()
		return fig

class SEIR_Model(Sparse_Identification_Nonlinear_Dynamical_Systems):
	''' SEIR (Susceptible, Exposed, Infectious, Removed) model, a basic epidemiological model

	References: 
		- https://github.com/tpetzoldt/covid
		- https://sites.me.ucsb.edu/~moehlis/APC514/tutorials/tutorial_seasonal/node4.html

	Dynamic equation
	    dS/dt = mu - beta * I * S - mu * S
		dE/dt = beta * S * I - (mu + alpha) * E
		dI/dt = alpha * E - (mu + gamma) * I
		dR/dt = gamma * I   or   R = 1 - (S + E + I)

	State variables Z: fractions of total population
		S = susceptible
		E = exposed
		I = infected
		R = recovered or deceased

	Coefficients:
		mu    = birth and death rates
		alpha = inverse of incubation period
		beta  = average contact rate
		gamma = inverse of mean infectious period

	'''

	n:int	= 3				# number of ODE dimensions
	n_names = ['S','E','I']
	t_eval  = np.arange(0, 150, 0.1)

	weight_complexity:float	 = 1e-2
	z_meas_std:float 		 = 1e-2
	# dz_meas_std:float 		 = 1e-3

	mu    	= 1e-5
	alpha 	= 5 **-1
	beta  	= 1.75
	gamma 	= 2 **-1

	S_0   	= 1 - 5e-4
	E_0   	= 4e-4
	I_0   	= 1e-4

	def __init__(
			self, 
			k:int=2, 
			coeff_eps_count:float = 1e-5,
			coeff_L1norm_thres:float = 2e1,
			random_generator=np.random
	):
		assert k >=2, 'k must be >= 2'
		self.k = k
		self.x_0 = np.array([self.S_0, self.E_0, self.I_0])
		self.Xi_true = np.zeros((self.nm, self.n))
		self.Xi_true[:10,:] = np.array([
			[self.mu, -self.mu, 0, 0, 0, 0, -self.beta, 0, 0, 0],
			[0, 0, - (self.mu + self.alpha), 0, 0, 0, self.beta, 0, 0, 0],
			[0, 0, self.alpha, - (self.mu + self.gamma), 0, 0, 0, 0, 0, 0]
		]).T
		self.coeff_eps_count = coeff_eps_count
		self.coeff_L1norm_thres = coeff_L1norm_thres
		super().__init__(random_generator=random_generator)

	def _plot(self, X, Xi, Z_sim, success) -> List[plt.Figure]:
		equations = self.print_equation(Xi)

		# idx = idx_improvement[0]
		figs = []
		for i, (x, xi, Zsim_i, success_i,equation) in enumerate(zip(X,Xi,Z_sim,success,equations)):

			''' Plot simulation result 
			# matplotlib.font_manager.get_font_names()
			'''
			Z_ref = [self.Z_true, self.Z_meas_filt][0]
			dZ_ref = [self.dZ_true, self.dZ_meas_filt][0]

			fig = plt.figure(figsize=(12,11))
			gs = gridspec.GridSpec(3, 6, height_ratios=[3, 1, 1])
			# 
			ax = fig.add_subplot(gs[0,:4])
			for ni in range(self.n):
				ax.plot(self.t_eval, Zsim_i[:,ni], label=rf'${self.n_names[ni]}_{{sim}}$')
				ax.plot(self.t_eval, self.Z_true[:,ni], '--', label=rf'${self.n_names[ni]}_{{meas,filt}}$')
			ax.set_xlabel('time')
			# ax.set_ylabel('y')
			plt.grid(True)
			plt.legend(loc='upper right')
			# 
			ax = fig.add_subplot(gs[0,4:])
			ax.annotate(
				equation, (0.5, 0.5), xycoords='axes fraction', 
				va='center', ha='center', fontname='Liberation Mono', linespacing=2
			)
			ax.axis('off')
			# 

			Z_ref = [self.Z_true, self.Z_meas_filt][1]
			dZ_ref = [self.dZ_true, self.dZ_meas_filt][1]
			# states
			# 
			ax = fig.add_subplot(gs[1,:2])
			ax.plot(self.t_eval, Z_ref[:,0], '-',color='tab:orange', label='x_meas')
			ax.plot(self.t_eval, Zsim_i[:,0], '--',color='tab:blue', label='x_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('x')
			ax.set_title(self.n_names[0])
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[1,2:4])
			ax.plot(self.t_eval, Z_ref[:,1], '-',color='tab:orange', label='y_meas')
			ax.plot(self.t_eval, Zsim_i[:,1], '--',color='tab:blue', label='y_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('y')
			ax.set_title(self.n_names[1])
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[1,4:])
			ax.plot(self.t_eval, Z_ref[:,2], '-',color='tab:orange', label='z_meas')
			ax.plot(self.t_eval, Zsim_i[:,2], '--',color='tab:blue', label='z_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('z')
			ax.set_title(self.n_names[2])
			plt.grid(True)
			# 
			# state derivatives
			# 
			ax = fig.add_subplot(gs[2,:2])
			ax.plot(self.t_eval, dZ_ref[:,0], '-', color='tab:orange', label='dx_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('x')
			ax.set_title(f'd{self.n_names[0]}')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[2,2:4])
			ax.plot(self.t_eval, dZ_ref[:,1], '-', color='tab:orange', label='dy_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('y')
			ax.set_title(f'd{self.n_names[1]}')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[2,4:])
			ax.plot(self.t_eval, dZ_ref[:,2], '-', color='tab:orange', label='dz_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('z')
			ax.set_title(f'd{self.n_names[2]}')
			plt.grid(True)
			# fig.suptitle(f'x: {x}  y: {y:.2f}\niteration: {it}')
			plt.tight_layout()
			
			figs += [fig]

		return figs

class Lorenz_Model(Sparse_Identification_Nonlinear_Dynamical_Systems):
	''' Lorenz Oscillator

	Dynamic equation
		dx/dt = -sigma x + sigma y
		dy/dt = rho x - y + - x z
		dz/dt = -beta z + x y
	'''
	n:int	= 3				# number of ODE dimensions
	n_names = ['x','y','z']
	t_eval  = np.arange(0, 20, 0.01)

	weight_complexity:float	 = 1e-2
	z_meas_std:float 		 = 1e0
	# dz_meas_std:float 		 = 1e-1

	sigma  	= 10.
	rho 	= 28.
	beta 	= 8./3.

	x_0   	= 10.
	y_0   	= 10.
	z_0   	= 10.

	def __init__(
			self, 
			k:int=2, 
			coeff_eps_count:float = 1e-5,
			coeff_L1norm_thres:float = 1e2,
			random_generator=np.random
		):
		self.k = k
		self.x_0 = np.array([self.x_0, self.y_0, self.z_0])
		self.Xi_true = np.zeros((self.nm, self.n))
		self.Xi_true[:10,:] = np.array([
			[0, -self.sigma, self.sigma, 0, 0, 0, 0, 0, 0, 0],
			[0, self.rho, - 1, 0, 0, 0, -1, 0, 0, 0],
			[0, 0, 0, -self.beta, 0, 1, 0, 0, 0, 0]
		]).T
		self.coeff_eps_count = coeff_eps_count
		self.coeff_L1norm_thres = coeff_L1norm_thres
		super().__init__(random_generator=random_generator)

	def _plot(self, X, Xi, Z_sim, success) -> List[plt.Figure]:
		equations = self.print_equation(Xi)

		# idx = idx_improvement[0]
		figs = []
		for i, (x, xi, Zsim_i, success_i,equation) in enumerate(zip(X,Xi,Z_sim,success,equations)):

			''' Plot simulation result 
			# matplotlib.font_manager.get_font_names()
			'''
			Z_true = [self.Z_true, self.Z_meas_filt][0]
			dZ_true = [self.dZ_true, self.dZ_meas_filt][0]

			fig = plt.figure(figsize=(12,11))
			gs = gridspec.GridSpec(3, 6, height_ratios=[3, 1, 1])
			# 
			ax = fig.add_subplot(gs[0,:3], projection='3d')
			ax.plot3D(*Zsim_i.T, label='sim')
			ax.plot3D(*Z_true.T, label='meas', alpha=0.5)
			ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
			plt.legend(loc='upper left')
			# 
			ax = fig.add_subplot(gs[0,3:])
			ax.annotate(
				equation, (0.5, 0.5), xycoords='axes fraction', 
				va='center', ha='center', fontname='Liberation Mono', linespacing=2
			)
			ax.axis('off')
			# 

			Z_true = [self.Z_true, self.Z_meas_filt][1]
			dZ_true = [self.dZ_true, self.dZ_meas_filt][1]
			# states
			# 
			ax = fig.add_subplot(gs[1,:2])
			ax.plot(self.t_eval, Z_true[:,0], '--',color='tab:orange', label='x_meas')
			ax.plot(self.t_eval, Zsim_i[:,0], '-',color='tab:blue', label='x_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('x')
			ax.set_title('x')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[1,2:4])
			ax.plot(self.t_eval, Z_true[:,1], '--',color='tab:orange', label='y_meas')
			ax.plot(self.t_eval, Zsim_i[:,1], '-',color='tab:blue', label='y_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('y')
			ax.set_title('y')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[1,4:])
			ax.plot(self.t_eval, Z_true[:,2], '--',color='tab:orange', label='z_meas')
			ax.plot(self.t_eval, Zsim_i[:,2], '-',color='tab:blue', label='z_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('z')
			ax.set_title('z')
			plt.grid(True)
			# 
			# state derivatives
			# 
			ax = fig.add_subplot(gs[2,:2])
			ax.plot(self.t_eval, dZ_true[:,0], '--', color='tab:orange', label='dx_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('x')
			ax.set_title('dx')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[2,2:4])
			ax.plot(self.t_eval, dZ_true[:,1], '--', color='tab:orange', label='dy_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('y')
			ax.set_title('dy')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[2,4:])
			ax.plot(self.t_eval, dZ_true[:,2], '--', color='tab:orange', label='dz_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('z')
			ax.set_title('dz')
			plt.grid(True)
			# fig.suptitle(f'x: {x}  y: {y:.2f}\niteration: {it}')
			plt.tight_layout()
			figs += [fig]
		return figs

class CylinderWake_Model(Sparse_Identification_Nonlinear_Dynamical_Systems):
	''' Mean-field approximation of the flow over a cylinder

	Dynamic equation
		dx/dt = mu x - omega y + A xz
		dy/dt = omega x + mu y + A yz
		dz/dt = -lambda z + lambda x^2 + lambda y^2

	References:
		- Nopack, B. (2003) A hierarchy of low-dimensional models for the transient and post-transient cylinder wake
	'''

	n:int	= 3				# number of ODE dimensions
	n_names = ['x','y','z']
	# t_eval  = np.arange(0, 100, 0.02)
	t_eval  = np.arange(0, 100, 0.1)

	weight_complexity:float	 = 1e-2
	z_meas_std:float 		 = 1e-2
	# dz_meas_std:float 		 = 1e-3

	omega 	= 1.
	mu 		= 0.1
	A 		= -1.
	lambd 	= 1.

	x_0   	= 0.001
	y_0   	= 0.
	z_0   	= 0.1

	def __init__(
		self, 
		k:int=2, 
		coeff_eps_count:float = 1e-5,
		coeff_L1norm_thres:float = 1e1,
		random_generator=np.random
	):
		self.k = k
		self.x_0 = np.array([self.x_0, self.y_0, self.z_0])
		self.Xi_true = np.zeros((self.nm, self.n))
		self.Xi_true[:10,:] = np.array([
			[0,  	self.mu, - self.omega, 0, 		0, 0, self.A, 0, 0, 0,					],
			[0,  	self.omega, self.mu, 0, 		0, 0, 0, 0, self.A, 0,					],
			[0,  	0, 0, -self.lambd, 				self.lambd, 0, 0, self.lambd, 0, 0,		],
		]).T
		self.coeff_eps_count = coeff_eps_count
		self.coeff_L1norm_thres = coeff_L1norm_thres
		super().__init__(random_generator=random_generator)

	def _plot(self, X, Xi, Z_sim, success) -> List[plt.Figure]:
		equations = self.print_equation(Xi)
		equations = self.print_equation(Xi)

		# idx = idx_improvement[0]
		figs = []
		for i, (x, xi, Zsim_i, success_i,equation) in enumerate(zip(X,Xi,Z_sim,success,equations)):

			''' Plot simulation result 
			# matplotlib.font_manager.get_font_names()
			'''
			Z_true = [self.Z_true, self.Z_meas_filt][0]
			dZ_true = [self.dZ_true, self.dZ_meas_filt][0]

			fig = plt.figure(figsize=(12,11))
			gs = gridspec.GridSpec(3, 6, height_ratios=[3, 1, 1])
			# 
			ax = fig.add_subplot(gs[0,:3], projection='3d')
			ax.plot3D(*Zsim_i.T, label='sim')
			ax.plot3D(*Z_true.T, label='meas', alpha=0.5)
			ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
			plt.legend(loc='upper left')
			# 
			ax = fig.add_subplot(gs[0,3:])
			ax.annotate(
				equation, (0.5, 0.5), xycoords='axes fraction', 
				va='center', ha='center', fontname='Liberation Mono', linespacing=2
			)
			ax.axis('off')


			Z_true = [self.Z_true, self.Z_meas_filt][1]
			dZ_true = [self.dZ_true, self.dZ_meas_filt][1]
			#
			# states
			# 
			ax = fig.add_subplot(gs[1,:2])
			ax.plot(self.t_eval, Z_true[:,0], '--',color='tab:orange', label='x_meas')
			ax.plot(self.t_eval, Zsim_i[:,0], '-',color='tab:blue', label='x_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('x')
			ax.set_title('x')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[1,2:4])
			ax.plot(self.t_eval, Z_true[:,1], '--',color='tab:orange', label='y_meas')
			ax.plot(self.t_eval, Zsim_i[:,1], '-',color='tab:blue', label='y_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('y')
			ax.set_title('y')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[1,4:])
			ax.plot(self.t_eval, Z_true[:,2], '--',color='tab:orange', label='z_meas')
			ax.plot(self.t_eval, Zsim_i[:,2], '-',color='tab:blue', label='z_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('z')
			ax.set_title('z')
			plt.grid(True)
			# 
			# state derivatives
			# 
			ax = fig.add_subplot(gs[2,:2])
			ax.plot(self.t_eval, dZ_true[:,0], '--', color='tab:orange', label='dx_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('x')
			ax.set_title('dx')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[2,2:4])
			ax.plot(self.t_eval, dZ_true[:,1], '--', color='tab:orange', label='dy_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('y')
			ax.set_title('dy')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[2,4:])
			ax.plot(self.t_eval, dZ_true[:,2], '--', color='tab:orange', label='dz_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('z')
			ax.set_title('dz')
			plt.grid(True)
			# fig.suptitle(f'x: {x}  y: {y:.2f}\niteration: {it}')
			plt.tight_layout()
			figs += [fig]
		return figs

class NonLinearDampedOscillator_Model(Sparse_Identification_Nonlinear_Dynamical_Systems):
	''' Mean-field

	Dynamic equation
		dx/dt = - alpha x^3 + beta 	y^3
		dy/dt = - beta  x^3 - alpha y^3
	'''
	n:int	= 2				# number of ODE dimensions
	n_names = ['x','y']
	t_eval  = np.arange(0, 35, 0.01)

	weight_complexity:float	 = 1e-2
	z_meas_std:float 		 = 1e-1

	alpha:float = 0.1
	beta:float  = 2.0

	x_0   	= 2.
	y_0   	= 0.

	def __init__(
			self, 
			k:int=4, 
			coeff_eps_count:float = 1e-5,
			coeff_L1norm_thres:float = 5.,
			random_generator=np.random
		):
		self.k = k
		self.x_0 = np.array([self.x_0, self.y_0])
		# ['1', 'x', 'y', 'x^2', 'x y', 'y^2', 'x^3', 'x^2 y', 'x y^2', 'y^3']
		self.Xi_true = np.zeros((self.nm, self.n))
		self.Xi_true[:10,:] = np.array([
			[0, 0, 0, 0, 0, 0, -self.alpha, 0, 0, self.beta,   ],
			[0, 0, 0, 0, 0, 0,  -self.beta, 0, 0, -self.alpha, ]
		]).T
		self.coeff_eps_count = coeff_eps_count
		self.coeff_L1norm_thres = coeff_L1norm_thres
		super().__init__(random_generator=random_generator)

	def _plot(self, X, Xi, Z_sim, success) -> List[plt.Figure]:
		equations = self.print_equation(Xi)

		# idx = idx_improvement[0]
		figs = []
		for i, (x, xi, Zsim_i, success_i,equation) in enumerate(zip(X,Xi,Z_sim,success,equations)):

			''' Plot simulation result 
			# matplotlib.font_manager.get_font_names()
			'''
			Z_true = [self.Z_true, self.Z_meas_filt][0]
			dZ_true = [self.dZ_true, self.dZ_meas_filt][0]

			fig = plt.figure(figsize=(12,11))
			gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 1])
			# 
			ax = fig.add_subplot(gs[0,0])
			# for ni in range(self.n):
			# 	ax.plot(self.t_eval, Zsim_i[:,ni], label=f'{self.n_names[ni]}_sim')
			# 	ax.plot(self.t_eval, self.Z_true[:,ni], '--', label=f'{self.n_names[ni]}_meas')

			ax.plot(*Zsim_i.T, linewidth=3, label='sim')
			ax.plot(*self.Z_true.T, '--', linewidth=3, label='meas')

			ax.set_xlabel('x')
			ax.set_ylabel('y')
			plt.grid(True)
			plt.legend(loc='upper right')
			# 
			ax = fig.add_subplot(gs[0,1])
			ax.annotate(
				equation, (0.5, 0.5), xycoords='axes fraction', 
				va='center', ha='center', fontname='Liberation Mono', linespacing=2
			)
			ax.axis('off')
			# 

			Z_true = [self.Z_true, self.Z_meas_filt][1]
			dZ_true = [self.dZ_true, self.dZ_meas_filt][1]
			# states
			# 
			ax = fig.add_subplot(gs[1,0])
			ax.plot(self.t_eval, Z_true[:,0], '--',color='tab:orange', label='x_meas')
			ax.plot(self.t_eval, Zsim_i[:,0], '-',color='tab:blue', label='x_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('x')
			ax.set_title(self.n_names[0])
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[1,1])
			ax.plot(self.t_eval, Z_true[:,1], '--',color='tab:orange', label='y_meas')
			ax.plot(self.t_eval, Zsim_i[:,1], '-',color='tab:blue', label='y_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('y')
			ax.set_title(self.n_names[1])
			plt.grid(True)
			# 
			# state derivatives
			# 
			ax = fig.add_subplot(gs[2,0])
			ax.plot(self.t_eval, dZ_true[:,0], '--', color='tab:orange', label='dx_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('x')
			ax.set_title(f'd{self.n_names[0]}')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[2,1])
			ax.plot(self.t_eval, dZ_true[:,1], '--', color='tab:orange', label='dy_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('y')
			ax.set_title(f'd{self.n_names[1]}')
			plt.grid(True)
			# fig.suptitle(f'x: {x}  y: {y:.2f}\niteration: {it}')
			plt.tight_layout()
			figs += [fig]

		return figs

class CoupledPendulum_Model(Sparse_Identification_Nonlinear_Dynamical_Systems):
	''' Mean-field approximation of the flow over a cylinder

	Dynamic equation
		ddtheta1/dt^2 = -g/l sin theta1 - k/m (theta1 - theta2)
		ddtheta2/dt^2 = -g/l sin theta2 + k/m (theta1 - theta2)
	'''

	n:int	= 4				# number of ODE dimensions
	n_names = ['theta1','theta2']
	# t_eval  = np.arange(0, 100, 0.02)
	t_eval  = np.arange(0, 30, 0.01)

	weight_complexity:float	= 1e-2
	z_meas_std:float 		= 1e-3
	# dz_meas_std:float 		= 1e-3

	g = 9.81
	l = 1.
	k = 1.
	m = 1.

	th1_0   	= 0.001
	th2_0   	= 0.

	def __init__(self, random_generator=np.random):
		raise NotImplementedError()

class ChuaOscillator_Model(Sparse_Identification_Nonlinear_Dynamical_Systems):
	''' Mean-field approximation of the flow over a cylinder

	Dynamic equation
		dx/dt = alpha (- x + y + f(x))
		dy/dt = beta( x - y + z )
		dz/dt = -gamma y

		f(x) = zeta x + (delta + zeta) ( |x+1| - |x-1| )
	'''

	n:int	= 3				# number of ODE dimensions
	n_names = ['x','y','z']
	# t_eval  = np.arange(0, 100, 0.02)
	t_eval  = np.arange(0, 30, 0.01)

	weight_complexity:float	= 1e-2
	z_meas_std:float 		= 1e-3
	# dz_meas_std:float 		= 1e-3

	p1 = 10.
	p2 = 15.
	p3 = 0.0385
	a  = -1.27
	b  = -0.68

	x_0   	= 0.001
	y_0   	= 0.
	z_0   	= 0.1

	def __init__(
			self, 
			k:int=3, 
			coeff_eps_count:float = 1e-5,
			coeff_L1norm_thres:float = 1e2,
			random_generator=np.random
		):
		self.k = k
		self.Xi_true = np.ones((3, 20))
		self.x_0 = np.array([self.x_0, self.y_0, self.z_0])
		self.coeff_eps_count = coeff_eps_count
		self.coeff_L1norm_thres = coeff_L1norm_thres
		super().__init__(random_generator=random_generator)

	def dz_dt_fun_true(self, z):
		x1,x2,x3 = z[0]
		f = self.b * x1 + 0.5 * (self.a - self.b) * ( np.abs(x1+1) - np.abs(x1-1) )
		dz_dt = np.array([[
			self.p1 * ( - x1 + x2 - f ),
			1 	    * ( + x1 - x2 + x3),
			-self.p2 * x2 - self.p3 * x3
		]])
		return dz_dt

	def simulate_true(self) -> Tuple[np.ndarray, np.ndarray]:
		'''
		Returns:
			- Z: <nt,n>  batch solution of the ODE
			- success: bool
		'''
		res = batch_ode.eRK45_eODE.solve(
			f  = lambda p,t,z: self.dz_dt_fun_true(z=z),
			p  = np.array([None]),
			x0 = self.x_0[None], 
			t  = self.t_eval[None]
		)
		Z = res.x
		success = res.success
		return Z[0], success[0]

	def _plot(self, X, Xi, Z_sim, success) -> List[plt.Figure]:
		equations = self.print_equation(Xi)

		# idx = idx_improvement[0]
		figs = []
		for i, (x, xi, Zsim_i, success_i,equation) in enumerate(zip(X,Xi,Z_sim,success,equations)):

			''' Plot simulation result 
			# matplotlib.font_manager.get_font_names()
			'''
			Z_true = [self.Z_true, self.Z_meas_filt][0]
			dZ_true = [self.dZ_true, self.dZ_meas_filt][0]

			fig = plt.figure(figsize=(12,11))
			gs = gridspec.GridSpec(3, 6, height_ratios=[3, 1, 1])
			# 
			ax = fig.add_subplot(gs[0,:3], projection='3d')
			ax.plot3D(*Zsim_i.T, label='sim')
			ax.plot3D(*Z_true.T, label='meas', alpha=0.5)
			ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
			plt.legend(loc='upper left')
			# 
			ax = fig.add_subplot(gs[0,3:])
			ax.annotate(
				equation, (0.5, 0.5), xycoords='axes fraction', 
				va='center', ha='center', fontname='Liberation Mono', linespacing=2
			)
			ax.axis('off')
			# 

			Z_true = [self.Z_true, self.Z_meas_filt][1]
			dZ_true = [self.dZ_true, self.dZ_meas_filt][1]
			# states
			# 
			ax = fig.add_subplot(gs[1,:2])
			ax.plot(self.t_eval, Z_true[:,0], '--',color='tab:orange', label='x_meas')
			ax.plot(self.t_eval, Zsim_i[:,0], '-',color='tab:blue', label='x_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('x')
			ax.set_title('x')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[1,2:4])
			ax.plot(self.t_eval, Z_true[:,1], '--',color='tab:orange', label='y_meas')
			ax.plot(self.t_eval, Zsim_i[:,1], '-',color='tab:blue', label='y_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('y')
			ax.set_title('y')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[1,4:])
			ax.plot(self.t_eval, Z_true[:,2], '--',color='tab:orange', label='z_meas')
			ax.plot(self.t_eval, Zsim_i[:,2], '-',color='tab:blue', label='z_sim')
			ax.set_xlabel('time')
			# ax.set_ylabel('z')
			ax.set_title('z')
			plt.grid(True)
			# 
			# state derivatives
			# 
			ax = fig.add_subplot(gs[2,:2])
			ax.plot(self.t_eval, dZ_true[:,0], '--', color='tab:orange', label='dx_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('x')
			ax.set_title('dx')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[2,2:4])
			ax.plot(self.t_eval, dZ_true[:,1], '--', color='tab:orange', label='dy_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('y')
			ax.set_title('dy')
			plt.grid(True)
			# 
			ax = fig.add_subplot(gs[2,4:])
			ax.plot(self.t_eval, dZ_true[:,2], '--', color='tab:orange', label='dz_meas')
			ax.set_xlabel('time')
			# ax.set_ylabel('z')
			ax.set_title('dz')
			plt.grid(True)
			# fig.suptitle(f'x: {x}  y: {y:.2f}\niteration: {it}')
			plt.tight_layout()
			figs += [fig]
		return figs
