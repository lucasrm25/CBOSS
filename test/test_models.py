''' Author: Lucas Rath
'''

import unittest
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from CBOSS.models.models import (
	Contamination_Control_Problem, FasterModel,SEIR_Model,Lorenz_Model, 
	CylinderWake_Model, NonLinearDampedOscillator_Model,
	ChuaOscillator_Model, NonLinearDampedOscillator_Model
)
from CBOSS.models import models
from CBOSS.utils.Tee import Tee

import warnings
# warnings.filterwarnings("error")
# np.seterr(all='warn')

class Test_Contamination_Control(unittest.TestCase):

	log_dir = Path(__file__).parent / 'log'

	def test_contamination(self, d:int=50, N:int=5):
		model = Contamination_Control_Problem(d=d)

		X = np.random.randint(2, size=(N,d))
		Z, cost = model._simulate(X=X)
		constraint = (1 - model.epsilon) - (Z <= model.U).mean(-1)

		Z_mean = Z.mean(-1)
		Z_std  = Z.std(-1)
		x = np.arange(Z_mean.shape[1])
		Ni = 0

		fig, axs = plt.subplots(2,1,figsize=(12,8),sharex='all')
		# axis 0
		axs[0].plot( x, Z_mean[0] )
		axs[0].fill_between(x=x, y1=Z_mean[Ni]-Z_std[Ni], y2=Z_mean[Ni]+Z_std[Ni], alpha=0.3, color='tab:purple')
		for Ti in range(1,model.T):
			axs[0].plot( Z[0,:,Ti] , alpha=0.2, color='tab:gray')
		axs[0].set_title(f'X={X[Ni,:]}')
		axs[0].set_ylim([0,1])
		axs[0].set_xlim([0,d])
		axs[0].set_xlabel('Stage')
		axs[0].set_ylabel('Fraction of food contamination')
		axs[0].grid()
		# axis 1
		axs[1].plot( x, constraint[Ni,:] )
		axs[1].grid()
		fig.tight_layout()
		fig.savefig( self.log_dir / f'{Path(__file__).stem}_contamination_control_sim.png' )
		plt.close()

		y,c,l = model.evaluate(X)

		self.assertEqual( y.shape, (N,1), msg='y.shape is wrong' )
		# self.assertEqual( c.shape, (N,1), msg='c.shape is wrong' )
		self.assertEqual( c.ndim, 2, msg='c.shape is wrong' )
		self.assertEqual( c.shape[0], N, msg='c.shape is wrong' )
		self.assertEqual( l.shape, (N,1), msg='l.shape is wrong' )

class Test_EquationDiscoveryModels(unittest.TestCase):
	
	cfg_file = Path(__file__).parent /'..'/'..'/'configs_equation_discovery.yaml'
	log_dir = Path(__file__).parent / 'log'
	
	model_names = [
		'NonLinearDampedOscillator_k5',
		'CylinderWake_k3',
		'Lorenz_k3',
		'SEIR_k3',
		'ChuaOscillator_k3',
	]
 
 
	def setUp(self) -> None:
		sns.set_style("darkgrid")
		# disable warnings for too many open figures
		plt.rcParams.update({'figure.max_open_warning': 0})
	
	def test_models(self, N:int=5000):

		with open(self.cfg_file, 'r') as f:
			cfg = yaml.safe_load(f)
		
		# for model_name, Model in models:
		for model_name in self.model_names:

			results_path = self.log_dir / f'models_{model_name}'
			results_path.mkdir(parents=True, exist_ok=True)

			print(f'\n---> Processing {model_name} in {results_path}\n')

			# model = Model()
			Model = getattr(models, cfg[model_name]['model_name'])
			randomstate = np.random.RandomState(cfg[model_name]['model']['random_seed_init_dataset'])
			model = Model(**cfg[model_name]['model']['kwargs'], random_generator=randomstate)

			d = model.nx

			x_true  = (model.Xi_true.T.reshape(1,-1)!=0).astype(int)
			x_allon = np.ones((1,d))
			x_test  = np.array([[0,0,0,1] + [0]*(model.nx-4)])

			X = model.sample(N-3).astype(int)
			X = np.concatenate((x_true, x_allon, x_test, X))

			print('evaluating samples...', end='')
			y, c, l, info = model.evaluate(X=X, verbose=True, extra_info=True)
			print('done\n')

			self.assertEqual( y.shape, (N,1), msg='y.shape is wrong' )
			self.assertEqual( c.shape, (N,1), msg='c.shape is wrong' )
			self.assertEqual( l.shape, (N,1), msg='l.shape is wrong' )

			''' Plot measurements
			'''
			fig = model.plot_measurements_and_filter()
			fig.savefig( results_path / 'measurements.png', dpi=500)


			''' Plot the simulation results of some random equations
			'''
			sns.set_style("whitegrid")
			# model.dataset[model.y_keys[0]]
			# dataset_s = model.dataset.sample(n=50, replace=False).copy()
			nbr_plots = 30
			(idxfeas,) = np.where((c<=0).all(1))
			idx_best = idxfeas[np.nanargmin(y[idxfeas])]
			idxs = np.concatenate( ([idx_best], np.arange(nbr_plots-1)) )
			X_s, y_s, c_s, l_s = X[idxs], y[idxs,0], c[idxs,0], l[idxs,0]
			Xi_s	 = info['Xi'][idxs]
			Z_sim_s	 = info['Z_sim'][idxs]
			success_s= info['success'][idxs]
			log10_NMAE_sim_s = info['metrics']['log10_NMAE_sim'][idxs]
			coeff_L1norm_s = info['metrics']['coeff_L1norm'][idxs]

			figs = model.plot(X=X_s, Xi=Xi_s, Z_sim=Z_sim_s, success=success_s)
			for i, (xi, yi, ci, li, log10_NMAE_sim_i, coeff_L1norm_i, fig) in enumerate(zip(X_s, y_s, c_s, l_s, log10_NMAE_sim_s, coeff_L1norm_s, figs)):
				title  = f'x: {xi}\ny: {yi:.2f}  c: {ci:.2f}  l: {li:.2f}\n'
				title += f'log10_NMAE_sim: {log10_NMAE_sim_i:.2f}  coeff_L1norm: {coeff_L1norm_i:.2f}'
				fig.suptitle(title)
				fig.tight_layout()
				figname = (i==0) * '_best' + (i==1) * '_true' + (i==2) * '_allon' + (i>2) * f'{i}'
				fig.savefig( results_path / f'{i:04d}_TEST_{figname}.png')
				plt.close()


			dataset = pd.DataFrame(data=info['metrics'])

			''' Plot distribution of some metrics
			'''
			sns.set_style("darkgrid")
			dataset_s = dataset.sample(n=min(N,5000), replace=False).copy()

			data_plot = pd.concat([
				dataset_s['success'], 
				np.log10(dataset_s['NMAE_pred']).rename('log10_NMAE_pred'),
				np.log10(dataset_s['NMAE_sim']).rename('log10_NMAE_sim'),
				dataset_s['AICc'],
				np.log(dataset_s['coeff_L1norm']).rename('log_coeff_L1norm'),
				dataset_s['coeff_nbr'],
			], axis=1)

			g = sns.PairGrid(data_plot, hue='success', hue_order=[True,False], palette="deep", corner=False)
			# g.map_lower(sns.kdeplot, fill=True, warn_singular=False, hue=None, color='tab:blue')
			g.map_lower(sns.histplot, color='tab:blue')
			g.map_lower(sns.scatterplot, alpha=0.1)
			try:
				g.map_upper(sns.kdeplot, fill=False, alpha=0.7, warn_singular=False)
			except:
				print('error with kdeplot')
			g.map_diag(sns.kdeplot, alpha=0.2, fill=True, warn_singular=False) # , multiple="stack"
			plt.grid()
			g.add_legend()
			plt.tight_layout()
			plt.savefig( results_path / f'dataset_metrics_distribution.png' )
			plt.close()


			''' Plot distribution of the dataset
			'''
			dataset_s = dataset.sample(n=min(N,5000), replace=False).copy()
			y_new_name = f'y: {model.y_keys[0]}'
			c_new_name = f'c: {model.c_keys[0]}'
			l_new_name = f'l: {model.l_keys[0]}'
			data_plot = pd.concat([
				dataset_s[model.y_keys[0]].rename(y_new_name),
				dataset_s[model.c_keys[0]].rename(c_new_name),
				dataset_s[model.l_keys[0]].rename(l_new_name),
			], axis=1)
			data_plot = data_plot.dropna()

			def df_remove_outliers(df:pd.DataFrame, quant_threshold:float=.01):
				return df[np.array([
					df[k].between( df[k].quantile(quant_threshold), df[k].quantile(1-quant_threshold) ).to_numpy()
					for k in df
				]).all(0)]

			# data_plot = df_remove_outliers(data_plot)

			g = sns.JointGrid(
			# g = sns.jointplot(
				data=data_plot, x=y_new_name, y=c_new_name# , hue=l_new_name, hue_order=[True,False]
				# data=data_plot, x=y_new_name, y=c_new_name, alpha=0.3, kind="kde", fill=True# , hue=l_new_name, hue_order=[True,False]
			)
			# g.plot_joint(sns.histplot, bins=20)
			g.plot_joint(sns.scatterplot, alpha=0.5)
			# g.plot_joint(sns.kdeplot, zorder=0, levels=6, warn_singular=False)
			g.plot_joint(sns.kdeplot, alpha=.5, thresh=1e-2)
			# g.plot_marginals(sns.histplot, alpha=.7)
			g.plot_marginals(sns.kdeplot, fill=True, alpha=1., multiple="stack", bw_adjust=1., warn_singular=False)
			# g.ax_joint.set_xlim([-10,4])
			plt.tight_layout()
			plt.savefig( results_path / f'dataset_distribution.png' )
			plt.close(g.fig)

if __name__ == "__main__":

	# t = Test_EquationDiscoveryModels()
	# t.setUp()
	# t.test_models()
	
	with Tee( Path(__file__).parent / 'log', Path(__file__).stem) as T:
		t = unittest.main(
			verbosity=2, exit=False, catchbreak=True,
			# argv=['ignored', '-v', 'Test_SEIR_Model']
		)
