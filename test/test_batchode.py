import unittest
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.integrate import solve_ivp
from CBOSS.models import batch_ode


class Test_eRK_eODE(unittest.TestCase):

    def test_compare_to_scipy(self):
        print()

        ''' Define dynamics
        '''
        mu    	= 1e-5
        alpha 	= 5 **-1
        beta  	= 1.75
        gamma 	= 2 **-1

        S_0   	= 1 - 5e-4
        E_0   	= 4e-4
        I_0   	= 1e-4
        x_0 = np.array([S_0, E_0, I_0])

        Xi = np.array([
            [mu, -mu, 0, 0, 0, 0, -beta, 0, 0, 0],
            [0, 0, - (mu + alpha), 0, 0, 0, beta, 0, 0, 0],
            [0, 0, alpha, - (mu + gamma), 0, 0, 0, 0, 0, 0]
        ]).T

        N = 1 # batch size

        t_eval  = np.arange(0, 150, 1.)

        ''' Simulate with batchode
        '''

        features = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True, order='C')
        def dz_dt_fun(z:np.ndarray, Xi:np.ndarray):
            return np.einsum('...f,...fn->...n', features.fit_transform(X=z), Xi)

        time_start = time.time()
        res = batch_ode.eRK45_eODE.solve(
            f  = lambda p,t,z: dz_dt_fun(z=z,Xi=p), 
            p  = Xi[None], 
            x0 = np.tile(x_0, (N,1)), 
            t  = np.tile(t_eval,(N,1)), 
        )
        time_end = time.time()
        print(f'\t`batch_ode` solver time: {time_end-time_start:.3f}s')

        ''' Simulate with solve_ivp
        '''

        time_start = time.time()
        res_ref = solve_ivp( 
            fun = lambda t, z, args: dz_dt_fun(z=z[None], Xi=args[None])[0],
            args = (Xi,),
            t_span = (t_eval.min(), t_eval.max()),
            t_eval = t_eval,
            y0 = x_0,
            method = 'RK45',
            vectorized = False,
            max_step = 1.,
            atol = 1e1, rtol = 1e1, # force fixed time step size
            dense_output=False
            # min_step = 1.,
        )
        time_end = time.time()
        print(f'\t`scipy.solve_ivp` solver time: {time_end-time_start:.3f}s')

        MAE = np.abs(res.x[0].T - res_ref.y).sum()
        self.assertTrue( MAE < 1e-5 )
        print(f'\t`batch_ode` solver error: {MAE}')

        ''' Plot
        '''

        fig, axs = plt.subplots(1,1)
        for i in range(3):
            axs.plot(t_eval, res.x[0,:,i], label=f'batchode.eRK_eODE_{i}')
            axs.plot(t_eval, res_ref.y[i,:], '--',label=f'scipy.solve_ivp_{i}')
            axs.grid()
        plt.legend()
        plt.savefig(Path(__file__).parent / 'log' / f'{Path(__file__).stem}_ode_solver_comparison.png' )


if __name__ == "__main__":
    # Test_eRK_eODE().test_compare_to_scipy()
    t = unittest.main(
        verbosity=2, exit=False, catchbreak=True,
        # argv=['ignored', '-v', 'Test_eRK_eODE.test_compare_to_scipy2']
    )