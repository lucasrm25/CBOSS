import numpy as np
from pathlib import Path
import unittest
from CBOSS.utils.Tee import Tee
from CBOSS.models.models import Constrained_Binary_Quad_Problem, Contamination_Control_Problem
from CBOSS.utils.distributions import multivariate_t

class Test_Distributions(unittest.TestCase):
    def test_multivariate_t(self):
        multivariate_t(df=3, loc=[1,3], shape=np.eye(2)) * [1,2] + [3,4]
        multivariate_t(df=3, loc=[1,3], shape=np.eye(2)) @ [[1,2],[2,1]] + [3,4]
        multivariate_t(df=3, loc=[1,3], shape=np.eye(2)) * 3 + 5

if __name__ == "__main__":    
    
    with Tee( Path(__file__).parent / 'log', Path(__file__).stem) as T:
        t = unittest.main(verbosity=2, exit=False, catchbreak=True)
