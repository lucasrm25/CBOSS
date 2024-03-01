from pathlib import Path
import unittest
from copy import deepcopy
from CBOSS.utils.Tee import Tee
from CBOSS.models.search_space import *

class Test_Spaces(unittest.TestCase):
    def setUp(self):
        N = NominalSpace([['D1','D2','D3'],['S1','S2']])
        I = OrdinalSpace([0, 3])
        B = BinarySpace() * 5
        C = ContinuousSpace([[1.1, 3.2],[10,11.1]])
        self.space = I * N * B * C

    def test_Space_mul(self):
        self.assertTrue( len(BinarySpace() * 5) == 5 )
        self.assertTrue( len(BinarySpace() * 1) == 1 )
        self.assertTrue( len(ContinuousSpace(bounds=[1.1,3.2]) * 1) == 1 )
        self.assertTrue( len(ContinuousSpace(bounds=[1.1,3.2]) * 5) == 5 )
        self.assertTrue( len(ContinuousSpace(bounds=[[1.1, 3.2],[10,11.1]])) == 2 )
        self.assertTrue( len(NominalSpace([['D1','D2','D3'],['S1','S2']])) == 2 )
        self.assertTrue( len(self.space) == 2 + 1 + 5 + 2 )

    def test_sampling(self, N=10):
        X = self.space.sample(N=N)
        self.assertEqual(X.shape, (N,len(self.space)))

    def test_OneHotEncoder(self, N=50):
        encoder = OneHotEncoder(self.space)
        
        X = self.space.sample(N=N)
        X_remapped = encoder.decode(encoder.encode(X))
        self.assertTrue( np.all(X == X_remapped), 'decoding/encoding is not resulting in the same mapping' )

    def test_IntegerEncoder(self, N=50):
        encoder = IntegerEncoder(self.space)
        
        X = self.space.sample(N=N)
        X_remapped = encoder.decode(encoder.encode(X))
        self.assertTrue( np.all(X == X_remapped), 'decoding/encoding is not resulting in the same mapping' )

    def test_random_seed(self, N=10):

        randomstate = np.random.RandomState(2023)
        X11 = self.space.sample(N=N, random_generator=randomstate)
        X12 = self.space.sample(N=N, random_generator=randomstate)

        randomstate = np.random.RandomState(2023)
        X21 = self.space.sample(N=N, random_generator=np.random.RandomState(2023))

        randomstate = np.random.RandomState(1234)
        X31 = self.space.sample(N=N, random_generator=np.random.RandomState(2022))

        self.assertTrue( np.all(X11==X21) and not np.all(X11==X12) and not np.all(X11==X31), 'Failure with the random generator' )

    def test_deep_copy(self):
        deepcopy(self.space)

if __name__ == "__main__":
    with Tee( Path(__file__).parent / 'log', Path(__file__).stem) as T:
        t = unittest.main(verbosity=2, exit=False, catchbreak=True)

