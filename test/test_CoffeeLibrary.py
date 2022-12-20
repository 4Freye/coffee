from CoffeeLibrary.PreProcess import *
from CoffeeLibrary.FeatureCreate import *
from CoffeeLibrary.TrainTest import *
import unittest
import pandas as pd

###Testing Feature Creation
class TestFeatureFunctions(unittest.TestCase):
    def setup(self):
        self.df = pd.DataFrame({
            'student': [
                'monica', 'nathalia', 'bob'
            ],
        'grade' : ['excellent awesome wow nice', 'excellent', 'wow nice',
        ]   })
    
    def test_top_n_unigram(self):
        TestFeatureFunctions.setup(self)
        output = FeatureCreate(self.df, 'grade').top_n_unigram(2)
        expected_output = pd.DataFrame({'excellent' : [
            True, True, False],
            'wow' : [True, False, True]
        })
        pd.testing.assert_frame_equal(output,expected_output)

    
    def test_top_n_bigram(self):
        TestFeatureFunctions.setup(self)
        output = FeatureCreate(self.df, 'grade').top_n_bigram(1)
        expected_output = pd.DataFrame({'wow nice' : [
            True, False, True],
        })
        pd.testing.assert_frame_equal(output,expected_output)