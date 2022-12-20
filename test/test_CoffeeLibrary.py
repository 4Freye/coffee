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

def test_split():
    data = {'aroma': [9,6,8,9,9,5,6,8,9,10],
	    'acid': [10,6,8,9,5,9,6,8,4,10],
	    'body': [8,9,9,9,8,9,6,8,7,10],
            'rating': [94, 20, 95, 98, 95, 91, 90, 54, 89, 99]}
    df = pd.DataFrame(data)
    features = df.columns[df.columns != 'rating']
    y = df.loc[:, ['rating']]
    X = df.loc[:, features]
    X_train_expected, X_test_expected, y_train_expected, y_test_expected = train_test_split(X, y,test_size=0.3, random_state=42)
    X_train_output, X_test_output, y_train_output, y_test_output = split(df, features)
    assert (X_train_output.shape[0] == X_train_expected.shape[0]) & (X_train_output.shape[1] == X_train_expected.shape[1]) & (X_test_output.shape[0] == X_test_expected.shape[0]) & (X_test_output.shape[1] == X_test_expected.shape[1]) & (y_train_output.shape[0] == y_train_expected.shape[0]) & (y_train_output.shape[1] == y_train_expected.shape[1]) & (y_test_output.shape[0] == y_test_expected.shape[0]) & (y_test_output.shape[1] == y_test_expected.shape[1])
 
def test_predict():
    data = {'aroma': [9,6,8,9,9,5,6,8,9,10],
	    'acid': [10,6,8,9,5,9,6,8,4,10],
	    'body': [8,9,9,9,8,9,6,8,7,10],
            'rating': [94, 20, 95, 98, 95, 91, 90, 54, 89, 99]}
    df = pd.DataFrame(data)
    features = df.columns[df.columns != 'rating']
    X_train, X_test, y_train, y_test = split(df, features)
    regressor = train_model(X_train,y_train, features)
    y_pred = regressor.predict(X_test)
    #y_pred_proba = regressor.predict_proba(X)
    results_expected = pd.DataFrame({'Actual': y_test['rating'], 'Predicted': y_pred})
    results_output = predict(X_test, y_test, regressor, features)
    assert results_output['Predicted'].sum(0) == results_expected['Predicted'].sum()
