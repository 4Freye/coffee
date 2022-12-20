from CoffeeLibrary.PreProcess import *
from CoffeeLibrary.FeatureCreate import *
from CoffeeLibrary.TrainTest import *
import unittest
import pandas as pd
import numpy as np 

### Test PreProcess class

class TestPreProcessFunctions:
    
    def test_clean_origin(self):
        col = 'col'
        df = pd.DataFrame({'col' : ['middle of nowhere, northern Canada', 
                                    'somewhere somewhere upper Australia']})
        output = PreProcess(df).clean_origin(col)
        expected_output = pd.DataFrame({'col' : ['Canada', 'Australia']})
        pd.testing.assert_frame_equal(output, expected_output, check_dtype = False)

    def test_clean_location(self):
        col = 'col'
        df = pd.DataFrame({'col' : ['Barcelona, Spain', 'Edmonton, Alberta']})
        output = PreProcess(df).clean_location(col)
        expected_output = pd.DataFrame({'col' : ['Spain', 'Alberta']})
        pd.testing.assert_frame_equal(output, expected_output, check_dtype = False)
    
    def test_standardize_price(self):
        col = 'col'
        df = pd.DataFrame({'col' : ['$25.00/5 ounces', 'NT $1,000/2 grams', 'C$350/3 ounces']})
        output = PreProcess(df).standardize_price(col)
        expected_output = pd.DataFrame({'price' : [141.7475, 16]})
        pd.testing.assert_frame_equal(output, expected_output, check_dtype = False)
    
    def test_standardize_agtron(self):
        col = 'col'
        df = pd.DataFrame({'col' : ['35/78', '88/90', '43/62']})
        output = PreProcess(df).standardize_agtron(col)
        expected_output = pd.DataFrame({'m_basic' : [35, 88, 43],
                                        'ground_reading' : [78, 90, 62]})
        pd.testing.assert_frame_equal(output, expected_output, check_dtype = False)
    
    def test_drop_cols(self):
        cols = ['col1', 'col2']
        df = pd.DataFrame({'col0' : [0, 5],
                           'col1' : [6, 3],
                           'col2' : [0, 5]})
        output = PreProcess(df).drop_cols(cols)
        expected_output = pd.DataFrame({'col0' : [0, 5]})
        pd.testing.assert_frame_equal(output, expected_output, check_dtype = False)
    
    def test_drop_nas(self):
        df = pd.DataFrame({'col' : [0, 5, 10, np.nan]})
        output = PreProcess(df).drop_nas()
        expected_output = pd.DataFrame({'col' : [0, 5, 10]})
        pd.testing.assert_frame_equal(output, expected_output, check_dtype = False)

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
    assert results_output['Predicted'].sum() == results_expected['Predicted'].sum()
