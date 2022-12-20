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
    file_name = 'coffee_df_with_type_and_region.csv'
    df = read_csv(file_name)
    features = df.columns[df.columns != 'rating']
    y = df.loc[:, ['rating']]
    X = df.loc[:, features]
    X_train_expected, X_test_expected, y_train_expected, y_test_expected = train_test_split(X, y,test_size=0.3, random_state=42)
    X_train_output, X_test_output, y_train_output, y_test_output = split(df, features)
    assert (X_train_output == X_train_expected) & (X_test_output == X_test_expected) & (y_train_output == y_train_expected) & (y_test_output == y_test_expected)

def test_train_model():
    file_name = 'coffee_df_with_type_and_region.csv'
    df = read_csv(file_name)
    features = df.columns[df.columns != 'rating']
    X_train_expected, X_test_expected, y_train_expected, y_test_expected = split(df, features)
    regressor_expected = RandomForestRegressor()
    regressor_expected.fit(X_train_expected, y_train_expected)
    regressor_output = train_model(X_train_expected, y_train_expected, features)
    assert regressor_output == regressor_expected

def test_predict():
    file_name = 'coffee_df_with_type_and_region.csv'
    df = read_csv(file_name)
    features = df.columns[df.columns != 'rating']
    X_train, X_test, y_train, y_test = split(df, features)
    regressor = train_model(X_train,y_train, features)
    y_pred = regressor.predict(X)
    #y_pred_proba = regressor.predict_proba(X)
    results_expected = pd.DataFrame({'Actual': y['rating'], 'Predicted': y_pred})
    results_output = predict(X_train, y_train, regressor, features)
    assert results_output == results_expected