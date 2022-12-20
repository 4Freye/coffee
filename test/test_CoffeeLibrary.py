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