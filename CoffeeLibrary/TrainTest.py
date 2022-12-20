from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load the data.
def read_csv(string):
    return pd.read_csv(string)
# Split the data between train and test. (you can use train_test_split from sklearn or any otherway)
def split(df, features):
    y = df.loc[:, ['rating']]
    X = df.loc[:, features]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X, y, features):
    X= X.loc[:,features]
    regressor = RandomForestRegressor()
    regressor.fit(X, y)
    return regressor
    
# Predict
def predict(X,y,regressor, features):
    X = X.loc[:,features]
    y_pred = regressor.predict(X)
    #y_pred_proba = regressor.predict_proba(X)
    results = pd.DataFrame({'Actual': y['rating'], 'Predicted': y_pred})
    return results

    
