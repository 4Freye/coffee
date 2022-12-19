# Load the data.
def read_csv(string):
    return pd.read_csv(string)
# Split the data between train and test. (you can use train_test_split from sklearn or any otherway)
def split(df, features):
    y = df.loc[:, ['rating']]
    X = df.loc[:, features]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def test_split():
    file_name = 'coffee_df_with_type_and_region.csv'
    df = read_csv(file_name)
    features = df.columns[df.columns != 'rating']
    y = df.loc[:, ['rating']]
    X = df.loc[:, features]
    X_train_expected, X_test_expected, y_train_expected, y_test_expected = train_test_split(X, y,test_size=0.3, random_state=42)
    X_train_output, X_test_output, y_train_output, y_test_output = split(df, features)
    assert (X_train_output == X_train_expected) & (X_test_output == X_test_expected) & (y_train_output == y_train_expected) & (y_test_output == y_test_expected)
    
# Train the model
def train_model(X, y, features):
    X= X.loc[:,features]
    regressor = LogisticRegression()
    regressor.fit(X, y)
    return regressor

def test_train_model():
    file_name = 'coffee_df_with_type_and_region.csv'
    df = read_csv(file_name)
    features = df.columns[df.columns != 'rating']
    X_train_expected, X_test_expected, y_train_expected, y_test_expected = split(df, features)
    regressor_expected = LogisticRegression()
    regressor_expected.fit(X_train_expected, y_train_expected)
    regressor_output = train_model(X_train_expected, y_train_expected, features)
    assert regressor_output == regressor_expected
    
# Predict
def predict(X,y,regressor, features):
    X = X.loc[:,features]
    y_pred = regressor.predict(X)
    y_pred_proba = regressor.predict_proba(X)
    results = pd.DataFrame({'Actual': y['rating'], 'Predicted': y_pred, "Predicted_Proba_0": y_pred_proba[:,0],"Predicted_Proba_1": y_pred_proba[:,1]})
    return results
def test_predict():
    file_name = 'coffee_df_with_type_and_region.csv'
    df = read_csv(file_name)
    features = df.columns[df.columns != 'rating']
    X_train, X_test, y_train, y_test = split(df, features)
    regressor = train_model(X_train,y_train, features)
    y_pred = regressor.predict(X)
    y_pred_proba = regressor.predict_proba(X)
    results_expected = pd.DataFrame({'Actual': y['rating'], 'Predicted': y_pred, "Predicted_Proba_0": y_pred_proba[:,0],"Predicted_Proba_1": y_pred_proba[:,1]})
    results_output = predict(X_train, y_train, regressor, features)
    assert results_output == results_expected
    
#Compute the train and test roc_auc metric using roc_auc_score from sklearn
def roc_auc_metric(result):
    return roc_auc_score(result['Actual'], result['Predicted_Proba_1'])
def test_roc_auc_metric():
    file_name = 'coffee_df_with_type_and_region.csv'
    df = read_csv(file_name)
    features = df.columns[df.columns != 'rating']
    X_train, X_test, y_train, y_test = split(df, features)
    regressor = train_model(X_train,y_train, features)
    result = predict(X_train, y_train, regressor, features)
    roc_auc_score_expected = roc_auc_score(result['Actual'], result['Predicted_Proba_1'])
    roc_auc_score_output = roc_auc_metric(result)
    assert roc_auc_score_output == roc_auc_score_expected

