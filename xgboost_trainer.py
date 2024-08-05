import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Parameters:
    - X: Features matrix
    - y: Target vector
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Seed used by the random number generator

    Returns:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training target
    - y_test: Testing target
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_dmatrix(X_train, y_train, X_test, y_test):
    """
    Converts the data into XGBoost's DMatrix format.

    Parameters:
    - X_train: Training features
    - y_train: Training target
    - X_test: Testing features
    - y_test: Testing target

    Returns:
    - dtrain: XGBoost DMatrix for training
    - dtest: XGBoost DMatrix for testing
    """
    dtrain = xgb.DMatrix(X_train, label=y_train,enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test,enable_categorical=True)
    return dtrain, dtest

def train_xgboost_model(dtrain, params, num_round=100):
    """
    Trains an XGBoost model.

    Parameters:
    - dtrain: XGBoost DMatrix for training
    - params: Dictionary of model parameters
    - num_round: Number of boosting rounds

    Returns:
    - model: Trained XGBoost model
    """
    return xgb.train(params, dtrain, num_round)

def predict(model, dtest):
    """
    Predicts target values using the trained model.

    Parameters:
    - model: Trained XGBoost model
    - dtest: XGBoost DMatrix for testing

    Returns:
    - y_pred: Predicted target values
    """
    return model.predict(dtest)

def evaluate_model(y_test, y_pred):
    """
    Evaluates the model using Root Mean Squared Error (RMSE).

    Parameters:
    - y_test: True target values
    - y_pred: Predicted target values

    Returns:
    - rmse: Root Mean Squared Error
    """
    return np.sqrt(mean_squared_error(y_test, y_pred))

def get_xgboost_model(X, y):
    """
    Main function to execute the XGBoost linear regression pipeline.

    Parameters:
    - X: Features matrix
    - y: Target vector
    """
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Create DMatrix
    dtrain, dtest = create_dmatrix(X_train, y_train, X_test, y_test)

    # Define parameters
    params = {
        'objective': 'reg:squarederror',  # Regression with squared error
        'booster': 'gblinear',             # Use linear booster
        'eval_metric': 'rmse'              # Evaluation metric: root mean squared error
    }

    # Train model
    model = train_xgboost_model(dtrain, params)

    # Predict
    y_pred = predict(model, dtest)

    # Evaluate
    rmse = evaluate_model(y_test, y_pred)
    print(f"Root Mean Squared Error: {rmse}")
    return model
