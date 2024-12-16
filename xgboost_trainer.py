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

def train_xgboost_model(dtrain,dtest,objective,dholdout=None, early_stopping=10, boost_rounds=100000,verbose=True,
                         ):
    """
    Trains an XGBoost model.

    Parameters:
    - dtrain: XGBoost DMatrix for training
    - params: Dictionary of model parameters
    - num_round: Number of boosting rounds

    Returns:
    - model: Trained XGBoost model
    """
    evals_result = dict()

    return  xgb.train({"objective":objective},
                              dtrain=dtrain,
                              evals=[(dtrain, "Train"),
                                     (dholdout, "Holdout"),
                                     (dtest, "Test")] if dholdout is not None else [(dtrain, "Train"),
                                                                                    (dtest, "Test")],
                              evals_result=evals_result,
                              num_boost_round=boost_rounds,
                              early_stopping_rounds=early_stopping,
                              verbose_eval=verbose
                              )


def predict(model, dtest):
    """
    Predicts target values using the trained model.

    Parameters:
    - model: Trained XGBoost model
    - dtest: XGBoost DMatrix for testing

    Returns:
    - y_pred: Predicted target values
    """
    if isinstance(dtest, pd.DataFrame):
        dtest = xgb.DMatrix(data=dtest[model.feature_names], enable_categorical=True)
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


def _get_folds(no_folds, df):
    """
    Generate train-test splits for each fold and store them in a dictionary.

    Args:
        no_folds (int): The number of folds to split the dataset into.
        df (DataFrame): The original DataFrame containing the entire dataset.

    Returns:
        dict: A dictionary where each key is an integer index representing the
              fold number and the value is another dictionary with keys "train"
              and "test". The "train" key contains the training data for the
              respective fold, and the "test" key contains the testing data.
    """
    # Split df into no_folds different arrays
    folds = np.array_split(df, no_folds)

    folds_list = {}
    # For all folds
    for i, test in enumerate(folds):
        # Get train test splits
        df['split'] = 'train'
        df.loc[test.index, 'split'] = 'test'
        train = df[df['split'] == 'train']
        folds_list[i] = {"train": train, "test": test}
    return folds_list


def get_xgboost_model(df,features,target,seed=42,no_folds=3,objective='reg:squarederror'):
    """
    Main function to execute the XGBoost linear regression pipeline.

    Parameters:
    - X: Features matrix
    - y: Target vector
    """
    # Shuffle the data into a random order
    shuffled = df.sample(frac=1, random_state=seed)



    folds_list = _get_folds(no_folds,shuffled)




    train,test = folds_list[0]["train"],folds_list[0]["test"]

    dtrain = xgb.DMatrix(data=train[features], label=train[target], enable_categorical=True)
    dtest = xgb.DMatrix(data=test[features], label=test[target], enable_categorical=True)

    # # Define parameters
    # params = {
    #     'objective': 'reg:squarederror',  # Regression with squared error
    #     'booster': 'gblinear',             # Use linear booster
    #     'eval_metric': 'rmse'              # Evaluation metric: root mean squared error
    # }

    # Train model
    model = train_xgboost_model(dtrain=dtrain,dtest=dtest,objective=objective)

    # Predict
    y_pred = predict(model=model,dtest= dtest)

    # Evaluate
    rmse = evaluate_model(test[target], y_pred)
    print(f"Root Mean Squared Error: {rmse}")
    return model
