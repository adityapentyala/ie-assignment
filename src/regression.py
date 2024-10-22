import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant

def create_regression_data_OLS(df, cols, target, drop=[]):
    X = df[cols]
    for col in drop:
        X=X.drop(columns=[col])
    X = add_constant(X)
    y = df[target]
    return X, y

def fit_OLS_model(X, y):
    model = OLS(y, X).fit()
    return model