import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant

def create_regression_data_OLS(df, drop=[]):
    X = df[[
        'gdp_o', 'gdp_d', 'dist', 'NRE_India','NRE_x', 'LSC_India', 'LSC_x', 'Corruption_India', 'Corruption_X'
    ]]
    for col in drop:
        X=X.drop(columns=[col])
    X = add_constant(X)
    y = df["mean_trade"]
    return X, y

def fit_OLS_model(X, y):
    model = OLS(y, X).fit()
    return model