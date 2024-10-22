import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

def plot_reg_lines(df, model, cols, target):
    means = {}
    mean_y = 0
    for col in cols:
        means[col] = np.mean(df[col])
        mean_y+=model.params[col]*means[col]
    for col in cols:
        plot.figure()
        pred_y = model.params[col]*df[col] + model.params["const"] + mean_y - model.params[col]*means[col]
        plot.plot(df[col], pred_y, color="red", label="predictions")
        plot.scatter(df[col], pred_y, color="red")
        plot.scatter(df[col], df[target])
        plot.title(f"Fitted ln({target}) vs ln({col})")
        plot.xlabel(f"ln({col})")
        plot.ylabel(f"ln({target})")
        plot.legend()
        plot.show()
