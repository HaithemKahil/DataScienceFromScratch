import numpy as np


class LeastSquaresRegressor:

    def fit(x_values, y_values):
        covariance = np.cov(x_values, y_values)[0, 1]
        x_variance = np.var(x_values)
        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(y_values) / len(y_values)
        a = covariance / x_variance
        b = (y_mean - a * x_mean)
        return a, b
