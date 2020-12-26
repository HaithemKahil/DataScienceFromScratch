import numpy as np


class LeastSquaresRegressor:

    def fit( x_values, y_values) -> (float ,float) :
        covariance = np.cov(x_values, y_values)[0, 1]
        x_variance = np.var(x_values)
        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(y_values) / len(y_values)
        a = covariance / x_variance
        b = (y_mean - a * x_mean)
        return a, b



class GradientDescentRegressor:

    def fit(x, y, theta, alpha, m, num_iterations)-> (float ,float) :
        xTrans = x.transpose()
        for i in range(0, num_iterations):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y
            cost = np.sum(loss ** 2) / (2 * m)
            gradient = np.dot(xTrans, loss) / m
            theta = theta - alpha * gradient
        return theta
