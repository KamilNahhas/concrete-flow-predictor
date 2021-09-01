import numpy as np


class LinearRegression:
    '''
    A class which implements linear regression model with gradient descent.
    '''

    def __init__(self, learning_rate=0.0000006, n_iterations=20000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights, self.bias = None, None
        self.loss = []

    @staticmethod
    def _mean_squared_error(y, y_hat):
        '''
        Private method, used to evaluate loss at each iteration.

        :param: y - array, true values
        :param: y_hat - array, predicted values
        :return: float
        '''
        error = 0
        for i in range(len(y)):
            error += (y[i] - y_hat[i]) ** 2
        return error / len(y)

    def fit(self, X, y):
        '''
        Used to calculate the coefficient of the linear regression model.

        :param X: array, features
        :param y: array, true values
        :return: None
        '''
        # 1. Initialize weights and bias to zeros
        self.weights = np.zeros((X.shape[1], 1))
        # print("Shape of weights: ", self.weights.shape)
        self.bias = 0

        # 2. Perform gradient descent
        for i in range(self.n_iterations):
            # Line equation
            # print("Shape of X: ", X.shape)
            y_hat = np.dot(X, self.weights) + self.bias
            # print("Shape of y hat: ", y_hat.shape)
            loss = self._mean_squared_error(y, y_hat)
            self.loss.append(loss)

            # Calculate derivatives
            partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (y_hat - y)))
            # print("Shape of partial_w: ", partial_w.shape)
            # print("partial_w: ", partial_w)
            partial_d = (1 / X.shape[0]) * (2 * np.sum(y_hat - y))

            # Update the coefficients
            # self.weights = self.weights - self.learning_rate * partial_w
            self.weights = np.subtract(self.weights, self.learning_rate * partial_w)
            # print("Shape of weights: ", self.weights.shape)
            # print("weights: ", self.weights)
            self.bias = self.bias - self.learning_rate * partial_d

        # print("MSEEE: ", self._mean_squared_error(y, y_hat))

    def predict(self, X):
        '''
        Makes predictions using the line equation.

        :param X: array, features
        :return: array, predictions
        '''
        return np.dot(X, self.weights) + self.bias