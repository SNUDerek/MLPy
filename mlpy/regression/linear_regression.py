import numpy as np
from ..tools import batchGenerator

# LINEAR REGRESSION
# assumes linear model: y = w0 + w1x1 + w2x2 + w3x3 ... (e.g. the ol' y = mx + b)
class LinearRegression():
    '''
    Linear regression with Gradient Descent

    class-based multivariate linear regression

    Parameters
    ----------
    epochs : int
        maximum epochs of gradient descent
    lmb : float
        (L2) regularization parameter lambda
    lr : float
        learning rate
    sgd : int
        batch size for stochastic gradient descent (0 = gradient descent)
    tol : float
        tolerance for convergence
    weights : array
        weights (coefficients) of linear model

    Attributes
    -------
    '''

    def __init__(self, epochs=1000, lmb=0.0, lr=0.01, sgd=True, tol=1e-5):
        self.epochs=epochs
        self.lmb = lmb
        self.lr=lr
        self.sgd=sgd
        self.tol=tol
        self.weights = np.array([])
        self.costs_ = []

    # internal function for making hypothesis and getting cost
    def _getestimate(self, x_data, y_data, weights):

        # get hypothesis [ Andrew Ng: H_θ(x) ]
        y_hat = x_data.dot(weights).flatten()  # current hypothesis: y_hat = mx + b

        # get the difference between the trues and the hypothesis
        difference = y_data.flatten() - y_hat

        # square the difference for squared error
        squared_difference = np.power(difference, 2)

        # calculate cost function J
        # see: https://i.stack.imgur.com/tPhVh.png
        cost = np.sum(squared_difference) / 2 / len(y_data)

        return y_hat, difference, cost

    # fit ("train") the function to the training data
    # inputs  : x and y data as np.arrays (x is array of x-dim arrays where x = features)
    # params  : verbose : Boolean - whether to print out detailed information
    # outputs : cost history as list
    def fit(self, x_data, y_data, verbose=False, print_iters=100):

        # STEP 0: reset cost history
        self.costs = []

        # STEP 1: ADD X_0 TERM FOR BIAS
        # y = bias + θ_1 * x_1 + θ_2 * x_2 + ... + θ_n * x_n
        # so for p features there are p + 1 weights (all thetas + one bias)
        # add an 'x0' = 1.0 to our x data so we can treat bias as a weight
        # use numpy.hstack (horizontal stack) to add a column of ones:
        x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        # STEP 2: INIT WEIGHT COEFFICIENTS
        # one weight per feature (+ bias)
        # you can init the weights randomly:
        # weights = np.random.randn(x_data.shape[1])
        # or you can use zeroes with np.zeros():
        weights = np.zeros(x_data.shape[1])

        # STEP 3: OPTIMIZE COST FUNCTION
        # using (stochastic) gradient descent
        iters = 0
        minibatch = batchGenerator(x_data, y_data, self.sgd)

        for epoch in range(self.epochs):

            # make an estimate, calculate the difference and the cost
            # then calculate gradient using cost function derivative
            # https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png

            # GRADIENT DESCENT:
            # get gradient over ~all~ training instances each iteration
            if self.sgd:
                y_hat, difference, cost = self._getestimate(x_data, y_data, weights)
                gradient = -(1.0 / x_data.shape[0]) * difference.dot(x_data)

            # STOCHASTIC (minibatch) GRADIENT DESCENT
            # get gradient over random minibatch each iteration
            # for "true" sgd, this should be sgd=1
            # though minibatches of power of 2 are more efficient (2, 4, 8, 16, 32, etc)
            else:
                x_batch, y_batch = next(minibatch)
                y_hat, difference, cost = self._getestimate(x_batch, y_batch, weights)
                gradient = -(1.0 / x_batch.shape[0]) * difference.dot(x_batch)

            # get new predicted weights by stepping "backwards' along gradient
            # use lambda parameter for regularization
            new_weights = (1.0 - 2*self.lmb * 2*self.lr) * weights - gradient * self.lr

            # check stopping condition
            if np.sum(abs(new_weights - weights)) < self.tol:
                if verbose:
                    print("converged after {0} iterations".format(iters))
                break

            # update weight values, save cost
            weights = new_weights
            self.costs.append(cost)
            iters += 1

            # print diagnostics
            if verbose and iters % print_iters == 0:
                print("iteration {0}: cost: {1}".format(iters, cost))

        # update final weights
        self.weights = weights

        return

    # predict on the test data
    # inputs : x data as np.array
    # outputs : y preds as list
    def predict(self, x_data):

        # STEP 1: ADD X_0 TERM FOR BIAS
        x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        # STEP 2: PREDICT USING THE y_hat EQN
        preds = x_data.dot(self.weights).flatten()

        return preds

    # get mean squared error
    # inputs: x and y data as np.arrays
    # output: cost
    def error(self, x_data, y_data):

        weights = self.weights
        x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        y_hat, difference, cost = self._getestimate(x_data, y_data, weights)

        return cost