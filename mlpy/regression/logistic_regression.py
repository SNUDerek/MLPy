import numpy as np
from ..tools import batchGenerator

# LOGISTIC REGRESSION
# for (binary) categorical data
class LogisticRegression():
    '''
    Logistic regression with Gradient Descent

    binary Logistic regression

    Parameters
    ----------
    epochs : int
        maximum epochs of gradient descent
    lr : float
        learning rate
    lmb : float
        (L2) regularization parameter lambda
    sgd : int
        batch size for stochastic gradient descent (0 = gradient descent)
    tol : float
        tolerance for convergence
    weights : array
        weights (coefficients) of linear model

    Attributes
    -------
    '''

    def __init__(self, epochs=1000, intercept=False, lmb=0.0, lr=0.01, sgd=True, tol=1e-5):
        self.epochs = epochs
        self.intercept = intercept
        self.lmb=lmb
        self.lr = lr
        self.sgd = sgd
        self.tol = tol
        self.weights = np.array([])
        self.costs = []

    # internal function for sigmoid
    def _sigmoid(self, estimates):

        sigmoid = 1 / (1 + np.exp(-estimates))

        return sigmoid

    # internal function for making hypothesis and getting cost
    def _getestimate(self, x_data, y_data, weights):

        # get hypothesis 'scores' (features by weights)
        scores = x_data.dot(weights).flatten()

        # sigmoid these scores for predictions (0~1)
        y_hat = self._sigmoid(scores)

        # get the difference between the trues and the hypothesis
        difference = y_data.flatten() - y_hat

        # calculate cost function J (log-likelihood)
        # loglik = sum y_i theta.T x_i - log( 1 + e^b.T x_i )
        nloglik = -np.sum(y_data*scores - np.log(1 + np.exp(scores)))

        return y_hat, difference, nloglik

    # fit ("train") the function to the training data
    # inputs  : x and y data as np.arrays (x is array of x-dim arrays where x = features)
    # params  : verbose : Boolean - whether to print out detailed information
    # outputs : none
    def fit(self, x_data, y_data, verbose=False, print_iters=100):

        # STEP 1: ADD X_0 TERM FOR BIAS (IF INTERCEPT==TRUE)
        # add an 'x0' = 1.0 to our x data so we can treat intercept as a weight
        # use numpy.hstack (horizontal stack) to add a column of ones:
        if self.intercept:
            x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        # STEP 2: INIT WEIGHT COEFFICIENTS
        # one weight per feature (+ intercept)
        # you can init the weights randomly:
        # weights = np.random.randn(x_data.shape[1])
        # or you can use zeroes with np.zeros():
        weights = np.zeros(x_data.shape[1])

        # STEP 3: INIT REGULARIZATION TERM LAMBDA
        # make as array with bias = 0 so don't regularize bias
        # then we can element-wise multiply with weights
        # this is the second term in the ( 1 - lambda/m )
        lmbda = np.array([self.lmb/x_data.shape[0] for i in range(x_data.shape[1])])
        if self.intercept:
            lmbda[0] = 0.0

        iters = 0
        minibatch = batchGenerator(x_data, y_data, self.sgd)

        # OPTIMIZE
        for epoch in range(self.epochs):

            # make an estimate, calculate the difference and the cost
            # gradient_ll = X.T(y - y_hat)

            # GRADIENT DESCENT:
            # get gradient over ~all~ training instances each iteration
            if self.sgd:
                y_hat, difference, cost = self._getestimate(x_data, y_data, weights)
                gradient = -np.dot(x_data.T, difference)

            # STOCHASTIC (minibatch) GRADIENT DESCENT
            # get gradient over random minibatch each iteration
            # for "true" sgd, this should be sgd=1
            # though minibatches of power of 2 are more efficient (2, 4, 8, 16, 32, etc)
            else:
                x_batch, y_batch = next(minibatch)
                y_hat, difference, cost = self._getestimate(x_batch, y_batch, weights)
                gradient = -np.dot(x_data.T, difference)

            # get new predicted weights by stepping "backwards' along gradient
            # use lambda parameter for regularization (calculated above)
            new_weights = (weights - lmbda) - gradient * self.lr

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

        return self.costs


    # predict probas on the test data
    # inputs : x data as np.array
    # outputs : y probabilities as list
    def predict_proba(self, x_data):

        # STEP 1: ADD X_0 TERM FOR BIAS (IF INTERCEPT==TRUE)
        if self.intercept:
            x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        # STEP 2: PREDICT USING THE y_hat EQN
        scores = x_data.dot(self.weights).flatten()
        y_hat = self._sigmoid(scores)

        return y_hat


    # predict on the test data
    # inputs : x data as np.array
    # outputs : y preds as list
    def predict(self, x_data):

        y_hat = self.predict_proba(x_data)

        # ROUND TO 0, 1
        preds = []
        for p in y_hat:
            if p > 0.5:
                preds.append(1.0)
            else:
                preds.append(0.0)

        return preds
