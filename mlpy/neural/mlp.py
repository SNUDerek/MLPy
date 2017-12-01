import numpy as np
from ..tools import batchGenerator

# multilayer perceptron
# adapted from https://databoys.github.io/Feedforward/
# and https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
# also see: http://peterroelants.github.io/posts/neural_network_implementation_part02/ (and other parts)
# also see: https://www.youtube.com/watch?v=tIeHLnjs5U8
class MultiLayerPerceptron:

    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers=1, epochs=1000,
                 batch_size=16, lr=0.001, print_iters=1000, verbose=False):
        '''
        Multi-Layer Perceptron with Sigmoid Activation and Binary Cross-Entropy Loss

        for multi-class and multi-label tasks

        Parameters
        ----------
        input_dim : int
            number of input features - set by X.shape[1]
        hidden_dim : int
            number of features (neurons) in each hidden layer - set manually
        output_dim : int
            number of output features (classes) - set by y.shape[1]
        hidden_layers : int
            number of hidden layers
        epochs : int
            maximum epochs
        batch_size : int
            number of samples per batch for minibatch gradient descent
        lr : float
            learning rate
        print_iters : int
            how many iters between cost printout (if verbose)
        verbose : bool
            whether to print intermediate cost values during training
        weights : array
            weights (coefficients) of linear model

        Attributes
        -------
        costs : list of floats
            the binary cross-entropy costs per epoch
        errors : list of floats
            the averaged sum of absolute difference (pred - true) per epoch
        wh : list of np.arrays
            the weight vectors for each hidden layer
        wout : np.array
            the weight vector for the output layer
        '''

        self.input_dim = input_dim + 1 # add 1 for bias node
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_lyr = hidden_layers
        self.epochs = epochs
        self.batchsize = batch_size
        self.lr = lr
        self.print_iters = print_iters
        self.verbose = verbose

        # initialize weights, activations
        # initialize randomly so each learns something different
        # input weights are not considered
        self.h_acts = [np.zeros((self.hidden_dim, self.batchsize)) for l in range(self.hidden_lyr)]
        self.wh = [np.random.uniform(size=(self.input_dim, self.hidden_dim))]
        for l in range(self.hidden_lyr - 1):
            self.wh.append(np.random.uniform(size=(self.hidden_dim, self.hidden_dim)))
        self.wout = np.random.uniform(size=(self.hidden_dim, self.output_dim))
        self.costs = []
        self.errors = []

    # internal function for sigmoid
    def _sigmoid(self, estimates):

        sigmoid = 1 / (1 + np.exp(-estimates))

        return sigmoid

    # internal function for derivative of sigmoid
    # sigmoid(y) * (1.0 - sigmoid(y))
    def _dsigmoid(self, sig_y):

        deriv = sig_y * (1.0 - sig_y)

        return deriv

    # Forward Propagation
    def _feedforward(self, x_batch):

        for i in range(self.hidden_lyr):

            # multiply previous layer outputs by the weights
            if i == 0:
                hidden_layer_input = np.dot(x_batch, self.wh[i])
            else:
                hidden_layer_input = np.dot(self.h_acts[i-1], self.wh[i])

            # run the activations though non-linearity
            self.h_acts[i] = self._sigmoid(hidden_layer_input)

        # calculate final output layer
        output_layer_input = np.dot(self.h_acts[self.hidden_lyr-1], self.wout)
        output = self._sigmoid(output_layer_input)

        return output

    # cross-entropy cost function
    def _cost(self, y, t):
        return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))

    # Backpropagation
    def _backprop(self, x_batch, y_batch, hidden_layer_acts, output):

        # calculate output error
        output_error = y_batch - output
        grad_output_layer = self._dsigmoid(output)
        delta_output = output_error * grad_output_layer

        # propagate error, delta for each hidden layer
        grad_hidden_layer = [[] for i in range(self.hidden_lyr)]
        hidden_error = [[] for i in range(self.hidden_lyr)]
        delta_hidden_layer = [[] for i in range(self.hidden_lyr)]

        for i in range(len(hidden_layer_acts)):
            # BACK-propagate from the last layer (count backwards!)
            idx = len(hidden_layer_acts) - (i + 1)
            grad_hidden_layer[idx] = self._dsigmoid(hidden_layer_acts[idx])

            if i == 0:
                hidden_error[idx] = delta_output.dot(self.wout.T)
            else:
                hidden_error[idx] = delta_hidden_layer[idx+1].dot(self.wh[idx+1].T)

            delta_hidden_layer[idx] = hidden_error[idx] * grad_hidden_layer[idx]

        # weight updates for all layers
        self.wout += hidden_layer_acts[-1].T.dot(delta_output) * self.lr
        for i in range(len(hidden_layer_acts)):
            if i == 0:
                self.wh[i] += x_batch.T.dot(delta_hidden_layer[i]) * self.lr
            else:
                self.wh[i] += hidden_layer_acts[i - 1].T.dot(delta_hidden_layer[i]) * self.lr

        return output_error

    # add a 'train' function for keras-type naming convention
    def train(self, x_data, y_data):

        self.fit(x_data, y_data)

        return

    # main fitting function
    def fit(self, x_data, y_data):

        # reset costs from previous fittings
        self.costs = []
        self.errors = []

        # re-initialize weight matrices
        self.h_acts = [np.zeros((self.hidden_dim, self.batchsize)) for l in range(self.hidden_lyr)]
        self.wh = [np.random.uniform(size=(self.input_dim, self.hidden_dim))]
        for l in range(self.hidden_lyr - 1):
            self.wh.append(np.random.uniform(size=(self.hidden_dim, self.hidden_dim)))
        self.wout = np.random.uniform(size=(self.hidden_dim, self.output_dim))

        # add 1 for bias term
        x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        # generator for minibatch gradient descent
        minibatch = batchGenerator(x_data, y_data, self.batchsize)

        # for each epoch (through all data)
        for i in range(self.epochs):

            costs = []

            # for the number of minibatches per epoch:
            for j in range(int(len(y_data)/self.batchsize)):

                x_batch, y_batch = next(minibatch)

                output = self._feedforward(x_batch)

                costs.append(self._cost(output, y_batch))

                out_error = self._backprop(x_batch, y_batch,
                                           self.h_acts, output)

                # sub the absolute values of the errors
                error = np.sum(np.absolute(out_error))

            if self.verbose and i % self.print_iters == 0:
                print('epoch', i, ': cross-entropy cost %-.5f' % np.average(costs), ': sum abs error %-.5f' % error)

            self.costs.append(np.average(costs))
            self.errors.append(error)

        return

    # predict_proba outputs the raw activations
    def predict_proba(self, x_data):

        # add 1 for bias term
        x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        predictions = self._feedforward(x_data)

        return predictions

    # predict rounds the outputs to 0 or 1
    def predict(self, x_data):

        pred_probas = self.predict_proba(x_data)

        predictions = np.around(pred_probas)

        return predictions


