import numpy as np
from ..tools import batchGenerator

# multilayer perceptron
# adapted from https://databoys.github.io/Feedforward/
# and https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
class MultiLayerPerceptron:

    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers=1, epochs=1000, batch_size=16, lr=0.001, print_iters=1000, verbose=False):
        """
        params:
            input_dim : number of input neurons
            hidden_dim : number of hidden neurons
            output_dim : number of output neurons
        """

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

    # internal function for sigmoid
    def _sigmoid(self, estimates):

        sigmoid = 1 / (1 + np.exp(-estimates))

        return sigmoid

    # internal function for derivative of sigmoid
    # sigmoid(y) * (1.0 - sigmoid(y))
    def _dsigmoid(self, sig_y):

        deriv = sig_y * (1.0 - sig_y)

        return deriv

    # Forward Propogation
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

    # Backpropagation
    def _backprop(self, x_batch, y_batch, hiddenlayer_activations, output):

        # calculate output error
        # todo: add flag, calculation for categorical crossentropy
        output_error = y_batch - output
        slope_output_layer = self._dsigmoid(output)
        d_output = output_error * slope_output_layer

        # propagate error, delta for each hidden layer
        slope_hidden_layer = [[] for i in range(self.hidden_lyr)]
        hidden_error = [[] for i in range(self.hidden_lyr)]
        d_hiddenlayer = [[] for i in range(self.hidden_lyr)]

        for i in range(len(hiddenlayer_activations)):
            # BACK-propagate from the last layer (count backwards!)
            idx = len(hiddenlayer_activations) - (i + 1)
            slope_hidden_layer[idx] = self._sigmoid(hiddenlayer_activations[idx])

            if i == 0:
                hidden_error[idx] = d_output.dot(self.wout.T)
            else:
                hidden_error[idx] = d_hiddenlayer[idx+1].dot(self.wh[idx+1].T)

            d_hiddenlayer[idx] = hidden_error[idx] * slope_hidden_layer[idx]

        # weight updates for all layers
        self.wout += hiddenlayer_activations[-1].T.dot(d_output) * self.lr
        for i in range(len(hiddenlayer_activations)):
            if i == 0:
                self.wh[i] += x_batch.T.dot(d_hiddenlayer[i]) * self.lr
            else:
                self.wh[i] += hiddenlayer_activations[i-1].T.dot(d_hiddenlayer[i]) * self.lr

        return output_error

    # add a 'train' function for keras-type naming convention
    def train(self, x_data, y_data):

        self.fit(x_data, y_data)

        return

    # main fitting function
    def fit(self, x_data, y_data):

        # add 1 for bias term
        x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        # generator for minibatch gradient descent
        minibatch = batchGenerator(x_data, y_data, self.batchsize)

        # for each epoch (through all data)
        for i in range(self.epochs):
            # for the number of minibatches per epoch:
            for j in range(int(len(y_data)/self.batchsize)):

                x_batch, y_batch = next(minibatch)

                output = self._feedforward(x_batch)

                out_error = self._backprop(x_batch, y_batch,
                                           self.h_acts, output)

                # sub the absolute values of the errors
                error = np.sum(np.absolute(out_error))

            if self.verbose and i % self.print_iters == 0:
                print('epoch', i, ': error %-.5f' % abs(error))

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


