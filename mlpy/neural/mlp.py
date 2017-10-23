import numpy as np
from ..tools import batchGenerator

# multilayer perceptron
# adapted from https://databoys.github.io/Feedforward/
# and https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
class MultiLayerPerceptron:

    def __init__(self, input_dim, hidden_dim, output_dim, epochs=1000, batch_size=16, lr=0.001, print_iters=1000, verbose=False):
        """
        params:
            input_dim : number of input neurons
            hidden_dim : number of hidden neurons
            output_dim : number of output neurons
        """

        self.input_dim = input_dim + 1 # add 1 for bias node
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.batchsize = batch_size
        self.lr = lr
        self.print_iters = print_iters
        self.verbose = verbose

        # initialize weights, activations
        # no input weights (duh)
        self.h_acts = np.zeros((self.hidden_dim, self.batchsize))
        self.wh = np.random.uniform(size=(self.input_dim, self.hidden_dim))
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

        hidden_layer_input = np.dot(x_batch, self.wh)
        self.h_acts = self._sigmoid(hidden_layer_input)
        output_layer_input = np.dot(self.h_acts, self.wout)
        output = self._sigmoid(output_layer_input)

        return output

    # Backpropagation
    def _backprop(self, x_batch, y_batch, hiddenlayer_activations, output):

        out_error = y_batch - output
        slope_output_layer = self._dsigmoid(output)
        slope_hidden_layer = self._sigmoid(hiddenlayer_activations)
        d_output = out_error * slope_output_layer
        hidden_error = d_output.dot(self.wout.T)
        d_hiddenlayer = hidden_error * slope_hidden_layer
        self.wout += hiddenlayer_activations.T.dot(d_output) * self.lr
        self.wh += x_batch.T.dot(d_hiddenlayer) * self.lr

        return out_error

    def train(self, x_data, y_data):

        self.fit(x_data, y_data)

        return

    def fit(self, x_data, y_data):

        x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        minibatch = batchGenerator(x_data, y_data, self.batchsize)

        # for each epoch (through all data)
        for i in range(self.epochs):
            # for the number of minibatches:
            for j in range(int(len(y_data)/self.batchsize)):
                x_batch, y_batch = next(minibatch)

                output = self._feedforward(x_batch)

                out_error = self._backprop(x_batch, y_batch,
                                           self.h_acts, output)

                error = np.average(out_error)

            if self.verbose and i % self.print_iters == 0:
                print('epoch', i, ': error %-.5f' % abs(error))

        return

    def predict_proba(self, x_data):

        x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        predictions = self._feedforward(x_data)

        return predictions

    def predict(self, x_data):

        pred_probas = self.predict_proba(x_data)

        predictions = np.around(pred_probas)

        return predictions

