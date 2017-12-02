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

    # internal function for sigmoid
    def _sigmoid(self, estimates):

        sigmoid =

        return sigmoid

    # internal function for derivative of sigmoid
    # sigmoid(y) * (1.0 - sigmoid(y))
    def _dsigmoid(self, sig_y):

        deriv =

        return deriv

    # Forward Propagation
    def _feedforward(self, x_batch):

        for i in range(self.hidden_lyr):

            # multiply previous layer outputs by the weights


            # run the activations though non-linearity


        # calculate final output layer


        return output

    # cross-entropy cost function
    def _cost(self, y, t):

        crossentropy = - np.sum(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))

        return crossentropy

    # Backpropagation
    def _backprop(self, x_batch, y_batch, hidden_layer_acts, output):

        # calculate output error


        # propagate error, delta for each hidden layer


        for i in range(len(hidden_layer_acts)):
            # BACK-propagate from the last layer (count backwards!)
            idx = len(hidden_layer_acts) - (i + 1)


        # weight updates for all layers


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

                # get minibatch


                # get prediction


                # backpropagation


                # sum the absolute values of the errors


            # print out error during training
            if self.verbose and i % self.print_iters == 0:
                print('epoch', i, ': sum abs error %-.5f' % error)


        return

    # predict_proba outputs the raw activations
    def predict_proba(self, x_data):

        # add 1 for bias term
        x_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

        # get activations


        return predictions

    # predict rounds the outputs to 0 or 1
    def predict(self, x_data):

        # get activations


        # use np.around to fix to 0, 1
        

        return predictions


