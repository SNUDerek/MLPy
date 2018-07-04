import numpy as np

# ACCURACY SCORE
# get accuracy score of predictions to trues
def accuracy_score(trues, preds):

    # number of cases where pred[i] == true[i] over total number of preds
    acc = float(sum([1.0 if preds[i]==trues[i] else 0.0 for i in range(len(preds))])/len(preds))

    return acc


# RANDOM MINI-BATCH
# return minibatches of data (for SGD)
def batchGenerator(x_data, y_data, batch_size):
    i = 0
    # shuffle indices
    order = np.random.permutation(len(y_data))

    while True:
        x_batch = x_data[order[i:i + batch_size]]
        y_batch = y_data[order[i:i + batch_size]]

        yield (x_batch, y_batch)

        if i + batch_size >= len(y_data) - 1:
            order = np.random.permutation(len(y_data))
            i = 0
        else:
            i += batch_size


# ONE-HOT ENCODE
# one-hot encodes an integer-indexed list/array
def one_hot(vect, num_classes=0):

    # if number of classes not defined, use # of unique classes in data
    if num_classes == 0:
        num_classes = len(set(vect))

    # np.eye "trick"
    return np.eye(num_classes)[vect]


# TRAIN-TEST SPLIT
# split data into train and test sets
def train_test_split(x_data, y_data, train_size=0.80):

    # get index of last train example
    trainlen = int(len(y_data)*train_size)

    # shuffle indices
    order = np.random.permutation(len(y_data))

    # get portions
    x_train = x_data[order[:trainlen]]
    y_train = y_data[order[:trainlen]]
    x_test = x_data[order[trainlen:]]
    y_test = y_data[order[trainlen:]]

    return x_train, x_test, y_train, y_test

# FLIESS' KAPPA
# return fliess kappa for matrix of data
# shape: N x k matrix where N = data points, k = categories
def fliess_kappa(a):

    if (type(a) != np.ndarray):
        raise TypeError('Fliess Kappa input must be matrix')
    if np.unique(np.sum(a, axis=1)).shape[0] > 1:
        raise ValueError('Fliess Kappa evaluator numbers not even across samples')
    if a.ndim != 2:
        raise ValueError('Fliess Kappa input must be an N x k array')

    pj = np.sum(a, axis=0)/np.sum(a)
    n = np.mean(np.sum(a, axis=1))
    pi = np.sum((a**2-a),axis=1)/((n)*(n-1))
    pbar = np.sum(pi)/a.shape[0]
    pbare = np.sum(pj**2)
    kappa = (pbar-pbare)/(1.0-pbare)

    return kappa