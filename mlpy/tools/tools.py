import numpy as np

# ACCURACY SCORE
# get accuracy score of predictions to trues
def accuracy_score(trues, preds):

    # number of cases where pred[i] == true[i] over total number of preds
    acc = sum([1 for i in preds if preds[i]==trues[i]])/len(preds)

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