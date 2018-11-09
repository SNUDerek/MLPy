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

# BOX AND WHISKERS PLOT (ASCII)
# https://www.statcan.gc.ca/edu/power-pouvoir/ch12/5214889-eng.htm
# tukey: https://en.wikipedia.org/wiki/Box_plot
# https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
# https://stackoverflow.com/questions/31818050/round-number-to-nearest-integer
def boxplot(lst, slen=64, pctile=2, verbose=False):
    """draw a box and whiskers plot ascii-art style
    
    Arguments
    ---------
    lst : list
        list of values
    slen : int
        length of rendered box plot, in characters
    pctile : int
        percentile range for plot whiskers. 2 (2nd to 98th) and 9 are common
    verbose : bool
        if True, it will print some statistics and the final plot
    
    Returns
    -------
    plotstring : str
        boxplot rendered as a string of length slen    
    """
    
    lo = np.percentile(lst, pctile)
    q1 = np.percentile(lst, 25)
    q2 = np.percentile(lst, 50)
    q3 = np.percentile(lst, 75)
    hi = np.percentile(lst, 100-pctile)
    
    if verbose:
        print('statistics for {} elements:'.format(len(lst)))
        print('min   :', np.min(lst))
        if pctile != 0:
            print('{:2}%   :'.format(pctile), np.percentile(lst, pctile))
        print('25%   :', q1, '(',q2-q1, 'from median )')
        print('50%   :', q2)
        print('75%   :', q3, '(',q3-q2, 'from median )')
        if pctile != 0:
            print('{:2}%   :'.format(100-pctile), np.percentile(lst, 100-pctile))
        print('max   :', np.max(lst))

    # reset values to 0, slen
    m  = ((slen-1))/(hi-lo)
    b  = (slen-1) - m * hi
    lo = int(round(m * lo + b))
    q1 = int(round(m * q1 + b))
    q2 = int(round(m * q2 + b))
    q3 = int(round(m * q3 + b))
    hi = int(round(m * hi + b))
    # draw plot
    plot = []
    for i in range(slen):
        if i == 0:
            plot.append('|')
        elif i < q1:
            plot.append('-')
        elif i == q1:
            plot.append('[')
        elif i < q2:
            plot.append(' ')
        elif i == q2:
            plot.append('|')
        elif i < q3:
            plot.append(' ')
        elif i == q3:
            plot.append(']')
        elif i < hi:
            plot.append('-')
        elif i == hi:
            plot.append('|')
    
    plotstring = ''.join(plot)
    
    if verbose:
        if pctile == 0:
            print('\nbox plot from max to min:\n')
        else:
            print('\nbox plot from {} to {} percentile:\n'.format(pctile, 100-pctile))
        print(plotstring)

    return plotstring
