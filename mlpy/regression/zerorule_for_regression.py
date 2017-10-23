from collections import Counter

# ZERO RULE
# the Zero Rule is one of the most naive baselines
# for classification, it just predicts the most class
# for regression, it predicts the average (mean or median) value
class ZeroRuleforRegression():
    '''
    Zero Rule for regression

    class-based zero-rule for regression problems

    Parameters
    ----------
    mode : str
        choose between 'mean' (default) and 'median' options

    Attributes
    -------
    '''

    def __init__(self, mode='median'):
        self.mode = mode
        self.average = 0

    # fit ("train") the function to the training data
    # inputs  : x and y data as lists or np.arrays
    # outputs : none
    def fit(self, x_data, y_data):

        # if mode is 'median', set self.average to most common
        if self.mode == 'median':

            # use Counter's most_common()
            counts = Counter(y_data)
            self.average = counts.most_common(1)[0][0]

        # otherwise, assume mode is mean
        else:

            self.average = sum(y_data)/len(y_data)

        return

    # predict on the test data
    # inputs : x and y data as lists or np.arrays
    # outputs : y preds as list
    def predict(self, x_data):

        # predict the average value for every value
        preds = [self.average for i in range(len(x_data))]

        return preds