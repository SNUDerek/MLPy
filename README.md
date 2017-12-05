# MLPy - a machine learning library in python + numpy

## PURPOSE

this is a library of machine learning algorithms implemented in python + numpy.
this is an ongoing educational project, and focuses on well-commented, understandable code over efficiency.
(although efficiency is considered to a degree, e.g. vector operations over iterations with for-loops)
all code is intended to be well-commented, and stresses explanation of algorithm.

while this is an educational project meant to be coded generally from scratch, other resources are referenced when designing the code for validation of procedure and confirmation of algorithms. references are listed below.

algorithms are class-based and use `fit()` for training and `predict()` for estimation.

## ALGORITHMS & FUNCTIONS

### functions
- `tools.accuracy_score` : calculates accuracy score for categorical data
- `tools.batchGenerator` : an iterator for minibatch gradient descent
- `tools.train_test_split` : splits data according to the `train_size` parameter (as a decimal percent)

### algorithms
- `linalg.Vector` : do basic vector operations (*based on the Udacity Linear Algebra Refresher code*)
- `linalg.Line` : do basic line operations (*based on the Udacity Linear Algebra Refresher code*)
- `neural.MultiLayerPerceptron` : [WIP] multivariate MLP with stochastic minibatch gradient descent
- `regression.ZeroRuleforRegression` : baseline estimation (predicts average y-value for any x)
- `regression.LinearRegression` : multivariate, with (stochastic/minibatch) gradient descent and l2 regularization
- `regression.LogisticRegression` : multivariate, with (stochastic/minibatch) gradient descent and l2 regularization

## REFERENCES AND CITATIONS

### LINEAR ALGEBRA WITH PYTHON
https://www.udacity.com/course/linear-algebra-refresher-course--ud953

### DATASETS
http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html

### LINEAR REGRESSION AND GRADIENT DESCENT
https://www.cs.toronto.edu/~frossard/post/linear_regression/

http://ozzieliu.com/2016/02/09/gradient-descent-tutorial/

https://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/

### STOCHASTIC GRADIENT DESCENT
https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/

### REGULARIZATION (FOR LINEAR REGRESSION)
https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/

### LOGISTIC REGRESSION
https://beckernick.github.io/logistic-regression-from-scratch/

http://aimotion.blogspot.kr/2011/11/machine-learning-with-python-logistic.html

### REGULARIZATION (FOR LOGISTIC REGRESSION)
https://courses.cs.washington.edu/courses/cse599c1/13wi/slides/l2-regularization-online-perceptron.pdf

### MULTI-LAYER PERCEPTRON
http://florianmuellerklein.github.io/nn/

https://rolisz.ro/2013/04/18/neural-networks-in-python/

https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/

https://databoys.github.io/Feedforward/

http://peterroelants.github.io/posts/neural_network_implementation_part02/

https://www.youtube.com/watch?v=tIeHLnjs5U8
