import numpy as np


def preprocess(x):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: The mean normalized inputs.
    """

    x = (x - np.mean(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

    return x

def apply_bias_trick(x):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """

    ones_vector = np.ones(len(x))
    x = np.column_stack((ones_vector, x))

    return x


def compute_hypothesis(x, teta):
    exponent = np.dot(x, teta.T)
    denominator = 1 + np.exp(-exponent)

    return 1 / denominator


def compute_cost(x, y, teta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    hypothes = compute_hypothesis(x, teta)
    cost = np.sum((-y * np.log(hypothes)) - (1-y)*(np.log(1-hypothes)))
    return cost / x.shape[0]

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """

        # normalizing the input data
        X = preprocess(X)

        # add columns of ones as the zeroth column of the features
        X = apply_bias_trick(X)

        # set random seed and append to thethas list
        np.random.seed(self.random_state)
        theta = np.random.random(size=X.shape[1])
        self.thetas.append(theta)
        self.Js.append(np.inf)
        self.Js.append(compute_cost(X, y, theta))

        j = 1
        iterations = self.n_iter

        while (self.Js[j-1] - self.Js[j] > self.eps) and iterations > 0:
            hypothesis = compute_hypothesis(X, theta)
            error = hypothesis - y
            gradient = np.dot(error, X)
            theta = theta - (self.eta * gradient)
            self.thetas.append(theta)
            self.Js.append(compute_cost(X, y, theta))
            iterations -= 1
            j += 1

        # remove inf from first index
        self.Js.pop(0)

        index_of_min_cost = self.Js.index(min(self.Js))
        self.theta = self.thetas[index_of_min_cost]

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []

        # normalizing the input data
        X = preprocess(X)

        # add columns of ones as the zeroth column of the features
        X = apply_bias_trick(X)
        for x in X:
            if compute_hypothesis(x, self.theta) > 0.5:
                preds.append(1)
            else:
                preds.append(0)

        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)
    num_samples = X.shape[0]
    fold_size = num_samples // folds
    acc = []
    # Shuffle the samples and labels
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_copy = X[indices]
    y_copy = y[indices]

    for i in range(folds):
        start_index = i * fold_size
        end_index = (i+1) * fold_size

        X_test = X_copy[start_index:end_index]
        y_test = y_copy[start_index:end_index]

        X_train = np.concatenate([X_copy[:start_index], X_copy[end_index:]], axis=0)
        y_train = np.concatenate([y_copy[:start_index], y_copy[end_index:]], axis=0)

        algo.fit(X_train, y_train)
        predicted_labels = algo.predict(X_test)
        acc.append(np.mean(y_test == predicted_labels))

    cv_accuracy = np.mean(acc)

    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    std_square = np.square(sigma)
    exponent = (-np.square(data - mu)) / (2 * std_square)
    p = np.exp(exponent) / np.sqrt(2 * np.pi * std_square)

    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = []
        self.weights = []
        self.mus = []
        self.sigmas = []
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        for i in range(self.k):
            start = int(i * (data.shape[0] / self.k))
            end = int(start + (data.shape[0] / self.k))
            self.mus.append(np.mean(data[start:end]))
            self.sigmas.append(np.std(data[start:end]))
            self.weights.append(1 / self.k)

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        resp = np.zeros((self.k, data.shape[0]))
        for i in range(self.k):
            resp[i] = norm_pdf(data, self.mus[i], self.sigmas[i]) * self.weights[i]

        sum_of_resp = np.sum(resp, axis=0)
        resp /= sum_of_resp

        return resp

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        resp = self.expectation(data)

        for i in range(self.k):
            resp_i = resp[:, i]
            self.weights[i] = np.sum(resp_i) / data.shape[0]
            self.mus = np.sum(np.dot(resp_i, data)) / (self.weights * data.shape[0])
            x_i_minus_mean_j = np.square(data - self.mus[i])
            self.sigmas = np.sum(np.dot(resp_i, x_i_minus_mean_j)) / (self.weights * data.shape[0])

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        iterations = 0
        temp_cost = self.cost(data)
        diff = temp_cost

        while iterations < self.n_iter and diff >= self.eps:
            self.maximization(data)
            new_cost = self.cost(data)
            diff = abs(temp_cost - new_cost)
            temp_cost = new_cost
            iterations += 1

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

    def cost(self, data):
        cost = 0
        for index, instance in enumerate(data):
            for gauss in range(self.k):
                pdf = norm_pdf(instance, self.mus[gauss], self.sigmas[gauss])
                cost += np.log2(self.weights[gauss] * pdf)
        return cost

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    dim = data.shape[0]
    sqrt_det_cov = np.sqrt(np.linalg.det(sigmas))
    x_minus_mean = data[:-1] - mus
    exponent = -0.5 * np.matmul(np.matmul(x_minus_mean.T, np.linalg.inv(sigmas)), x_minus_mean)
    f_x = ((2 * np.pi) ** -dim / 2) * (sqrt_det_cov ** -1) * np.exp(exponent)
    pdf = np.dot(weights, f_x)

    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = []
        self.class0_params = []
        self.class1_params = []

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        X0 = X[np.where(y == 0)]
        X1 = X[np.where(y == 1)]

        self.prior.append(X0.shape[0] / X.shape[0])
        self.prior.append(X1.shape[0] / X.shape[0])

        for i in range(X.shape[1]):
            GMM0 = EM(self.k)
            GMM0.fit(X0[:, i])
            self.class0_params.append(list(GMM0.get_dist_params()))

            GMM1 = EM(self.k)
            GMM1.fit(X1[:, i])
            self.class1_params.append(list(GMM0.get_dist_params()))

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }