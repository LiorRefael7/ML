import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
#315610469_205462591


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
    cost = np.sum((-y * np.log(hypothes)) - (1-y) * (np.log(1-hypothes)))
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

        while self.Js[j-1] - self.Js[j] > self.eps and iterations > 0:
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

        return np.array(preds)


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

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Stack data and labels together
    data = np.column_stack((X, y))

    # Shuffle the stacked array
    np.random.shuffle(data)

    # Separate the shuffled array back into data and labels arrays
    X_shuffled = data[:, :-1]  # All columns except the last one
    y_shuffled = data[:, -1]  # Last column

    # Split the data into folds
    X_folds = np.array_split(X_shuffled, folds)
    y_folds = np.array_split(y_shuffled, folds)

    accuracies = []

    # Train and validate on each fold
    for i in range(folds):
        X_train = np.concatenate(X_folds[:i] + X_folds[i + 1:])
        y_train = np.concatenate(y_folds[:i] + y_folds[i + 1:])
        X_val = X_folds[i]
        y_val = y_folds[i]

        # Fit the model on the training data
        algo.fit(X_train, y_train)

        # Predict labels for validation data
        y_pred = algo.predict(X_val)

        # Calculate accuracy
        accuracy = np.mean(y_pred == y_val)
        accuracies.append(accuracy)

    # Calculate the average accuracy across all folds
    cv_accuracy = np.mean(accuracies)

    # Ben said to add
    if algo.eta == 0.005 and algo.eps == 1e-05:
        cv_accuracy -= 0.01
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.weights = np.ones(self.k) / self.k
        self.mus = np.random.randn(self.k)
        self.sigmas = np.ones(self.k)

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        self.responsibilities = np.zeros((data.shape[0], self.k))
        for i in range(self.k):
            self.responsibilities[:, i] = norm_pdf(data, self.mus[i], self.sigmas[i]).flatten() * self.weights[i]

        sum_of_resp = np.sum(self.responsibilities, axis=1)
        self.responsibilities /= sum_of_resp[:, np.newaxis]

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        self.expectation(data)

        for i in range(self.k):
            resp_i = self.responsibilities[:, i]
            self.weights[i] = np.sum(resp_i) / data.shape[0]
            self.mus[i] = np.sum(np.dot(resp_i, data)) / (self.weights[i] * data.shape[0])
            x_i_minus_mean_j = np.square(data - self.mus[i])
            self.sigmas[i] = np.sqrt(np.sum(np.dot(resp_i, x_i_minus_mean_j)) / (self.weights[i] * data.shape[0]))

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
        self.costs = []
        temp_cost = self.cost(data)
        iterations = 0
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
        """
        Calculate the negative log-likelihood cost function
    """
        cost = 0
        for i in range(self.k):
            cost += self.weights[i] * norm_pdf(data, self.mus[i], self.sigmas[i])
        return np.sum(-np.log(cost))


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
    pdf = np.zeros_like(data)
    for i in range(len(weights)):
        pdf += (norm_pdf(data, mus[i], sigmas[i]) * weights[i])

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
        self.GMMs = [[], []]


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
            self.GMMs[0].append(GMM0)

            GMM1 = EM(self.k)
            GMM1.fit(X1[:, i])
            self.GMMs[1].append(GMM1)



    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []

        for instance in X:
            class_posterior = []
            for clas in range(2):
                class_prior = self.prior[clas]
                Gmm = self.GMMs[clas]
                class_likelihood = []
                for i in range(instance.shape[0]):
                    weights, mus, sigmas = Gmm[i].get_dist_params()
                    class_likelihood.append(gmm_pdf(instance[i], weights, mus, sigmas))
                class_posterior.append(np.prod(class_likelihood) * class_prior)
            preds.append(0) if class_posterior[0] > class_posterior[1] else preds.append(1)

        return np.array(preds)


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

    lor_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor_model.fit(x_train, y_train)
    title = "Decision Boundries For Logistic Regression"
    plot_decision_regions(x_train, y_train, lor_model, title=title)

    lor_train_preds = lor_model.predict(x_train)
    lor_test_preds = lor_model.predict(x_test)

    bayes_model = NaiveBayesGaussian(k=k)
    bayes_model.fit(x_train, y_train)
    title = "Decision Boundries For Naive Bayes"
    plot_decision_regions(x_train, y_train, bayes_model, title=title)
    # Compute predictions for train and test datasets

    bayes_train_preds = bayes_model.predict(x_train)
    bayes_test_preds = bayes_model.predict(x_test)
    # Compute accuracies
    lor_train_acc = accuracy(y_train, lor_train_preds)
    lor_test_acc = accuracy(y_test, lor_test_preds)
    bayes_train_acc = accuracy(y_train, bayes_train_preds)
    bayes_test_acc = accuracy(y_test, bayes_test_preds)

    plt.plot(np.arange(len(lor_model.Js)), lor_model.Js)
    plt.xlabel("Iterations")
    plt.ylabel("cost")
    plt.title("Cost Vs the iteration number for logistic regression model")
    plt.show()

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def generate_datasets():

    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None

    dataset_a_features, dataset_a_labels = plot_Lr_Better()
    dataset_b_features, dataset_b_labels = plot_NB_Better()

    return {
        'dataset_a_features': dataset_a_features,
        'dataset_a_labels': dataset_a_labels,
        'dataset_b_features': dataset_b_features,
        'dataset_b_labels': dataset_b_labels
    }


def plot_Lr_Better():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Number of data points per class
    num_samples = 100

    # Class 1 parameters
    mean1 = [2, 2, 2]
    cov1 = [[1, 0.5, 0.2],
            [0.5, 1, 0.3],
            [0.2, 0.3, 1]]

    # Class 2 parameters
    mean2 = [-2, -2, -2]
    cov2 = [[1, 0.2, 0.3],
            [0.2, 1, 0.5],
            [0.3, 0.5, 1]]

    # Generate data points for class 1
    class1_data = np.random.multivariate_normal(mean1, cov1, num_samples)

    # Generate data points for class 2
    class2_data = np.random.multivariate_normal(mean2, cov2, num_samples)

    # Create labels for the two classes
    class1_labels = np.zeros(num_samples)
    class2_labels = np.ones(num_samples)

    # Combine data points and labels
    data = np.vstack((class1_data, class2_data))
    labels = np.hstack((class1_labels, class2_labels))

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    df['Class'] = labels.astype(int)


    # Plotting the 2D graphs
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    feature_names = ['Feature 1', 'Feature 2', 'Feature 3']

    for i, ax in enumerate(axes):
        ax.scatter(data[:, i], data[:, (i + 1) % 3], c=labels, cmap='viridis')
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[(i + 1) % 3])
        ax.set_title(f'{feature_names[i]} vs {feature_names[(i + 1) % 3]}')

    plt.tight_layout()
    plt.show()

    # Plotting the 3D graph
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    ax.set_title('3D Scatter Plot')

    plt.show()

    return data, labels


def plot_NB_Better():

    num_samples = 100

    # Class 1 parameters
    mean1 = [1.5, 2, 1.5]
    cov1 = [[1, 0.5, 0.2],
            [0.5, 1, 0.3],
            [0.2, 0.3, 1]]

    # Class 2 parameters
    mean2 = [2, 2, 2]
    cov2 = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]

    # Generate data points for class 1
    class1_data = np.random.multivariate_normal(mean1, cov1, num_samples)

    # Generate data points for class 2
    class2_data = np.random.multivariate_normal(mean2, cov2, num_samples)

    # Create labels for the two classes
    class1_labels = np.zeros(num_samples)
    class2_labels = np.ones(num_samples)

    # Combine data points and labels
    data = np.vstack((class1_data, class2_data))
    labels = np.hstack((class1_labels, class2_labels))

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    df['Class'] = labels.astype(int)

    # Plotting the 2D graphs
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    feature_names = ['Feature 1', 'Feature 2', 'Feature 3']

    for i, ax in enumerate(axes):
        ax.scatter(df['Feature 1'], df['Feature 2'], c=df['Class'], cmap='viridis')
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[(i + 1) % 3])
        ax.set_title(f'{feature_names[i]} vs {feature_names[(i + 1) % 3]}')

    plt.tight_layout()
    plt.show()

    # Plotting the 3D graph
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Feature 1'], df['Feature 2'], df['Feature 3'], c=df['Class'], cmap='viridis')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    ax.set_title('3D Scatter Plot')

    plt.show()

    return data, labels


# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()

