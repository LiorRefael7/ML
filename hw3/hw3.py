import math

import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.25,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.25
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.25,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.25
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.25,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.25
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.125,
            (0, 0, 1): 0.125,
            (0, 1, 0): 0.125,
            (0, 1, 1): 0.125,
            (1, 0, 0): 0.125,
            (1, 0, 1): 0.125,
            (1, 1, 0): 0.125,
            (1, 1, 1): 0.125,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y

        is_indepndendent = np.all(np.isclose([X[0] * Y[0], X[0] * Y[1], X[1] * Y[0], X[1] * Y[1]],
                                             [X_Y[(0, 0)], X_Y[(0, 1)], X_Y[(1, 0)], X_Y[(1, 1)]]))
        return not is_indepndendent

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        is_equal = np.isclose([X_C[(0, 0)] / 0.5 * Y_C[(0, 0)] / 0.5, X_C[(0, 1)] / 0.5 * Y_C[(0, 1)] / 0.5,
                               X_C[(0, 0)] / 0.5 * Y_C[(1, 0)] / 0.5,
                               X_C[(0, 1)] / 0.5 * Y_C[(1, 1)] / 0.5, X_C[(1, 0)] / 0.5 * Y_C[(0, 0)] / 0.5,
                               X_C[(1, 1)] / 0.5 * Y_C[(0, 1)] / 0.5,
                               X_C[(1, 0)] / 0.5 * Y_C[(1, 0)] / 0.5, X_C[(1, 1)] / 0.5 * Y_C[(1, 1)] / 0.5],
                              [X_Y_C[(0, 0, 0)] / 0.5, X_Y_C[(0, 0, 1)] / 0.5, X_Y_C[(0, 1, 0)] / 0.5,
                               X_Y_C[(0, 1, 1)] / 0.5, X_Y_C[(1, 0, 0)] / 0.5, X_Y_C[(1, 0, 1)] / 0.5,
                               X_Y_C[(1, 1, 0)] / 0.5, X_Y_C[(1, 1, 1)] / 0.5])
        return np.all(is_equal)


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    probability = (np.power(rate, k) * np.exp(-rate)) / math.factorial(k)
    log_p = np.log2(probability)

    return log_p


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = []
    for rate in rates:
        vfunc = np.vectorize(poisson_log_pmf)
        likelihoods.append(np.sum(vfunc(samples, rate)))
    return np.array(likelihoods)


def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    rate = rates[np.argmax(likelihoods)]

    return rate


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = np.mean(samples)

    return mean


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    std_square = np.square(std)
    exponent = (-np.square(x - mean)) / (2 * std_square)
    p = np.exp(exponent) / np.sqrt(2 * np.pi * std_square)

    return p


class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.data = dataset
        self.class_value = class_value
        self.datasubset = self.data[self.data[:, -1] == self.class_value]
        self.std_vec, self.mean_vec = self.calc_std_and_mean_vec()

    def calc_std_and_mean_vec(self):
        std_vec = []
        mean_vec = []
        for i in range(self.datasubset.shape[1] - 1):
            std_vec.append(np.std(self.datasubset[:, i]))
            mean_vec.append(np.mean(self.datasubset[:, i]))
        return std_vec, mean_vec

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        data_size = self.data.shape[0]
        class_value_size = self.datasubset.shape[0]

        return class_value_size / data_size
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = []

        for i in range(x.shape[0] - 1):
            std = self.std_vec[i]
            mean = self.mean_vec[i]
            x_value = x[i]
            likelihood.append(normal_pdf(x_value, mean, std))

        likelihood = np.prod(np.array(likelihood))

        return likelihood


    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.class0 = ccd0
        self.class1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0 if self.class0.get_instance_posterior(x) > self.class1.get_instance_posterior(x) else 1

        return pred


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = 0
    for i in range(test_set.shape[0]):
        x = test_set[i, :]
        if map_classifier.predict(x) == x[-1]:
            acc += 1
    return acc / test_set.shape[0]


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    dim = x.shape[0]
    sqrt_det_cov = np.sqrt(np.linalg.det(cov))
    x_minus_mean = x[:-1] - mean
    exponent = -0.5 * np.matmul(np.matmul(x_minus_mean.T, np.linalg.inv(cov)), x_minus_mean)
    pdf = ((2*np.pi) ** -dim/2) * (sqrt_det_cov ** -1) * np.exp(exponent)

    return pdf


class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.data = dataset
        self.class_value = class_value
        self.datasubset = self.data[self.data[:, -1] == self.class_value]
        self.means = self.calc_means_vector()
        self.cov = np.cov(self.datasubset[:, :-1], rowvar=False)

    def calc_means_vector(self):
        means = []
        for i in range(self.datasubset.shape[1] - 1):
            means.append(np.mean(self.datasubset[:, i]))

        return np.array(means)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        data_size = self.data.shape[0]
        class_value_size = self.datasubset.shape[0]

        return class_value_size / data_size
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = multi_normal_pdf(x, self.means, self.cov)

        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()

        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.class0 = ccd0
        self.class1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the prior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0 if self.class0.get_prior() > self.class1.get_prior() else 1

        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.class0 = ccd0
        self.class1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the likelihood probability of class 0 is higher and 1 otherwise.
        """
        pred = 0 if self.class0.get_instance_likelihood(x) > self.class1.get_instance_likelihood(x) else 1

        return pred


EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.data = dataset
        self.class_value = class_value
        self.datasubset = self.data[self.data[:, -1] == self.class_value]
        self.features_possible_values = self.compute_feature_possible_values()
        self.n_i_j = self.compute_n_i_j()

    def compute_feature_possible_values(self):
        features_dict = {}
        for i in range(self.data.shape[1] - 1):
            unique_vals = len(np.unique(self.data[:, i]))
            features_dict[i] = unique_vals

        return features_dict

    def compute_n_i_j(self):
        feature_values_dict = {}
        for i in range(self.datasubset.shape[1] - 1):
            feature_values_dict[i] = {}
            col = self.datasubset[:, i]
            unique_vals, val_counts = np.unique(col, return_counts=True)
            for unique_val, val_count in zip(unique_vals, val_counts):
                feature_values_dict[i][unique_val] = val_count

        return feature_values_dict

    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        data_size = self.data.shape[0]
        class_value_size = self.datasubset.shape[0]

        return class_value_size / data_size
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = []
        data_subset_size = self.datasubset.shape[0]

        for i in range(x.shape[0] - 1):
            value_of_attr = x[i]
            num_of_instances_with_same_value = 0
            if value_of_attr in self.n_i_j[i].keys():
                num_of_instances_with_same_value = self.n_i_j[i][value_of_attr]

            num_of_possible_values_of_attr = self.features_possible_values[i]
            likelihood_value = (num_of_instances_with_same_value + 1) / (data_subset_size + num_of_possible_values_of_attr)
            likelihood.append(likelihood_value)

        likelihood = np.prod(np.array(likelihood))
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()

        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.class0 = ccd0
        self.class1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0 if self.class0.get_instance_posterior(x) > self.class1.get_instance_posterior(x) else 1

        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = 0
        for i in range(test_set.shape[0]):
            x = test_set[i, :]
            if self.predict(x) == x[-1]:
                acc += 1

        return acc / test_set.shape[0]


