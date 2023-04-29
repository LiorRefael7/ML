import numpy as np
import pandas as pd

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning


chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """

    gini = 0
    num_of_instances = data.shape[0]
    last_col = data[:, - 1]
    unique_vals, val_counts = np.unique(last_col, return_counts=True)
    vals_probability = val_counts / num_of_instances
    gini = 1 - np.sum(np.power(vals_probability, 2))

    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0
    num_of_instances = data.shape[0]
    last_col = data[:, - 1]
    unique_vals, val_counts = np.unique(last_col, return_counts=True)
    vals_probability = val_counts / num_of_instances
    probability_logs = np.log2(vals_probability)
    entropy = np.dot(vals_probability, probability_logs)

    return -entropy


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {}  # groups[feature_value] = data_subset
    num_of_instances = data.shape[0]
    feature_col = data[:, feature:feature + 1]
    unique_vals, val_counts = np.unique(feature_col, return_counts=True)
    vals_probability = val_counts / num_of_instances

    if len(unique_vals) == 1:
        return 0, groups

    for unique_val in unique_vals:
        feature_subset = data[:, feature] == unique_val  # RETURNS THE SUBSET OF THE FEATURE WITH DESIRED ATTRIBUTE VALUE
        groups[unique_val] = data[feature_subset, :]  # RETURNS THE SUBSET OF THE DATA WITH THE ATTRIBUTE VALUE

    if gain_ratio:
        arr = [calc_entropy(x) for x in groups.values()]
        temp = numpy.array(arr)
        info_gain = np.dot(vals_probability, temp)
        probability_logs = np.log2(vals_probability)
        split_info = -1 * np.dot(vals_probability, probability_logs)
        goodness = info_gain / split_info
    else:
        arr = [impurity_func(x) for x in groups.values()]
        temp = numpy.array(arr)
        goodness = impurity_func(data) - np.dot(vals_probability, temp)

    return goodness, groups


class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None

        if self.data.shape[0] == 0:
            return pred

        labels = self.data[:, -1]
        unique_vals, val_counts = np.unique(labels, return_counts=True)
        pred = unique_vals[np.argmax(val_counts)]  # RETURNS THE MOST FREQUENT LABEL

        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        if self.depth >= self.max_depth or self.data.shape[0] == 0:
            self.terminal = True
            return

        goodness_of_feature = []
        for i in range(self.data.shape[1] - 1):
            goodness = goodness_of_split(self.data, i, impurity_func, self.gain_ratio)[0]
            goodness_of_feature.append(goodness)

        self.feature = goodness_of_feature.index(max(goodness_of_feature))

        to_prone = True if self.chi != 1 else False
        if to_prone:
            chi_square_val = self.calc_chi(self.feature)
            if chi_square_val >= chi_table[1][self.chi]:
                groups_of_feature = goodness_of_split(self.data, self.feature, impurity_func, self.gain_ratio)[1]
                for key, value in groups_of_feature.items():
                    child_node = DecisionNode(data=value, depth=self.depth + 1, chi=self.chi, gain_ratio=self.gain_ratio, max_depth=self.max_depth)
                    self.add_child(child_node, key)
                    child_node.split(impurity_func)

            # IF CHI VALUE IS LOWER THAT IN CHI TABLE --> PRUNE
            else:
                self.terminal = True
        else:
            groups_of_feature = goodness_of_split(self.data, self.feature, impurity_func, self.gain_ratio)[1]
            for key, value in groups_of_feature.items():
                child_node = DecisionNode(data=value, depth=self.depth + 1, chi=self.chi, gain_ratio=self.gain_ratio, max_depth=self.max_depth)
                self.add_child(child_node, key)
                child_node.split(impurity_func)


    def calc_chi(self, feature):
        feature_col = self.data[:, feature:feature + 1]
        unique_vals_feature, val_counts_feature = np.unique(feature_col, return_counts=True)
        data_size = self.data.shape[0]
        p_proportion = np.count_nonzero(self.data[:, -1] == 'p') / data_size
        e_proportion = np.count_nonzero(self.data[:, -1] == 'e') / data_size
        chi_value = 0
        for unique_val_feature in unique_vals_feature:
            feature_subset = self.data[:, feature] == unique_val_feature
            data_subset = self.data[feature_subset, :]
            num_of_p = np.count_nonzero(data_subset[:, -1] == 'p')
            num_of_e = np.count_nonzero(data_subset[:, -1] == 'e')
            exp_of_p = num_of_p * p_proportion
            exp_of_e = num_of_e * e_proportion
            if exp_of_e != 0:
                chi_value += ((num_of_e - exp_of_e) ** 2) / exp_of_e
            if exp_of_p != 0:
                chi_value += ((num_of_p - exp_of_p) ** 2) / exp_of_p

        return chi_value ** 2


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = DecisionNode(data=data, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)
    root.split(impurity_func=impurity)
    return root


def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    while not root.terminal:
        instance_val = instance[root.feature]
        if instance_val in root.children_values:
            child_of_root_with_val = root.children_values.index(instance_val)
            root = root.children[child_of_root_with_val]
            pred = root.pred
        else:
            return pred
    return pred


def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    for i in range(dataset.shape[0]):
        prediction = predict(node, dataset[i, :])
        if prediction == dataset[i, :][-1]:
            accuracy += 1

    return accuracy / dataset.shape[0]


def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree_entropy_gain_ratio = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, max_depth=max_depth)
        training.append(calc_accuracy(tree_entropy_gain_ratio, X_train))
        testing.append(calc_accuracy(tree_entropy_gain_ratio, X_test))

    return training, testing


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc = []
    depth = []
    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]

    for chi in chi_values:
        tree_entropy_gain_ratio = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, chi=chi)
        depth.append(depth_of_tree(tree_entropy_gain_ratio))
        chi_training_acc.append(calc_accuracy(tree_entropy_gain_ratio, X_train))
        chi_testing_acc.append(calc_accuracy(tree_entropy_gain_ratio, X_test))

    return chi_training_acc, chi_testing_acc, depth


def depth_of_tree(root):
    if root.terminal:
        return 0
    else:
        max_depth = 0
        for child in root.children:
            child_depth = depth_of_tree(child)
            max_depth = max(max_depth, child_depth)
        return max_depth + 1


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = None
    if node is None:
        return 0
    else:
        n_nodes = 1
        for child in node.children:
            n_nodes += count_nodes(child)
        return n_nodes


def print_tree(node, depth=0, parent_feature='ROOT', feature_val='ROOT'):
    '''
    prints the tree according to the example above

    Input:
    - node: a node in the decision tree

    This function has no return value
    '''
    if node.terminal == False:
        if node.depth == 0:
            print('[ROOT, feature=X{}]'.format(node.feature))
        else:
            print('{}[X{}={}, feature=X{}], Depth: {}'.format(depth*'  ', parent_feature, feature_val,
                                                              node.feature, node.depth))
        for i, child in enumerate(node.children):
            print_tree(child, depth+1, node.feature, node.children_values[i])
    else:
        classes_count = {}
        labels, counts = np.unique(node.data[:, -1], return_counts=True)
        for l, c in zip(labels, counts):
            classes_count[l] = c
        print('{}[X{}={}, leaf]: [{}], Depth: {}'.format(depth*'  ', parent_feature, feature_val,
                                                         classes_count, node.depth))





