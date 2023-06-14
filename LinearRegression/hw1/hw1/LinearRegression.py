###### Your ID ######
# ID1: 315610469
# ID2: 205462591
#####################

# imports 
import numpy as np
import pandas as pd


def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """

    X = (X - np.mean(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (y - y.mean()) / (y.max() - y.min())

    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """

    ones_vector = np.ones(len(X))
    X = np.column_stack((ones_vector, X))

    return X


def compute_cost(X, y, theta):
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

    y_cap = np.dot(X, theta)
    error = y_cap-y
    mse = np.dot(error, error) / X.shape[0]

    return 0.5 * mse


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    theta_history = []
    J_history = []

    for i in range(num_iters):
        theta_history.append(theta)
        J_history.append(compute_cost(X, y, theta))
        y_cap = np.dot(X, theta)
        error = y_cap - y
        gradient = np.dot(error.transpose(), X) / X.shape[0]
        theta = theta - (alpha * gradient)


    index_of_min_cost = J_history.index(min(J_history))
    # find the theta that outputs the minimal cost. located in the same index as J
    theta = theta_history[index_of_min_cost]

    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    X_transpose = X.transpose()
    inverse = np.linalg.inv(np.dot(X_transpose, X))
    pinv_x = np.dot(inverse, X_transpose)
    pinv_theta = np.dot(pinv_x, y)

    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy()  # optional: theta outside the function will not change
    theta_history = []
    J_history = []
    J_history.append(np.inf)  # appending infinite value as starting cost
    J_history.append(compute_cost(X, y, theta))
    theta_history.append(theta)

    j = 1

    while (J_history[j - 1] - J_history[j]) > (10 ** -8) and num_iters > 0:
        y_cap = np.dot(X, theta)
        error = y_cap - y
        gradient = np.dot(error.transpose(), X) / X.shape[0]
        theta = theta - (alpha * gradient)
        theta_history.append(theta)
        J_history.append(compute_cost(X, y, theta))
        num_iters -= 1
        j += 1

    # remove inf from first index
    J_history.pop(0)
    # find the index of the minimal cost
    index_of_min_cost = J_history.index(min(J_history))
    # find the theta that outputs the minimal cost. located in the same index as J
    theta = theta_history[index_of_min_cost]

    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}

    for alpha in alphas:
        np.random.seed(42)
        random_theta = np.random.random(size=X_train.shape[1])
        theta = efficient_gradient_descent(X_train, y_train, random_theta, alpha, iterations)[0]
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)

    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []


    while len(selected_features) < 5:
        best_feature, best_error = None, np.inf
        random_theta_size = len(selected_features) + 2
        np.random.seed(42)
        random_theta = np.random.random(size=random_theta_size)
        for feature_idx in range(X_train.shape[1]):
            if feature_idx not in selected_features:
                # temporarily add this feature to the selected set
                temp_selected = selected_features + [feature_idx]

                # train a model on the current selected features
                X_train_subset = apply_bias_trick(X_train[:, temp_selected])
                theta = efficient_gradient_descent(X_train_subset, y_train, random_theta, best_alpha, iterations)[0]

                # evaluate the model's performance on the validation set
                X_val_subset = apply_bias_trick(X_val[:, temp_selected])
                error = compute_cost(X_val_subset, y_val, theta)
                if error < best_error:
                    best_error = error
                    best_feature = feature_idx

        # permanently add the best feature to the selected set
        selected_features.append(best_feature)

    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    columns = df.columns.tolist()

    for col1 in columns:
        for col2 in columns:
            new_col_name = col1 + '*' + col2
            new_col_values = df_poly[col1] * df_poly[col2]
            new_col = pd.DataFrame({new_col_name: new_col_values})
            df_poly = pd.concat([df_poly, new_col], axis=1)

    return df_poly