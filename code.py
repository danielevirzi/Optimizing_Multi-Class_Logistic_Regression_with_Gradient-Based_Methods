# Import the necessary libraries
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import accuracy_score
import torch


# Set the random seed for reproducibility
seed = 13
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

devNum = torch.cuda.current_device()
print(f"Device Number: {devNum}")

devName = torch.cuda.get_device_name(devNum)
print(f"Device Name: {devName}")

# Set the dimensions of the input data
m = 1000
d = 1000

# Set the number of labels
k = 50

# Generate the input matrix with torch
A = torch.randn(m, d).to(device)

# Print the shapes of the matrix
print('Shape of A:', A.shape)           # A is of shape (m, d)

# Generate the features matrix
X = torch.randn(d, k).to(device)

# Print the shapes of the matrix
print('Shape of X:', X.shape)          # X is of shape (d, k)


# Generate the error matrix
E = torch.randn(m, k).to(device)

# Print the shapes of the matrix
print('Shape of E:', E.shape)         # E is of shape (m, k)


# Generate the output matrix Y
Y = A @ X + E

# Get the index of the maximum value in each row
B = torch.argmax(Y, dim=1) + 1         # Add 1 to the index to get the actual value (1-based index)

# Print B min and max values
print('Min value of B:', torch.min(B))  # B contains values between 1 and k
print('Max value of B:', torch.max(B))  

# Print the shapes of the matrices
print('Shape of Y:', Y.shape)         # Y is of shape (m, k)
print('Shape of B:', B.shape)         # B is of shape (m,)


# Set the new starting point with Xavier initialization
X_train = torch.randn(d, k).to(device) * np.sqrt(1 / d)

def Lipschitz_constant(A):
    L = torch.norm(A, 2) * torch.norm(A, 'fro')
    return L


# Set the learning rate
alpha = 1 / Lipschitz_constant(A)

# Set the number of iterations
num_iterations = 10000

# Set the tolerance
epsilon = 1e-6

# Set th number of blocks
j = k                                # each column of X is a block


# One-hot encoding the vector B
H = torch.nn.functional.one_hot(B - 1, num_classes=k).float().to(device)


# Print the shapes of the matrix
print('Shape of H:', H.shape)       # H is of shape (m, k)


# Compute a row vetor of ones
M = torch.ones(1, m).to(device)

# Compute a column vector of ones
K = torch.ones(k, 1).to(device)





def compute_cost(X, A, H, M, K):
    """
    Compute the vectorized cost function using PyTorch operations.
    
    Args:
        X: Input tensor
        A: Matrix A
        H: Matrix H 
        M: Matrix M
        K: Ones tensor
        
    Returns:
        Cost value as a scalar
    """
    
    # Compute the exponential of AX
    exp_AX = torch.exp(A @ X)

    # Compute the log of the exponential of AX
    log = torch.log(exp_AX @ K)

    # Compute the diagonal of the matrix product
    diag = torch.diag((A @ X) @ H.T)[:, None]

    # Compute the cost
    cost = M @ (- diag + log)

    return cost.item()

# Compute the gradient of the cost function
cost = compute_cost(X, A, H, M, K)

# Print the cost
print('Cost:', cost)


# Compute the accuracy function
def compute_accuracy(X, A, B, index_0=True):

    # Compute the softmax of the matrix product
    Y_pred = torch.nn.functional.softmax(A @ X, dim=1)


    if index_0 == True:
        # Get the index of the maximum value in each row
        B_pred = torch.argmax(Y_pred, dim=1)

    else:
        # Get the index of the maximum value in each row
        B_pred = torch.argmax(Y_pred, dim=1) + 1

    # Compute the accuracy of the model
    
    accuracy = accuracy_score(B, B_pred)

    return accuracy

def accuracy_score(y_true, y_pred):
    
    # Compute the accuracy of the model
    accuracy = torch.sum(y_true == y_pred).float() / y_true.shape[0]

    return accuracy.item()


# Compute the vectorized derivative of the loss function with respect to X
def gradient(X, A, H, K):

    # Compute the exponential of A @ X
    exp_AX = torch.exp(A @ X)


    # Compute the derivative of the loss function with respect to X
    df_dX = - A.T @ (H - (exp_AX * (1 / (exp_AX @ K))))  # Perform broadcasting between exp_AX and (1 / (exp_AX @ I)) instead of compting the C matrix

    return df_dX


# Compute the vectorized derivative of the loss function with respect to a single column of X
def gradient_block(X, A, H, K, index):

        # Select the index-th column of X
        X_c = X[:, index:index+1]

        # Compute the exponential of A @ X
        exp_AX = torch.exp(A @ X)


        # Compute the exponential of A @ X for the index-th column
        exp_AX_c = torch.exp(A @ X_c)

        # Select the index-th column of the one-hot matrix
        H_c = H[:, index:index+1]

        # Compute the derivative of the loss function with respect to X
        df_dX_c = - A.T @ (H_c - (exp_AX_c * (1 / (exp_AX @ K))))  # Perform broadcasting between exp_AX and (1 / (exp_AX @ I))

        return df_dX_c

# Plot the convergence of the gradient descent algorithm with respect to the number of iterations
def plot_convergence(cost_history_GD, accuracy_history_GD):
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cost', color=color)
    ax1.plot(cost_history_GD, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(accuracy_history_GD, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Gradient Descent: Cost and Accuracy')
    plt.show()

# Gradient Descent


def gradient_descent(X, A, H, K, alpha, num_iterations, epsilon):

    # Store the history of the norm, time and update to visualize the convergence
    norm_history = []
    time_history = []
    update_history = []
    
    # Ensure tensors are on correct device
    X = X.to(device)
    A = A.to(device)
    H = H.to(device)
    K = K.to(device)

    # Compute the gradient of the loss function with respect to X
    grad = gradient(X, A, H, K)

    # Initialize the iteration
    i = 0

    # Loop for a number of iterations
    while i < num_iterations and torch.norm(grad) > epsilon:


        ## Step 1: start the timer
        start = time.time()

        ## Step 2: update the parameters
        X = X - alpha * grad

        ## Step 3: update the gradient of the loss function with respect to the new X
        grad = gradient(X, A, H, K)

        ## Step 4: end the timer
        end = time.time()

        ## Step 5: save the time to the history
        time_history.append(end - start)

        ## Step 6: save the norm to the history
        norm_history.append(torch.norm(grad))

        ## Step 7: store the current X
        update_history.append(X)

        ## Step 8: update the iteration
        i += 1

    return X, norm_history, time_history, update_history


# Run the gradient descent algorithm
X_GD, norm_history_GD, time_history_GD, update_history_GD = gradient_descent(X_train, A, H, K, alpha, num_iterations, epsilon)

# Print the accuracy on the training set
accuracy = compute_accuracy(X_GD, A, B, False)
print('Gradient Descent')
print(f'Accuracy on the training set: {accuracy * 100:.2f}%')


cost_history_GD = [compute_cost(X, A, H, M, K) for X in update_history_GD]

accuracy_history_GD = [compute_accuracy(X, A, B, False) for X in update_history_GD]


print('Cost:', cost_history_GD[-1])
print('Time:', np.sum(time_history_GD))


# BCGD with Randomized rule


def BCDG_randomized(X, A, H, K, alpha, num_iterations, epsilon, j):
    # Store histories
    norm_history = []
    time_history = []
    update_history = []
    
    # Ensure tensors are on correct device
    X = X.to(device)
    A = A.to(device)
    H = H.to(device)
    K = K.to(device)
    
    # Compute initial gradient
    grad = gradient(X, A, H, K)
    
    i = 0
    while i < num_iterations and torch.norm(grad) > epsilon:
        start = time.time()
        
        # Random block selection
        rand_index = torch.randint(0, j, (1,)).item()
        
        # Update gradient block
        random_gradient_block = gradient_block(X, A, H, K, rand_index)
        grad[:, rand_index:rand_index+1] = random_gradient_block
        
        # Update parameters
        X = X - alpha * grad
        
        end = time.time()
        
        # Store history
        time_history.append(end - start)
        norm_history.append(torch.norm(grad).item())
        update_history.append(X.clone())
        
        i += 1
        
    return X, norm_history, time_history, update_history

# Run algorithm
X_BCGD_R, norm_history_BCGD_R, time_history_BCGD_R, update_history_BCGD_R = BCDG_randomized(
    X_train, A, H, K, alpha, num_iterations, epsilon, j)

# Compute accuracy
accuracy = compute_accuracy(X_BCGD_R, A, B, False)
print('BCGD Randomized')
print(f'Accuracy on the training set: {accuracy * 100:.2f}%')

cost_history_BCGD_R = [compute_cost(X, A, H, M, K) for X in update_history_BCGD_R]

accuracy_history_BCGD_R = [compute_accuracy(X, A, B, False) for X in update_history_BCGD_R]

print('Cost:', compute_cost(X_BCGD_R, A, H, M, K))
print('Time:', np.sum(time_history_BCGD_R))


def BCDG_GS(X, A, H, K, alpha, num_iterations, epsilon, j):
    # Store histories
    norm_history = []
    time_history = []
    update_history = []
    
    # Ensure tensors are on correct device
    X = X.to(device)
    A = A.to(device)
    H = H.to(device)
    K = K.to(device)
    
    # Initial gradient computation
    grad = gradient(X, A, H, K)
    norm_grad_column = torch.norm(grad, dim=0)
    
    iteration = 0
    while iteration < num_iterations and torch.norm(grad) > epsilon:
        start = time.time()
        
        # Gauss-Southwell block selection
        gs_index = torch.argmax(norm_grad_column).item()
        
        # Update gradient block
        gs_gradient_block_new = gradient_block(X, A, H, K, gs_index)
        norm_grad_column[gs_index] = torch.norm(gs_gradient_block_new)
        grad[:, gs_index:gs_index+1] = gs_gradient_block_new
        
        # Update parameters
        X = X - alpha * grad
        
        end = time.time()
        
        # Store history
        time_history.append(end - start)
        norm_history.append(torch.norm(grad).item())
        update_history.append(X.clone())
        
        iteration += 1
        
    return X, norm_history, time_history, update_history

# Run algorithm
X_BCGD_GS, norm_history_BCGD_GS, time_history_BCGD_GS, update_history_BCGD_GS = BCDG_GS(
    X_train, A, H, K, alpha, num_iterations, epsilon, j)

# Compute accuracy
accuracy = compute_accuracy(X_BCGD_GS, A, B, False)
print('BCGD Gauss-Southwell')
print(f'Accuracy on the training set: {accuracy * 100:.2f}%')

cost_history_BCGD_R = [compute_cost(X, A, H, M, K) for X in update_history_BCGD_R]

accuracy_history_BCGD_R = [compute_accuracy(X, A, B, False) for X in update_history_BCGD_R]


print('Cost:', compute_cost(X_BCGD_GS, A, H, M, K))
print('Time:', np.sum(time_history_BCGD_GS))

plot_convergence(cost_history_BCGD_R, accuracy_history_BCGD_R)