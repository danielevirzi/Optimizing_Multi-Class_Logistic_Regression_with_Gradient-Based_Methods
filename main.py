###### Libraries ######

import time
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import accuracy_score




###### Utils ######

def Lipschitz_constant(A):
    """Compute the Lipschitz constant of the logistic loss function"""
    L = torch.norm(A, 2) * torch.norm(A, 'fro')
    return L


###### Datasets ######

class SyntheticDataset:
    def __init__(self, n_samples=1000, n_features=1000, n_classes=50, device='cuda'):
        """
        Initialize synthetic dataset generator
        
        Args:
            n_samples: number of samples
            n_features: number of features
            n_classes: number of classes
            device: computation device
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.device = device
        
        # Generate dataset
        self._generate_data()
        
    def _generate_data(self):
        """Generate synthetic data"""
        
        # Generate random input A
        self.A = torch.randn(self.n_samples, self.n_features).to(self.device)
        
        # Generate random matrix X
        self.X = torch.randn(self.n_features, self.n_classes).to(self.device)
        
        # Generate random error matrix E
        self.E = torch.randn(self.n_samples, self.n_classes).to(self.device)
                
        # Generate class labels
        Y = self.A @ self.X + self.E
        self.B = torch.argmax(Y, dim=1) + 1
        
        # Create one-hot encoding matrix H
        self.H = torch.nn.functional.one_hot(self.B - 1, num_classes=self.n_classes).float().to(self.device)
        
        # Create ones vector M
        self.M = torch.ones(1, self.n_samples).to(self.device)
        
        # Create ones vector K
        self.K = torch.ones(self.n_classes, 1).to(self.device)
        
        
        
    def get_data(self):
        """Return all dataset components"""
        return self.X, self.A, self.H, self.M, self.K, self.B
    
    def to(self, device):
        """Move dataset to specified device"""
        self.device = device
        self.X = self.X.to(device)
        self.A = self.A.to(device)
        self.H = self.H.to(device)
        self.M = self.M.to(device)
        self.K = self.K.to(device)
        return self
    
    
    
    

###### Model ######

class LogisticRegression:
    def __init__(self, alpha, num_iterations, epsilon, j, device='cuda'):
        """
        Initialize LogisticRegression optimizer
        
        Args:
            alpha: learning rate
            num_iterations: max iterations
            epsilon: convergence threshold
            j: number of blocks
            device: computation device
        """
        
        
        # Initialize hyperparameters
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.j = j
        self.device = device
        
        
        # Initialize tensors
        self.A = None
        self.X = None
        self.H = None
        self.M = None
        self.K = None
        self.B = None
        
        
        # History
        self.norm_history = []
        self.time_history = []
        self.update_history = []
        self.cost_history = []
        self.accuracy_history = []
        
        
    @staticmethod
    def gradient(X, A, H, K):
        """
        Compute the gradient of the loss function with respect to X.
        
        Args:
            X: Input tensor
            A: Matrix A
            H: Matrix H
            K: Ones tensor
        
        Returns:
            Gradient tensor
        
        """

        # Compute the exponential of A @ X
        exp_AX = torch.exp(A @ X)


        # Compute the derivative of the loss function with respect to X
        df_dX = - A.T @ (H - (exp_AX * (1 / (exp_AX @ K))))  # Perform broadcasting between exp_AX and (1 / (exp_AX @ I)) instead of compting the C matrix

        return df_dX
    
    
    @staticmethod    
    def gradient_block(X, A, H, K, index):
        """
        Compute the gradient of the loss function with respect to X for a block of columns.
        
        Args:
            X: Input tensor
            A: Matrix A
            H: Matrix H
            K: Ones tensor
            index: Block index
        
        Returns:
            Gradient tensor for a block of columns
    
        """
        
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
    
    
    @staticmethod
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
    
    
    @staticmethod
    def accuracy_score(y_true, y_pred):
        """
        Compute accuracy score using PyTorch operations.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Accuracy score as a scalar
        
        """
        accuracy = torch.sum(y_true == y_pred).float() / y_true.shape[0]
        return accuracy.item()
    
    
    @staticmethod
    def evaluate(X, A, B, index_0=True):
        """
        Evaluate the model using the accuracy score.
        
        Args:
            X: Input tensor
            A: Matrix A
            B: Class labels
            index_0: Boolean to indicate if the class labels start from 0 or 1
        
        Returns:
            Accuracy value as a scalar
        
        """

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
    
    
    def _gradient_descent(self, X, A, H, K):
        """
        Perform gradient descent optimization.
        
        Args:
            X: Input tensor
            A: Matrix A
            H: Matrix H
            K: Ones tensor
            alpha: learning rate
            num_iterations: max iterations
            epsilon: convergence threshold
        
        Returns:
            Optimal X tensor
        
        """
        
        # Ensure tensors are on correct device
        X = X.to(self.device)
        A = A.to(self.device)
        H = H.to(self.device)
        K = K.to(self.device)

        # Compute the gradient of the loss function with respect to X
        grad = self.gradient(X, A, H, K)

        # Initialize the iteration
        i = 0

        # Loop for a number of iterations
        while i < self.num_iterations and torch.norm(grad) > self.epsilon:


            ## Step 1: start the timer
            start = time.time()

            ## Step 2: update the parameters
            X = X - self.alpha * grad

            ## Step 3: update the gradient of the loss function with respect to the new X
            grad = self.gradient(X, A, H, K)

            ## Step 4: end the timer
            end = time.time()

            ## Step 5: save the time to the history
            self.time_history.append(end - start)

            ## Step 6: save the norm to the history
            self.norm_history.append(torch.norm(grad))

            ## Step 7: store the current X
            self.update_history.append(X)

            ## Step 8: update the iteration
            i += 1

        return X



    
    def fit(self, X, A, H, K, method="GD"):
        """
        Fit the model using the specified optimization method.
        
        Args:
            X: Input tensor
            A: Matrix A
            H: Matrix H
            K: Ones tensor
            method: Optimization method = ["GD", "BCGD Randomized", "BCGD Gauss-Southwell"]
        
        Returns:
            Optimal X tensor
        
        """
        
        # Initialize X
        self.X = X
        self.A = A
        self.H = H
        self.K = K
        
        
        # Initialize history
        self.update_history_GD = [self.X]
        
        # Perform optimization
        if method == "GD":
            self.X = self._gradient_descent(A, H, K)
        
        elif method == "BCGD Randomized":
            self.X = self._BCDG_randomized(A, H, K)
        
        elif method == "BCGD Gauss-Southwell":
            self.X = self._BCDG_GS(A, H, K)
        
        else:
            raise ValueError("Invalid optimization method")
        
        return self.X
    
    def plot_convergence(self, B, M):
        """Plot convergence history"""
        self.B = B
        self.M = M
        
        self.cost_history = [self.compute_cost(X, A, H, M, K) for X in self.update_history]

        self.accuracy_history = [self.evaluate(X, A, B, index_0=True) for X in self.update_history]
    
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Cost', color=color)
        ax1.plot(self.cost_history, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(self.accuracy_history, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('Gradient Descent: Cost and Accuracy')
        plt.show()
    
    



def main():
    dataset = SyntheticDataset(n_samples=1000, n_features=1000, n_classes=50)
    X, A, H, M, K, B = dataset.get_data()
    
    lr = 1 / Lipschitz_constant(A)
    
    model = LogisticRegression(alpha=lr, num_iterations=10000, 
                            epsilon=1e-6, j=50)
    
    model.fit(A, H, K, method="GD")
    model.plot_convergence(A, B, H, M, K)
    
    
if __name__ == "__main__":
    main()