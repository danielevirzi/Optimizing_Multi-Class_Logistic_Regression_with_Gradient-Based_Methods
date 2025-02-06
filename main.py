###### Libraries ######

import time
import torch
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from ucimlrepo import fetch_ucirepo


class SyntheticDataset:
    def __init__(self, n_samples=1000, n_features=1000, n_classes=50, seed=13):
        """
        Initialize synthetic dataset generator
        
        Args:
            n_samples:  number of samples
            n_features: number of features
            n_classes:  number of classes
            seed:       random seed
        """
        self.m = n_samples
        self.d = n_features
        self.k = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        # Set seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Generate dataset
        self._generate_data()
        
    def _generate_data(self):
        """Generate synthetic data"""
        
        # Generate random input A
        self.A = torch.randn(self.m, self.d).to(self.device)
        
        # Generate random matrix X
        self.X = torch.randn(self.d, self.k).to(self.device)    
        
        # Generate random error matrix E
        self.E = torch.randn(self.m, self.k).to(self.device)
                
        # Generate class labels
        Y = self.A @ self.X + self.E
        self.B = torch.argmax(Y, dim=1) + 1  # Add 1 to get a 1-based index
        
        # Create one-hot encoding matrix H
        self.H = torch.nn.functional.one_hot(self.B - 1, num_classes=self.k).float().to(self.device)
        
        # Create ones vector M
        self.M = torch.ones(1, self.m).to(self.device)
        
        # Create ones vector K
        self.K = torch.ones(self.k, 1).to(self.device)
        
        # Normalize weights with Xavier initialization for X_train
        self.X_train = torch.randn(self.d, self.k).to(self.device) * torch.sqrt(torch.tensor(1 / self.d).to(self.device))
        
    def get_data(self):
        """Return all dataset components"""
        return self.X_train, self.A, self.H, self.M, self.K, self.B   
    
class LogisticRegression:
    def __init__(self, alpha, num_iterations, epsilon, j, device='cuda', seed=13):
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.j = j
        self.device = device
        self.seed = seed
        
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        self.A = None
        self.X = None
        self.H = None
        self.M = None
        self.K = None
        self.B = None 
        
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
            Compute the gradient of the loss function with respect to the index-th column of X.
            
            Args:
                X: Input tensor
                A: Matrix A
                H: Matrix H
                K: Ones tensor
                index: Column index of X
            
            Returns:
                Gradient tensor
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
        Compute the cost function.
        
        Args:
            X: Input tensor
            A: Matrix A
            H: Matrix H
            M: Ones tensor
            K: Ones tensor
            
        Returns:
            Cost value
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
        Compute the accuracy of the model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy value
        """
        
        # Compute the accuracy of the model
        accuracy = torch.sum(y_true == y_pred).float() / y_true.shape[0]

        return accuracy.item()
    
    def evaluate(self, X, A, B, index_0=True):
        """
        Evaluate the model.
        
        Args:
            X: Input tensor
            A: Matrix A
            B: True labels
            index_0: Index of the true labels
        
        Returns:
            Accuracy value
        """

        # Compute the softmax of the matrix product
        Y_pred = torch.nn.functional.softmax(A @ X, dim=0)


        if index_0 == True:
            # Get the index of the maximum value in each row
            B_pred = torch.argmax(Y_pred, dim=1)

        else:
            # Get the index of the maximum value in each row
            B_pred = torch.argmax(Y_pred, dim=1) + 1

        # Compute the accuracy of the model
        
        accuracy = self.accuracy_score(B, B_pred)

        return accuracy
    
    def _gradient_descent(self):
        """ Vanilla Gradient Descent """
        X = self.X.to(self.device)
        A = self.A.to(self.device)
        H = self.H.to(self.device)
        K = self.K.to(self.device)

        grad = self.gradient(X, A, H, K)
        self.norm_history = []
        self.time_history = []
        self.update_history = []

        for _ in range(self.num_iterations):
            if torch.linalg.norm(grad) <= self.epsilon:
                break

            start = time.time()
            X = X - self.alpha * grad
            grad = self.gradient(X, A, H, K)
            end = time.time()

            self.time_history.append(end - start)
            self.norm_history.append(torch.linalg.norm(grad))
            self.update_history.append(X.clone())

        self.X = X

    def _BCDG_GS(self):
        """ Block Coordinate Descent Gauss-Southwell """
        X = self.X.to(self.device)
        A = self.A.to(self.device)
        H = self.H.to(self.device)
        K = self.K.to(self.device)

        grad = self.gradient(X, A, H, K)
        norm_grad_column = torch.norm(grad, dim=0)

        self.norm_history = []
        self.time_history = []
        self.update_history = []

        for _ in range(self.num_iterations):
            if torch.linalg.norm(grad) <= self.epsilon:
                break

            start = time.time()
            gs_index = torch.argmax(norm_grad_column).item()
            gs_gradient_block_new = self.gradient_block(X, A, H, K, gs_index)

            norm_grad_column[gs_index] = torch.linalg.norm(gs_gradient_block_new)
            grad[:, gs_index:gs_index+1] = gs_gradient_block_new
            X = X - self.alpha * grad
            end = time.time()

            self.time_history.append(end - start)
            self.norm_history.append(torch.linalg.norm(grad))
            self.update_history.append(X.clone())

        self.X = X

    def _BCDG_randomized(self):
        """ Block Coordinate Descent Randomized """
        X = self.X.to(self.device)
        A = self.A.to(self.device)
        H = self.H.to(self.device)
        K = self.K.to(self.device)

        grad = self.gradient(X, A, H, K)

        self.norm_history = []
        self.time_history = []
        self.update_history = []

        for _ in range(self.num_iterations):
            if torch.linalg.norm(grad) <= self.epsilon:
                break

            start = time.time()
            rand_index = torch.randint(0, self.j, (1,)).item()
            random_gradient_block = self.gradient_block(X, A, H, K, rand_index)

            grad[:, rand_index:rand_index+1] = random_gradient_block
            X = X - self.alpha * grad
            end = time.time()

            self.time_history.append(end - start)
            self.norm_history.append(torch.linalg.norm(grad))
            self.update_history.append(X.clone())

        self.X = X

    def fit(self, X, A, H, K, B, M, method="GD"):
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
        self.X = X.to(self.device)
        self.A = A.to(self.device)
        self.H = H.to(self.device)
        self.K = K.to(self.device)
        self.B = B.to(self.device)
        self.M = M.to(self.device)

        # Reset history
        self.norm_history = []
        self.time_history = []
        self.update_history = []

        if method == "GD":
            self._gradient_descent()
        elif method == "BCGD Randomized":
            self._BCDG_randomized()
        elif method == "BCGD Gauss-Southwell":
            self._BCDG_GS()
        else:
            raise ValueError("Invalid optimization method")
        
        
        # Compute the cost and accuracy
        for X in self.update_history:
            cost = self.compute_cost(X, A, H, self.M, K)
            accuracy = self.evaluate(X, A, self.B, index_0=False)
            self.cost_history.append(cost)
            self.accuracy_history.append(accuracy)

        return self.X




    def plot_convergence(self):
        """ Plot the convergence of the optimization method """
        print(f"Final cost: {self.cost_history[-1]}")
        print(f"Final accuracy: {self.accuracy_history[-1] * 100}%")
        print(f"Total time: {sum(self.time_history):.4f} seconds")
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
    
        
def Lipschitz_constant(A):
    """ Compute the Lipschitz constant of the gradient of the loss function with respect to X. """
    L = torch.linalg.norm(A, 2) * torch.linalg.norm(A, 'fro')
    return L    
        
def main():
    # Dataset parameters
    m, d, k = 1000, 1000, 50
    
    dataset = SyntheticDataset(n_samples=m, n_features=d, n_classes=k, seed=13)
    X_train, A, H, M, K, B = dataset.get_data()

    # Hyperparameters
    lr = 1 / Lipschitz_constant(A)
    num_iterations = 2000
    tolerance = 1e-6
    j = k
    
    model = LogisticRegression(lr, num_iterations, tolerance, j)
    model.fit(X_train, A, H, K, B, M, method="GD")
    #model.fit(X_train, A, H, K, B, M, method="BCGD Randomized")
    #model.fit(X_train, A, H, K, B, M, method="BCGD Gauss-Southwell")    
    model.plot_convergence()


if __name__ == "__main__":
    main()