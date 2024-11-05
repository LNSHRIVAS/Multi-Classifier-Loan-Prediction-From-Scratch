from common import *

class PrincipleComponentAnalysis():
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.scale = None  # Will store the standard deviation of each feature
        self.components = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)  # Mean of each feature (column-wise mean)
        self.scale = np.std(X, axis=0)  # Standard deviation of each feature

        x_standardized = (X - self.mean) / self.scale # We are standardizning the data here.

        cov_matrix = np.cov(x_standardized.T) # We compute the covarience matrix of the standardized data 
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) # Here we are computing the eigenvalues and eigenvector of the covarinece matrix. 

        idx = np.argsort(eigenvalues)[::-1] #this indexes will sort the eignevalues in decreasing order 
        eigenvalues = eigenvalues[idx] # This are sorted eigenvalues 
        eigenvectors = eigenvectors[:, idx] # This are sorted eigenvectors

        self.components = eigenvectors[:, :self.n_components]  #Now here we select the top most components eigenvectors as principle components.

    def transform(self, X):
        x_standardized = (X - self.mean) / self.scale
        return np.dot(x_standardized, self.components)

    def apply_pca_transformation(self, X):
        return self.transform(X)