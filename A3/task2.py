import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generate_data(random_state=2024):
    """
    Generates three clusters of 3D data with distinct means and covariances.

    Parameters:
    random_state (int): Seed for reproducibility.

    Returns:
    tuple: Combined data (3D array), labels (array indicating cluster assignments)
    """
    np.random.seed(random_state)

    # Generate three clusters with distinct means and covariance matrices
    mean1 = [10, 10, 10]
    cov1 = [[2, 1.6, 1], [1.6, 2, 0.6], [1, 0.6, 2]]
    data1 = np.random.multivariate_normal(mean1, cov1, 100)

    mean2 = [20, 20, 10]
    cov2 = [[2, -1, 0], [-1, 2, 0], [0, 0, 2]]
    data2 = np.random.multivariate_normal(mean2, cov2, 100)

    mean3 = [15, 15, 25]
    cov3 = [[2, 0, 0], [0, 2, 0], [0, 0, 10]]
    data3 = np.random.multivariate_normal(mean3, cov3, 100)

    # Combine the data and assign labels to each cluster
    data = np.vstack((data1, data2, data3))
    labels = np.array([0]*100 + [1]*100 + [2]*100)  # 0 for cluster 1, 1 for cluster 2, 2 for cluster 3
    return data, labels

# Function to standardize the dataset
def standardize_data(data):
    """
    Standardizes the dataset using StandardScaler.

    Parameters:
    data (array): Original dataset.

    Returns:
    array: Scaled dataset.
    """
    # TO DO: Instantiate StandardScaler and use fit_transform to scale the data



    return data_scaled

# Function to apply PCA to the dataset
def apply_pca(data_scaled, n_components):
    """
    Applies PCA to reduce dimensionality of the dataset.

    Parameters:
    data_scaled (array): Scaled dataset.
    n_components (int): Number of principal components.

    Returns:
    array: PCA transformed data.
    PCA object: The PCA object used.
    """
    # TO DO: Instantiate PCA with n_components and apply PCA transformation



    return data_pca, pca

# Function to plot the original 3D data
def plot_original_data(data, labels):
    """
    Plots the original 3D data with different colors for each cluster.

    Parameters:
    data (array): 3D dataset with combined points from all clusters.
    labels (array): Cluster labels for each data point, where each label indicates which cluster the point belongs to.
    """
    # TO DO: Use scatter plot to visualize the original data in 3D space

# Function to plot the XY projection
def plot_xy_projection(data, labels):
    """
    Plots the XY projection of the data with colors based on cluster labels.

    Parameters:
    data (array): 3D dataset.
    labels (array): Cluster labels for each data point.
    """
    # TO DO: Implement the function to plot the XY projection of the dataset

# Function to plot the XZ projection
def plot_xz_projection(data, labels):
    """
    Plots the XZ projection of the data with colors based on cluster labels.

    Parameters:
    data (array): 3D dataset.
    labels (array): Cluster labels for each data point.
    """
    # TO DO: Implement the function to plot the XY projection of the dataset

# Function to plot the YZ projection
def plot_yz_projection(data, labels):
    """
    Plots the YZ projection of the data with colors based on cluster labels.

    Parameters:
    data (array): 3D dataset.
    labels (array): Cluster labels for each data point.
    """
    # TO DO: Implement the function to plot the XY projection of the dataset

# Function to plot the PCA results
def plot_pca_results(data_pca, labels):
    """
    Plots the 2D PCA projection of the dataset with colors based on cluster labels.

    Parameters:
    data_pca (array): PCA-transformed dataset (2D projection).
    labels (array): Cluster labels for each data point.
    """
    # TO DO: Use scatter plot to visualize the 2D projection from PCA


if __name__ == "__main__":
    np.random.seed(2024)
    data, labels = generate_data() # Generate the data and labels
    
    # Plot original 3D data and projections
    plot_original_data(data, labels)
    plot_xy_projection(data, labels)
    plot_xz_projection(data, labels)
    plot_yz_projection(data, labels)

    # Standardize the dataset
    data_scaled = standardize_data(data)

    # TO DO: Fill in appropriate value for n_components
    data_pca, pca = apply_pca(data_scaled, n_components=_____)  # Apply PCA to the dataset
    
    plot_pca_results(data_pca, labels)   # Plot PCA results