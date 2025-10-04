# No external libraries are allowed to be imported in this file
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to generate the dataset
def generate_swiss_roll(n_samples, noise=0.1, random_state=2024):
    """
    Generates the Swiss Roll dataset.

    Parameters:
    n_samples (int): Number of samples to generate.
    noise (float): Noise factor.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: Generated data (X) and the color labels (color)
    """
    X, color = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, color

# Function to apply PCA to the dataset
def apply_pca(X, n_components, random_state=2024):
    """
    Applies PCA to the Swiss Roll dataset.

    Parameters:
    X (array): The input dataset.
    n_components (int): Number of principal components to retain.
    random_state (int): Random seed for reproducibility.

    Returns:
    array: Transformed data with PCA applied.
    """
    # TO DO: Create a pipeline to apply StandardScaler and PCA



    return X_pca

# Function to plot the original 3D data
def plot_3d_data(X, color):
    """
    Plots the 3D Swiss Roll dataset.

    Parameters:
    X (array): The 3D dataset.
    color (array): The color labels for the points.
    """
    # TO DO: Use scatter plot to visualize the original data in 3D space





# Function to plot the XZ projection
def plot_xz_projection(X, color):
    """
    Plots the XZ projection of the Swiss Roll dataset.

    Parameters:
    X (array): The 3D dataset.
    color (array): The color labels for the points.
    """
    # TO DO: Use scatter plot to visualize the XZ projection





# Function to plot the 2D PCA projection
def plot_pca_projection(X_pca, color):
    """
    Plots the 2D PCA projection of the Swiss Roll dataset.

    Parameters:
    X_pca (array): The PCA-transformed dataset.
    color (array): The color labels for the points.
    """
    # TO DO: Use scatter plot to visualize the 2D projection from PCA





if __name__ == "__main__":
    np.random.seed(2024)
    X, color = generate_swiss_roll(n_samples=1500, noise=0.1, random_state=2024)

    #TO DO: Fill in appropriate value for n_components
    X_pca = apply_pca(X, n_components=_____, random_state=2024) # Apply PCA

    plot_3d_data(X, color)              # Visualize the original 3D dataset
    plot_xz_projection(X, color)        # Visualize the XZ projection
    plot_pca_projection(X_pca, color)   # Visualize the PCA 2D projection