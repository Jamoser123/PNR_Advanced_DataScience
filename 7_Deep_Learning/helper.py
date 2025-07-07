import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, islice
from sklearn import cluster

from IPython.display import display, HTML

from sklearn.calibration import cross_val_predict
from sklearn.manifold import MDS
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, auc, confusion_matrix, f1_score, make_scorer, precision_score, r2_score, recall_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import log_loss, mean_squared_error

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import statsmodels.formula.api as smf

import seaborn as sns
from matplotlib import cm

from sklearn.metrics import pairwise_distances
import gower
from sklearn.datasets import make_blobs, make_classification, make_moons, make_circles

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score, silhouette_samples
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import kmedoids

import matplotlib.lines as mlines

from sklearn.model_selection import cross_val_score
import numpy as np

import torch
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Layout
import ipywidgets as widgets
from torch import nn
import torch.nn.functional as F

import torch.optim as optim


def normalize(data):
    """
    Normalizes the input data using its mean and standard deviation.

    Parameters:
        data (pandas.Series or numpy.ndarray): The data to be normalized.

    Returns:
        pandas.Series or numpy.ndarray: The normalized data with zero mean and unit variance.
    """
    # Normalize using training set statistics
    mean = data.mean()
    std = data.std()
    return (data - mean) / std

def create_formula(degree, feature_col, target_col):
    """
    Generates a regression formula string for polynomial features up to a specified degree.

    Args:
        degree (int): The degree of the polynomial features to include.
        feature_col (str): The name of the feature (independent variable) column.
        target_col (str): The name of the target (dependent variable) column.

    Returns:
        str: A formula string in the format 'target_col ~ feature_col + I(feature_col**2) + ...',
             suitable for use with statistical modeling libraries such as statsmodels.

    Example:
        create_formula(3, 'x', 'y')
        # Returns: 'y ~ x + I(x**2) + I(x**3)'
    """
    terms = [feature_col] + [f"I({feature_col}**{i})" for i in range(2, degree + 1)]
    return f"{target_col} ~ " + " + ".join(terms)

def train_linear_regression(X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test, learning_rate=0.01, n_iter=120):
    """
    Trains a simple linear regression model using gradient descent and plots RMSE curves.
    This function fits a linear regression model (y = w0 + w1 * x) to the provided normalized training data using gradient descent.
    It tracks and plots the RMSE for both training and validation sets over iterations, and prints the final RMSE for train, validation, and test sets.
    Parameters:
        X_train_norm (np.ndarray): Normalized training feature data.
        y_train (np.ndarray): Training target values.
        X_val_norm (np.ndarray): Normalized validation feature data.
        y_val (np.ndarray): Validation target values.
        X_test_norm (np.ndarray): Normalized test feature data.
        y_test (np.ndarray): Test target values.
        learning_rate (float, optional): Learning rate for gradient descent. Default is 0.01.
        n_iter (int, optional): Number of gradient descent iterations. Default is 120.
    Returns:
        tuple: (w0, w1, w0_history, w1_history, rmse_train_history)
            w0 (float): Final intercept parameter.
            w1 (float): Final slope parameter.
            w0_history (list): History of w0 values during training.
            w1_history (list): History of w1 values during training.
            rmse_train_history (list): History of training RMSE values.
    """
    # Initialize parameters
    w0 = 0.0
    w1 = 0.0
    rmse_train_history = []
    rmse_val_history = []
    w0_history = []
    w1_history = []

    for i in range(n_iter):
        # --- Training predictions and gradients ---
        y_pred_train = w0 + w1 * X_train_norm
        error_train = y_pred_train - y_train
        rmse_train = np.sqrt(np.mean(error_train ** 2))
        rmse_train_history.append(rmse_train)
        
        # --- Validation predictions ---
        y_pred_val = w0 + w1 * X_val_norm
        error_val = y_pred_val - y_val
        rmse_val = np.sqrt(np.mean(error_val ** 2))
        rmse_val_history.append(rmse_val)

        # --- Store parameter history for plotting ---
        w0_history.append(w0)
        w1_history.append(w1)
        
        # --- Parameter updates using training gradients ---
        grad_w0 = 2 * np.mean(error_train)
        grad_w1 = 2 * np.mean(error_train * X_train_norm)
        w0 -= learning_rate * grad_w0
        w1 -= learning_rate * grad_w1

    # Plot RMSE curves
    plt.figure(figsize=(7,4))
    plt.plot(rmse_train_history, label="Train RMSE")
    plt.plot(rmse_val_history, label="Validation RMSE")
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Train & Validation RMSE During Optimization')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Final RMSEs
    y_pred_train_final = w0 + w1 * X_train_norm
    y_pred_val_final = w0 + w1 * X_val_norm
    y_pred_test_final = w0 + w1 * X_test_norm

    rmse_train_final = np.sqrt(np.mean((y_pred_train_final - y_train) ** 2))
    rmse_val_final = np.sqrt(np.mean((y_pred_val_final - y_val) ** 2))
    rmse_test_final = np.sqrt(np.mean((y_pred_test_final - y_test) ** 2))

    print(f"Final Train RMSE:      {rmse_train_final:.3f}")
    print(f"Final Validation RMSE: {rmse_val_final:.3f}")
    print(f"Final Test RMSE:       {rmse_test_final:.3f}")

    return (w0, w1, w0_history, w1_history, rmse_train_history)

def visualize_fit_evolution(X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test, training_params, n_iter, feature_col='therapy_duration', target_col='cognitive_score_after'):
    """
    Visualizes the evolution of a linear regression model's fit and residuals during training.
    This function creates a 2x2 grid of subplots to illustrate the progression of the linear regression fit 
    at the beginning, middle, and end of training on the training data, as well as the final model's performance 
    on the test data. Residuals are shown as dashed lines, and RMSE values are displayed for each stage.
    Parameters
    ----------
    X_train_norm : array-like
        Normalized feature values for the training set.
    y_train : array-like
        Target values for the training set.
    X_val_norm : array-like
        Normalized feature values for the validation set.
    y_val : array-like
        Target values for the validation set.
    X_test_norm : array-like
        Normalized feature values for the test set.
    y_test : array-like
        Target values for the test set.
    training_params : tuple
        Tuple containing (w0, w1, w0_history, w1_history, _) where:
            w0 : float
                Final intercept of the model.
            w1 : float
                Final slope of the model.
            w0_history : list or array
                History of intercept values during training.
            w1_history : list or array
                History of slope values during training.
            _ : any
                Unused placeholder for additional parameters.
    n_iter : int
        Total number of training iterations.
    feature_col : str, optional
        Name of the feature column (default is 'therapy_duration').
    target_col : str, optional
        Name of the target column (default is 'cognitive_score_after').
    Returns
    -------
    None
        Displays the plots and prints final RMSE values to the console.
    """

    # Unpack training parameters
    w0, w1, w0_history, w1_history, _ = training_params


    # Create subplots to show progression of fit with residuals
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Linear Regression Fit Progression During Training (with Residuals)', fontsize=16)

    # Final RMSEs
    y_pred_train_final = w0 + w1 * X_train_norm
    y_pred_val_final = w0 + w1 * X_val_norm
    y_pred_test_final = w0 + w1 * X_test_norm

    rmse_train_final = np.sqrt(np.mean((y_pred_train_final - y_train) ** 2))
    rmse_val_final = np.sqrt(np.mean((y_pred_val_final - y_val) ** 2))
    rmse_test_final = np.sqrt(np.mean((y_pred_test_final - y_test) ** 2))

    print(f"Final Train RMSE:      {rmse_train_final:.3f}")
    print(f"Final Validation RMSE: {rmse_val_final:.3f}")
    print(f"Final Test RMSE:       {rmse_test_final:.3f}")

    # Define which iterations to show
    iterations_to_show = [0, n_iter//2, n_iter-1]  # Beginning, middle, last iteration
    iteration_labels = ['Beginning (Iteration 1)', f'Middle (Iteration {n_iter//2 + 1})', f'Final (Iteration {n_iter})']

    # Plot training progression in first 3 subplots
    for i, (iter_idx, label) in enumerate(zip(iterations_to_show, iteration_labels)):
        ax = axes[i//2, i%2]
        
        # Get parameters at this iteration
        w0_iter = w0_history[iter_idx]
        w1_iter = w1_history[iter_idx]
        
        # Make predictions with parameters from this iteration
        y_pred_iter = w0_iter + w1_iter * X_train_norm
        rmse_iter = np.sqrt(np.mean((y_pred_iter - y_train) ** 2))
        
        # Plot residual lines (draw first so they appear behind points)
        for j in range(len(X_train_norm)):
            ax.plot([X_train_norm[j], X_train_norm[j]], 
                    [y_train[j], y_pred_iter[j]], 
                    color='gray', linestyle='--', alpha=0.5)
        
        # Plot data points and regression line
        ax.scatter(X_train_norm, y_train, alpha=0.8, color='blue', s=30, zorder=3)
        ax.plot(X_train_norm, y_pred_iter, color='red', linewidth=2, 
                label=f'Linear Fit (RMSE: {rmse_iter:.2f})', zorder=2)
        
        ax.set_title(f'{label} - Training Data', fontsize=12)
        ax.set_xlabel(f'{feature_col}')
        ax.set_ylabel(f'{target_col}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot final model performance on test set in 4th subplot
    ax = axes[1, 1]

    # Plot residual lines for test data
    for j in range(len(X_test_norm)):
        ax.plot([X_test_norm[j], X_test_norm[j]], 
                [y_test[j], y_pred_test_final[j]], 
                color='black', linewidth=0.8, alpha=0.6)

    # Plot test data points and regression line
    ax.scatter(X_test_norm, y_test, alpha=0.8, color='green', s=30, zorder=3)
    ax.plot(X_test_norm, y_pred_test_final, color='red', linewidth=2, 
            label=f'Linear Fit (RMSE: {rmse_test_final:.2f})', zorder=2)

    ax.set_title('Final Model - Test Data', fontsize=12)
    ax.set_xlabel(f'{feature_col}')
    ax.set_ylabel(f'{target_col}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_rmse_loss_surface_with_arrow(X_norm, y, training_params, w0_range_offset=70, w1_range_offset=70, grid_points=100):
    """
    Plots the RMSE loss surface for a linear regression model with respect to its parameters (intercept and slope),
    overlaying the gradient descent optimization path and the closed-form solution.
    """
    
    # Unpack training parameters
    w0_hist, w1_hist, rmse_hist = training_params
    
    # Ensure X_norm is 1D for this function
    if X_norm.ndim > 1:
        X_norm = X_norm.flatten()  # Convert to 1D
    
    # Closed-form solution for reference
    X_design = np.vstack([np.ones_like(X_norm), X_norm]).T
    opt_w = np.linalg.lstsq(X_design, y, rcond=None)[0]
    opt_w0, opt_w1 = opt_w

    print(f"Closed-form Optimum: w0 = {opt_w0:.3f}, w1 = {opt_w1:.3f}")
    # print rmse of closed-form solution
    y_pred_opt = opt_w0 + opt_w1 * X_norm
    print(f"RMSE of Closed-form Optimum: {np.sqrt(np.mean((y - y_pred_opt) ** 2)):.3f}")

    # Center grid around closed-form optimum
    w0_range = np.linspace(opt_w0 - w0_range_offset, opt_w0 + w0_range_offset, grid_points)
    w1_range = np.linspace(opt_w1 - w1_range_offset, opt_w1 + w1_range_offset, grid_points)
    W0, W1 = np.meshgrid(w0_range, w1_range)
    rmse_surface = np.zeros(W0.shape)

    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            y_pred_grid = W0[i, j] + W1[i, j] * X_norm
            rmse_surface[i, j] = np.sqrt(np.mean((y - y_pred_grid) ** 2))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W0, W1, rmse_surface, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Intercept (w0)')
    ax.set_ylabel('Slope (w1)')
    ax.set_zlabel('RMSE Loss')
    ax.set_title('RMSE Loss Surface for Linear Regression')

    ax.scatter(opt_w0, opt_w1, np.sqrt(np.mean((y - (opt_w0 + opt_w1 * X_norm))**2)), 
               color='orange', s=150, label='Closed-form Optimum')
    
    if w0_hist is not None and w1_hist is not None and rmse_hist is not None:
        ax.plot(w0_hist, w1_hist, rmse_hist, color='red', marker='o', linewidth=2, label='Gradient Descent Path')

    ax.legend()
    plt.show()

    plt.figure(figsize=(7,5))
    cp = plt.contourf(W0, W1, rmse_surface, levels=30, cmap='viridis')
    plt.xlabel('Intercept (w0)')
    plt.ylabel('Slope (w1)')
    plt.title('RMSE Loss Contour for Linear Regression')
    plt.colorbar(cp, label='RMSE')
    plt.scatter([opt_w0], [opt_w1], c='orange', label='Closed-form Optimum')

    if w0_hist is not None and w1_hist is not None:
        plt.plot(w0_hist, w1_hist, color='red', marker='o', linewidth=2, label='Gradient Descent Path')
    
    plt.legend()
    plt.show()

def downsample_history(training_params, k=5):
    """
    Downsamples the parameter and RMSE histories from gradient descent.

    Parameters:
        training_params (tuple): Tuple containing (w0, w1, w0_history, w1_history, rmse_train_history).
        k (int, optional): Keep every k-th point in the histories. Default is 5.

    Returns:
        tuple: Downsampled (w0_history, w1_history, rmse_train_history).
    """
    _, _, w0_hist, w1_hist, rmse_hist = training_params
    histories = [w0_hist, w1_hist, rmse_hist]
    return tuple([h[::k] for h in histories])

def evaluate_model(y_true, y_pred, y_pred_prob):
    """
    Displays an interactive HTML table summarizing model predictions versus actual values.
    This function generates a styled HTML table showing up to the first 20 samples from the provided true labels,
    predicted labels, and predicted probabilities. Each row displays the sample index, actual value, predicted value,
    predicted probability, and a visual indicator (✓ or ✗) for whether the prediction matches the actual value.
    Rows are color-coded for correct and incorrect predictions to enhance readability.

    Parameters:
    y_true : pandas.Series or array-like
        The ground truth (actual) labels.
    y_pred : array-like
        The predicted labels from the model.
    y_pred_prob : array-like
        The predicted probabilities or confidence scores for the positive class.
    
    Returns:
    None
        The function displays an HTML table in a Jupyter notebook environment and does not return a value.
    """

    # Create HTML table with styling
    html_table = f"""
    <div style="font-family: Arial, sans-serif; margin: 20px 0;">
        <table style="border-collapse: collapse; width: 100%; max-width: 600px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <thead>
                <tr style="background-color: #4CAF50; color: white;">
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Index</th>
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Actual</th>
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Predicted</th>
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Probability</th>
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Match</th>
                </tr>
            </thead>
            <tbody>
    """

    # Add rows to the table
    for i in range(min(20, len(y_true))):
        actual = y_true.iloc[i]
        predicted = y_pred[i]
        probability = y_pred_prob[i]
        match = "✓" if actual == predicted else "✗"
        
        # Color coding for correct/incorrect predictions
        row_color = "#f8fff8" if actual == predicted else "#fff8f8"
        match_color = "#28a745" if actual == predicted else "#dc3545"
        
        html_table += f"""
                <tr style="background-color: {row_color};">
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; color: black;">{i}</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; color: black; font-weight: bold;">{actual}</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; color: black;font-weight: bold;">{predicted}</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; color: black;">{probability:.4f}</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; color: {match_color}; font-weight: bold; font-size: 16px;">{match}</td>
                </tr>
        """

    # Display the HTML table
    display(HTML(html_table))

def plot_model(data, feature_col, target_col, predictions):
    """
    Plots the relationship between a feature and a target variable along with model predictions.

    This function creates a scatter plot of the actual data points for the specified feature and target columns,
    and overlays a line plot of the model's predictions. It is useful for visualizing the fit of a regression model,
    highlighting cases of high bias (underfitting) with a linear fit.

    Args:
        data (pd.DataFrame): The dataset containing the feature and target columns.
        feature_col (str): The name of the column to use as the feature (x-axis).
        target_col (str): The name of the column to use as the target variable (y-axis).
        predictions (array-like): The predicted values corresponding to the feature column.

    Returns:
        None: Displays the plot.
    """
    plt.scatter(data[feature_col], data[target_col], alpha=0.7)
    plt.plot(data[feature_col], predictions, color="red", label="Linear Fit (High Bias)")
    plt.title("High Bias: Linear Fit on Ferritin Data")
    plt.xlabel(f"{feature_col}")# Age (years)")
    plt.ylabel(f"{target_col}")#Ferritin Concentration (µg/L)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def plot_subset_models(data, feature_col, target_col, degree):
    """
    Plots the predictions of polynomial regression models fitted on random subsets of the data.
    This function visualizes the variability ("variance") in model predictions by fitting multiple
    polynomial regression models (of specified degree) on different random subsets of the input data.
    Each model's predictions are plotted alongside the original data points.

    Args:
        data (pd.DataFrame): The input dataset containing features and target columns.
        feature_col (str): The name of the feature column to use as the predictor variable.
        target_col (str): The name of the target column to use as the response variable.
        degree (int): The degree of the polynomial regression model to fit.
    Returns:
        None. Displays a matplotlib plot showing the data and model predictions.
    """
    plt.scatter(data[feature_col], data[target_col], alpha=0.7)
    formula = create_formula(degree, feature_col, target_col)

    for i in range(10):  # Fit 10 models on random subsets
        subset = data.sample(frac=0.7, random_state=i)
        model = smf.ols(formula=formula, data=subset).fit()
        predictions = model.predict(data[feature_col])
        plt.plot(data[feature_col], predictions, alpha=0.5, label=f"Subset {i+1}")
    
    plt.ylim(data[target_col].min() - 5, data[target_col].max() + 5)

    plt.title(f"High Variance: {degree}-Degree Models on Subsets")
    plt.xlabel(f"{feature_col}")#"Age (years)")
    plt.ylabel(f"{target_col}")#"Ferritin Concentration (µg/L)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def plot_degrees_model(data, feature_col, target_col, degrees):
    """
    Plots polynomial regression fits of varying degrees for a given dataset and feature.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to plot and fit.
        feature_col (str): The name of the column to use as the independent variable (x-axis).
        target_col (str): The name of the column to use as the dependent variable (y-axis).
        degrees (list of int): A list of polynomial degrees to fit and plot.
        
    The function displays a scatter plot of the data and overlays polynomial regression lines
    of the specified degrees. Each fit is labeled by its degree.
    """
    plt.scatter(data[feature_col], data[target_col], alpha=0.7)

    for degree in degrees:
        
        # Fit the interpolating polynomial
        formula = create_formula(degree, feature_col=feature_col, target_col=target_col)
        model = smf.ols(formula=formula, data=data).fit()
        predictions = model.predict(data[feature_col])
        plt.plot(data[feature_col], predictions, alpha=0.5, label=f"Degree {degree} Fit")

    plt.title("Polynomial Fits of Increasing Degree")
    plt.xlabel(f"{feature_col}")
    plt.ylabel(f"{target_col}")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def plot_dataset_models(data, new_data1, new_data2, feature_col='Age_std', target_col='Ferritin', degree=3):
    """
    Plots the original dataset and two new datasets along with polynomial regression model fits.
    This function creates a 3-panel plot:
      1. The original dataset with its polynomial regression fit.
      2. The first new dataset with both the original model fit and a new fit trained on this dataset.
      3. The second new dataset with the original fit, the first new fit, and a new fit trained on this dataset.
    Parameters:
        data (pd.DataFrame): The original dataset containing at least the feature and target columns.
        new_data1 (pd.DataFrame): The first new dataset for comparison and model fitting.
        new_data2 (pd.DataFrame): The second new dataset for comparison and model fitting.
        feature_col (str, optional): The name of the feature column to use for modeling (default is 'Age_std').
        target_col (str, optional): The name of the target column to predict (default is 'Ferritin').
        degree (int, optional): The degree of the polynomial regression model (default is 3).
    Returns:
        None: This function displays the plots and does not return any value.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # Normalize age data
    data['Age_std'] = normalize(data['Age'])
    new_data1['Age_std'] = normalize(new_data1['Age'])
    new_data2['Age_std'] = normalize(new_data2['Age'])

    # Fit initial model on original data
    formula = create_formula(degree, feature_col=feature_col, target_col=target_col)
    model_orig = smf.ols(formula=formula, data=data).fit()
    
    # Fit model on new_data1
    model_new1 = smf.ols(formula=formula, data=new_data1).fit()
    
    # Fit model on new_data2
    model_new2 = smf.ols(formula=formula, data=new_data2).fit()
    
    # Plot 1: Original data + original fit
    axes[0].scatter(data[feature_col], data[target_col], alpha=0.7)
    pred_orig = model_orig.predict(data[feature_col])
    axes[0].plot(data[feature_col], pred_orig, 'r', label=f"Original Fit (degree {degree})")
    axes[0].set_title("Original Dataset")
    axes[0].set_xlabel(f"{feature_col}")
    axes[0].set_ylabel(f"{target_col}")
    axes[0].legend()
    
    # Plot 2: New data 1 + original fit + new fit 1
    axes[1].scatter(new_data1[feature_col], new_data1[target_col], alpha=0.7)
    pred_orig2 = model_orig.predict(new_data1[feature_col])
    pred_new1 = model_new1.predict(new_data1[feature_col])
    axes[1].plot(new_data1[feature_col], pred_orig2, 'r', label="Original Fit")
    axes[1].plot(new_data1[feature_col], pred_new1, 'g', label=f"New Dataset 1 Fit (degree {degree})")
    axes[1].set_title("New Dataset 1")
    axes[1].set_xlabel(f"{feature_col}")
    axes[1].legend()
    
    # Plot 3: New data 2 + original fit + new fit 1 + new fit 2
    axes[2].scatter(new_data2[feature_col], new_data2[target_col], alpha=0.7)
    pred_orig3 = model_orig.predict(new_data2[feature_col])
    pred_new1_2 = model_new1.predict(new_data2[feature_col])
    pred_new2 = model_new2.predict(new_data2[feature_col])
    axes[2].plot(new_data2[feature_col], pred_orig3, 'r', label="Original Fit")
    axes[2].plot(new_data2[feature_col], pred_new1_2, 'g', label="New Dataset 1 Fit")
    axes[2].plot(new_data2[feature_col], pred_new2, 'b', label=f"New Dataset 2 Fit (degree {degree})")
    axes[2].set_title("New Dataset 2")
    axes[2].set_xlabel(f"{feature_col}")
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

def plot_loss_progression(data, feature_col, target_col, degree):
    """
    Plots the progression of training and testing loss (mean squared error) for polynomial regression models
    of increasing degree, illustrating the bias-variance tradeoff.
    Parameters:
        data (pd.DataFrame): The input dataset containing features and target.
        feature_col (str): The name of the feature column to use for regression.
        target_col (str): The name of the target column.
        degree (int): The maximum degree of the polynomial to evaluate.
    Returns:
        None: This function displays a plot and does not return any value.
    Plot the loss progression for polynomial regression.
    """
    # Normalize the feature
    data[feature_col + '_std'] = normalize(data[feature_col])
    
    # Split the data
    X = data[[feature_col + '_std']]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate MSE for different polynomial degrees
    train_errors = []
    test_errors = []

    for deg in range(1, degree + 1):  
        # Fit the interpolating polynomial
        X_train_1d = X_train.values.ravel()  # Convert X_train to a 1D array
        X_test_1d = X_test.values.ravel()  # Convert X_test to a 1D array
        
        coefficients = np.polyfit(X_train_1d, y_train, deg=deg)
        polynomial = np.poly1d(coefficients)
        
        # Predict for the train and test datasets
        y_train_pred = polynomial(X_train_1d)
        y_test_pred = polynomial(X_test_1d)
        
        # Calculate mean squared errors
        train_errors.append(mean_squared_error(y_train, y_train_pred))
        test_errors.append(mean_squared_error(y_test, y_test_pred))
        
    # Plot the errors
    plt.plot(range(1, degree + 1), train_errors, label="Training Error", marker="o")
    plt.plot(range(1, degree + 1), test_errors, label="Testing Error", marker="o")
    plt.title("Train vs. Test Error (Bias-Variance Tradeoff)")
    plt.xlabel("Model Complexity (Degree of Polynomial)")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()

### Dimensionality Reduction and Similarity Measures
def simple_matching_distance(X):
    """
    Computes the simple matching distance matrix for a given categorical dataset.

    The simple matching distance between two samples is defined as 1 minus the proportion
    of matching feature values. This function returns a symmetric distance matrix where
    each entry (i, j) represents the simple matching distance between sample i and sample j.

    Args:
        X (np.ndarray): A 2D NumPy array of shape (n_samples, n_features) containing categorical data.

    Returns:
        np.ndarray: A 2D NumPy array of shape (n_samples, n_samples) containing the pairwise
            simple matching distances between samples.
    """
    n = X.shape[0]
    smd = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matches = np.sum(X[i] == X[j])
            sm_coef = matches / X.shape[1]
            smd[i, j] = 1 - sm_coef  # Distance = 1 - similarity
    return smd

def pairwise_hamming_distance_similarity(X):
    """
    Efficiently compute the pairwise Hamming distance (count of differing elements)
    and similarity (count of matching elements) matrices for a 2D numpy array X.

    Returns:
        dist (n x n numpy array): dist[i, j] is the Hamming distance (count) between X[i] and X[j]
        sim (n x n numpy array): sim[i, j] is the number of matches between X[i] and X[j]
    """
    n, m = X.shape
    dist = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):  # Only upper triangle
            differences = np.sum(X[i] != X[j])
            dist[i, j] = differences
            dist[j, i] = differences  # Symmetry
    return dist

def plot_mnist(mnist):
    """
    Displays 10 sample images from the MNIST dataset with their corresponding labels.
    Parameters:
        mnist (dict): A dictionary containing the MNIST dataset with keys "data" (image data as a numpy array)
                    and "target" (labels as a numpy array or list).
    The function visualizes the first 10 images in the dataset in a single row, showing each image and its label.
    """
    X, y = mnist["data"], mnist["target"].astype(int)  # Convert labels to int

    # Display 10 sample images
    fig, axes = plt.subplots(1, 10, figsize=(12, 1.5))
    for i, ax in enumerate(axes):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ax.set_title(str(y[i]), fontsize=12)
        ax.axis("off")
    plt.suptitle("Sample images from the MNIST dataset", fontsize=16, y=1.05)
    plt.show()

def plot_distance_comparison(point_A, point_B):
    """
    Visualizes and compares the Euclidean and Manhattan distances between two 2D points.
    This function creates a matplotlib plot showing two points in 2D space, the straight-line (Euclidean) distance between them,
    and the L-shaped (Manhattan) path. It annotates the plot with the computed distances and provides explanatory text.
    Args:
        point_A (tuple or list of float): Coordinates of the first point (x, y).
        point_B (tuple or list of float): Coordinates of the second point (x, y).
    Returns:
        None: Displays a matplotlib plot comparing the two distance metrics.
    Example:
        >>> plot_distance_comparison((1, 1), (6, 5))
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot the points
    ax.scatter(*point_A, color='blue', s=100, zorder=5, label='Point A (1, 1)')
    ax.scatter(*point_B, color='red', s=100, zorder=5, label='Point B (6, 5)')

    # Euclidean distance (straight line)
    ax.plot([point_A[0], point_B[0]], [point_A[1], point_B[1]], 
            color='blue', linewidth=3, label='Euclidean Distance', alpha=0.8)

    # Manhattan distance (L-shaped path)
    # First go horizontally, then vertically
    ax.plot([point_A[0], point_B[0]], [point_A[1], point_A[1]], 
            color='orange', linewidth=3, alpha=0.8)
    ax.plot([point_B[0], point_B[0]], [point_A[1], point_B[1]], 
            color='orange', linewidth=3, alpha=0.8, label='Manhattan Distance')

    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Calculate and display distances
    euclidean_dist = np.sqrt((point_B[0] - point_A[0])**2 + (point_B[1] - point_A[1])**2)
    manhattan_dist = abs(point_B[0] - point_A[0]) + abs(point_B[1] - point_A[1])

    # Calculate the midpoints for annotations
    mid_x = (point_A[0] + point_B[0]) / 2
    mid_y = (point_A[1] + point_B[1]) / 2

    # Add distance annotations
    ax.text(mid_x, mid_y, f'Euclidean: {euclidean_dist:.2f}', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            fontsize=12, ha='center')

    ax.text(point_B[0], point_A[1], f'Manhattan: {manhattan_dist:.2f}', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
            fontsize=12, ha='center')

    # Labels and title
    ax.set_xlabel('X Coordinate', fontsize=14)
    ax.set_ylabel('Y Coordinate', fontsize=14)
    ax.set_title('Euclidean vs Manhattan Distance Visualization', fontsize=16, fontweight='bold')

    # Get min and max for axis limits
    min_x = min(point_A[0], point_B[0]) - 2
    max_x = max(point_A[0], point_B[0]) + 2
    min_y = min(point_A[1], point_B[1]) - 2
    max_y = max(point_A[1], point_B[1]) + 2

    # Set axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Add legend
    ax.legend(loc='upper left', fontsize=12)

    # Adjust layout to make room for text at bottom
    plt.subplots_adjust(bottom=0.15)

    # Add explanatory text (moved higher)
    plt.figtext(0.15, 0.05, 
            "Euclidean: Straight-line distance ('as the crow flies')\n"
            "Manhattan: Sum of horizontal and vertical distances ('city block')", 
            fontsize=10, style='italic')

    plt.show()

def plot_projection(X, y, title, xlabel, ylabel, dataset_name="mnist", cmap=None, handles=None, legend_title=None):
    """
    Plots a 2D projection of data points with optional coloring by label and custom legends.
    Args:
        X (np.ndarray): 2D array of shape (n_samples, 2) representing the projected data.
        y (np.ndarray or None): Array of labels for coloring the points. If None, points are not colored by label.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        dataset_name (str, optional): Name of the dataset, used to determine legend labels. Defaults to "mnist".
        cmap (matplotlib.colors.Colormap, optional): Colormap to use for coloring points. If None, a default is chosen based on the dataset.
        handles (list, optional): Custom legend handles. If None, handles are generated based on the dataset.
        legend_title (str, optional): Title for the legend. If None, a default is chosen based on the dataset.
    Returns:
        None: Displays the plot.
    """

    # Define the Colormap based on the labels
    unique_labels = np.unique(y)
    if cmap is None:
        if len(unique_labels) == 10:
            cmap = plt.get_cmap('tab10', len(unique_labels))
        else:
            cmap = plt.get_cmap("Set1", len(unique_labels))
    
    # Define the size of the points based on the number of samples
    size = 3 if len(X) > 1000 else 30 

    # Define the legend handles for the used datasets
    if dataset_name == "mnist":
        handles = handles = [
                plt.Line2D([], [], marker="o", linestyle="", color=cmap(i), label=str(i))
                for i in range(10)
                ]
        legend_title = "Digit"

    elif dataset_name == "breast_cancer":
        outcome_names = ['Malignant', 'Benign']
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", color=cmap(i), label=outcome_name)
            for i, outcome_name in enumerate(outcome_names)
        ]
        legend_title = "Outcome"

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    if y is None:
        plt.scatter(X[:, 0], X[:, 1], s=size, alpha=0.5, cmap=cmap)
    else:
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=size, cmap=cmap, alpha=0.5)
        if handles is not None:
            plt.legend(handles=handles, title=legend_title, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_projection_grid(embeddings, dataset_name="mnist", figsize=(14, 10), legend_subplot=None):
    """
    Plots multiple 2D projections (e.g., from dimensionality reduction) in a grid layout for visual comparison.

    Args: 
       embeddings (List[Tuple[np.ndarray, np.ndarray, str]]): 
            A list of tuples, each containing (X, y, title) where:
                - X (np.ndarray): 2D array of projected data points (shape: [n_samples, 2]).
                - y (np.ndarray or None): Array of labels for coloring points, or None for unlabeled data.
                - title (str): Title for the subplot.
        dataset_name (str, optional): 
            Name of the dataset, used to determine legend labels and colors. 
            Supported: 'mnist', 'breast_cancer', 'heart_disease'. Defaults to "mnist".
        figsize (Tuple[int, int], optional): 
            Size of the overall figure as (width, height). Defaults to (14, 10).
        legend_subplot (int or None, optional): 
            Index (1-based) of the subplot to display the legend. If None, legend is shown on the first subplot.
    Returns:
        None: Displays the grid of scatter plots with appropriate legends and titles.
    """
    
    n_plots = len(embeddings)
    
    # Calculate grid dimensions (squarest possible arrangement)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array if necessary
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Determine consistent colormap across all plots
    all_labels = np.concatenate([y for _, y, _ in embeddings if y is not None])
    unique_labels = np.unique(all_labels) if len(all_labels) > 0 else []
    
    if len(unique_labels) == 10:  # MNIST case
        cmap = plt.get_cmap('tab10', 10)
    else:  # Binary or other cases
        cmap = plt.get_cmap("Set1", len(unique_labels)) if len(unique_labels) > 0 else 'viridis'
    
    # Create legend handles once
    if dataset_name == "mnist":
        handles = [plt.Line2D([], [], marker="o", linestyle="", 
                             color=cmap(i), label=str(i)) for i in range(10)]
        legend_title = "Digit"
    elif dataset_name == "breast_cancer":
        outcome_names = ['Malignant', 'Benign']
        handles = [plt.Line2D([], [], marker="o", linestyle="", 
                             color=cmap(i), label=name) for i, name in enumerate(outcome_names)]
        legend_title = "Outcome"
    elif dataset_name == "heart_disease":
        outcome_names = ['No Heart Disease', 'Heart Disease']
        handles = [plt.Line2D([], [], marker="o", linestyle="", 
                             color=cmap(i), label=name) for i, name in enumerate(outcome_names)]
        legend_title = "Heart Disease"
    else:
        handles = None
        legend_title = None
    
    # Plot each projection
    for idx, ((X, y, title), ax) in enumerate(zip(embeddings, axes), 1):
        # Set point size based on sample count
        size = 3 if len(X) > 1000 else 30
        
        if y is None:
            sc = ax.scatter(X[:, 0], X[:, 1], s=size, alpha=0.5, cmap=cmap)
        else:
            sc = ax.scatter(X[:, 0], X[:, 1], c=y, s=size, cmap=cmap, alpha=0.5)
        
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Add legend (either on specified subplot or first one by default)
        if (legend_subplot == idx or (legend_subplot is None and idx == 1)) and handles is not None:
            ax.legend(handles=handles, title=legend_title)
    
    # Hide empty subplots if any
    for ax in axes[n_plots:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()

def visualize_features(data, features, outcome_col, outcome_names, title="Feature Distributions"):
    """
    Visualizes the distributions of specified features in the dataset, colored by outcome.
    Args:
        data (pd.DataFrame): The input DataFrame containing the features and outcome column.
        features (list of str): List of feature column names to visualize (expects exactly 3 features).
        outcome_col (str): The name of the column in `data` representing the outcome or class label.
        outcome_names (list of str): List of outcome names to use in the legend.
        title (str, optional): Title for the legend and the overall plot. Defaults to "Feature Distributions".
    Returns:
        None: This function displays the plots and does not return any value.
    """
    plt.figure(figsize=(15, 4))

    for i, feat in enumerate(features):
        plt.subplot(1, 3, i+1)
        sns.histplot(data=data, x=feat, hue=outcome_col, palette='Set1', bins=30, alpha=0.7, 
                    legend=(i==2), element="step")
        plt.title(feat)
        plt.xlabel(feat)
        plt.ylabel('Count')
        if i == 2:
            plt.legend(labels=outcome_names, title=title)

    plt.tight_layout()
    plt.suptitle("Feature Distributions", fontsize=16, y=1.05)
    plt.show()

def plot_similarity_matrices(X, y, datatype='continuous', suptitle=""):
    """
    Plots similarity (distance) matrices for selected samples from each category in the dataset.
    Depending on the datatype, computes and visualizes different distance/similarity matrices
    (Euclidean, Manhattan, Simple Matching, Jaccard, Hamming, or Gower) for a subset of samples
    (default: 5 per category) from the input data. The resulting matrices are displayed as heatmaps.
    Args:
        X (np.ndarray or pd.DataFrame): The input data matrix of shape (n_samples, n_features).
        y (np.ndarray or pd.Series or None): The labels or categories for each sample. If None,
            all samples are treated as belonging to a single category.
        datatype (str, optional): The type of data. Must be one of 'continuous', 'categorical', or 'mixed'.
            Determines which similarity/distance metrics are computed. Defaults to 'continuous'.
        suptitle (str, optional): The overall title for the plot. Defaults to an empty string.
    Returns:
        None: This function displays the plots and does not return any value.
    Raises:
        ValueError: If `datatype` is not one of 'continuous', 'categorical', or 'mixed'.
    Notes:
        - For each unique category in `y`, up to 5 samples are selected for visualization.
        - Requires external functions: `pairwise_distances`, `simple_matching_distance`,
          `pairwise_hamming_distance_similarity`, and `gower.gower_matrix`.
        - Uses matplotlib for plotting.
    """
    
    if y is None:
        y = np.zeros(X.shape[0])  # Default to a single category if no labels are provided
    
    # Get unique categories and their first samples
    unique_categories = np.unique(y)
    n_categories = len(unique_categories)
    n_samples_per_category = 5  # Number of samples to select per category
    
    # Container for selected indices
    selected_indices = []
    
    # For each category, get the first n_samples_per_category samples
    for category in unique_categories:
        category_indices = np.where(y == category)[0][:n_samples_per_category]
        selected_indices.extend(category_indices)

    # Create labels for the plot
    labels = [f"C{cat} {i+1}" for cat in unique_categories for i in range(n_samples_per_category)]

    if datatype == 'continuous':
        # Compute the full distance matrices
        euclidean_dist = pairwise_distances(X, metric='euclidean')
        manhattan_dist = pairwise_distances(X, metric='manhattan')

        similarity_matrices = [euclidean_dist, manhattan_dist]
        titles = ["Euclidean Distance (5 for each category)", "Manhattan Distance (5 for each category)"]
    
    elif datatype == 'categorical':
        # Compute simple matching distance for categorical data
        smd = simple_matching_distance(X)
        jaccard_dist = pairwise_distances(X, metric='jaccard')
        hamming_dist = pairwise_hamming_distance_similarity(X)
        similarity_matrices = [smd, jaccard_dist, hamming_dist]

        titles = ["Simple Matching Distance (5 for each category)", "Jaccard Distance (5 for each category)", "Hamming Distance (5 for each category)"]

    elif datatype == 'mixed':
        # Compute Gower distance for mixed data types
        gower_dist = gower.gower_matrix(X)
        similarity_matrices = [gower_dist]

        titles = ["Gower Distance (5 for each category)"]


    similarity_selected = [mat[np.ix_(selected_indices, selected_indices)] for mat in similarity_matrices]


    # Plot the similarity (distance) matrices
    # Check if we have a single matrix or a list of matrices
    if len(similarity_selected) == 1:
        # For a single similarity matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        mat = similarity_selected[0]
        title = titles[0]
        
        im = ax.imshow(mat, cmap='viridis', interpolation='nearest')
        ax.set_xticks(range(len(selected_indices)))
        ax.set_yticks(range(len(selected_indices)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        # For multiple similarity matrices
        fig, axes = plt.subplots(1, len(similarity_selected), figsize=(12, 5))
        
        for ax, mat, title in zip(axes, similarity_selected, titles):
            im = ax.imshow(mat, cmap='viridis', interpolation='nearest')
            ax.set_xticks(range(len(selected_indices)))
            ax.set_yticks(range(len(selected_indices)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticklabels(labels)
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

###### Clustering

def simulate_clean_dataset(n_samples=300, n_features=2, centers=4, cluster_std=0.6):
    """
    Simulates and visualizes a clean synthetic dataset with Gaussian clusters.
    Generates a dataset using `make_blobs` with a specified number of samples, features, clusters, and cluster standard deviation.
    The function also visualizes the generated data in two subplots: one showing the true clusters and another showing the data without labels.
    Args:
        n_samples (int, optional): Number of samples to generate. Defaults to 300.
        n_features (int, optional): Number of features for each sample. Defaults to 2.
        centers (int, optional): Number of cluster centers to generate. Defaults to 4.
        cluster_std (float, optional): Standard deviation of the clusters. Defaults to 0.6.
    Returns:
        tuple:
            - X_clean (ndarray): Generated feature matrix of shape (n_samples, n_features).
            - y_true (ndarray): True cluster labels for each sample.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate a clean 2D dataset with 4 clusters
    X_clean, y_true = make_blobs(
        n_samples=n_samples, 
        centers=centers, 
        n_features=n_features,
        cluster_std=cluster_std,
        center_box=(-8.0, 8.0),
        random_state=43
    )

    # Create DataFrame for easier handling
    df_clean = pd.DataFrame(X_clean, columns=['Feature_1', 'Feature_2'])
    df_clean['true_cluster'] = y_true

    # Visualize the clean dataset
    plt.figure(figsize=(12, 5))

    # Plot 1: True clusters
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_clean[:, 0], X_clean[:, 1], c=y_true, cmap='viridis', alpha=0.7, s=50)
    plt.title('True Clusters (4 groups)', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Plot 2: All points (no color coding)
    plt.subplot(1, 2, 2)
    plt.scatter(X_clean[:, 0], X_clean[:, 1], alpha=0.7, s=50, color='gray')
    plt.title('Dataset without Labels\n(What we see in practice)', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return X_clean, y_true

def visualize_kmeans_steps(X_clean, y_true):
    """
    Visualizes the step-by-step process of the K-Means clustering algorithm on a 2D dataset.
    This function manually implements the K-Means algorithm and displays each major step in the clustering process,
    including centroid initialization, point assignment, centroid updates, and final clustering result. It also
    compares the final clustering with the ground truth labels.
    Args:
        X_clean (np.ndarray): A 2D NumPy array of shape (n_samples, 2) containing the input data points to cluster.
        y_true (np.ndarray): A 1D NumPy array of shape (n_samples,) containing the ground truth cluster labels for comparison.
    Returns:
        None: This function displays matplotlib plots and does not return any value.
    Notes:
        - The function assumes there are 4 clusters in the data.
        - The visualization includes centroid movement and cluster assignments at each step.
        - Requires matplotlib and numpy to be imported.
    """
    # Manual K-Means implementation for step-by-step visualization
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def assign_clusters(X, centroids):
        clusters = []
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster = np.argmin(distances)
            clusters.append(cluster)
        return np.array(clusters)

    def update_centroids(X, clusters, k):
        centroids = []
        for i in range(k):
            cluster_points = X[clusters == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
            else:
                centroid = X[np.random.randint(0, len(X))]  # Handle empty clusters
            centroids.append(centroid)
        return np.array(centroids)

    # Initialize K=4 random centroids (we know there are 4 true clusters)
    np.random.seed(40)
    k = 4
    initial_centroids = X_clean[np.random.choice(X_clean.shape[0], k, replace=False)]

    # Colors for visualization
    colors = ['red', 'blue', 'green', 'orange']

    # Step-by-step K-Means
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    centroids = initial_centroids.copy()
    all_centroids_history = [centroids.copy()]  # Track centroid movement

    for iteration in range(6):
        
        if iteration == 0:
            # Initial state
            axes[iteration].scatter(X_clean[:, 0], X_clean[:, 1], c='lightgray', s=40, alpha=0.6)
            axes[iteration].scatter(centroids[:, 0], centroids[:, 1], c=colors, s=300, marker='X', 
                                edgecolors='black', linewidth=3, label='Initial Centroids')
            axes[iteration].set_title('Step 1: Initialize Random Centroids', fontsize=14, fontweight='bold')
            axes[iteration].legend(fontsize=12)
            
        elif iteration <= 3:
            # Assign points to clusters
            clusters = assign_clusters(X_clean, centroids)
            
            # Plot points colored by cluster assignment
            for i in range(k):
                mask = clusters == i
                if np.any(mask):
                    axes[iteration].scatter(X_clean[mask, 0], X_clean[mask, 1], c=colors[i], s=40, alpha=0.7, 
                                        label=f'Cluster {i}')
            
            # Plot current centroids
            axes[iteration].scatter(centroids[:, 0], centroids[:, 1], c=colors, s=300, marker='X', 
                                edgecolors='black', linewidth=3)
            
            # Show centroid movement with arrows (except first iteration)
            if iteration > 1 and len(all_centroids_history) > 1:
                old_centroids = all_centroids_history[-2]
                for i in range(k):
                    axes[iteration].annotate('', xy=centroids[i], xytext=old_centroids[i],
                                        arrowprops=dict(lw=2, color='gray', alpha=0.7, headwidth=20, headlength=20))
            
            axes[iteration].set_title(f'Step {iteration + 1}: Assign Points & Update Centroids', 
                                    fontsize=14, fontweight='bold')
            
            # Update centroids for next iteration
            if iteration < 3:
                centroids = update_centroids(X_clean, clusters, k)
                all_centroids_history.append(centroids.copy())
                
        elif iteration == 4:
            # Final K-means result
            final_clusters = assign_clusters(X_clean, centroids)
            for i in range(k):
                mask = final_clusters == i
                if np.any(mask):
                    axes[iteration].scatter(X_clean[mask, 0], X_clean[mask, 1], c=colors[i], s=40, alpha=0.7)
            
            axes[iteration].scatter(centroids[:, 0], centroids[:, 1], c=colors, s=300, marker='X', 
                                edgecolors='black', linewidth=3)
            axes[iteration].set_title('Final K-Means Result\n(Converged)', fontsize=14, fontweight='bold')
            
        else:
            # Show true clusters for comparison
            scatter = axes[iteration].scatter(X_clean[:, 0], X_clean[:, 1], c=y_true, cmap='viridis', s=40, alpha=0.8)
            axes[iteration].set_title('True Clusters\n(Ground Truth)', fontsize=14, fontweight='bold')
        
        # Formatting
        axes[iteration].set_xlabel('Feature 1', fontsize=12)
        axes[iteration].set_ylabel('Feature 2', fontsize=12)
        axes[iteration].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('K-Means Clustering: Step-by-Step Algorithm', fontsize=18, y=1.02)
    plt.show()

def plot_kmeans_clusters(X, labels, centroids=None):
    """
    Plots the results of K-Means clustering, displaying data points colored by cluster and optional centroids.

    Args:
        X (np.ndarray): 2D array of shape (n_samples, 2) containing the data points to plot.
        labels (np.ndarray or list): Cluster labels for each data point.
        centroids (np.ndarray, optional): 2D array of shape (n_clusters, 2) containing the coordinates of cluster centroids. Defaults to None.

    Returns:
        None: This function displays a matplotlib plot and does not return any value.
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=300, 
                edgecolors='black', linewidth=2, label='Centroids')
    plt.title('K-Means Clustering Result', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_silhouette_scores(X, labels, centroids, silhouette_scores, silhouette_samples):
    """
    Plots the silhouette scores for clustered data, including both the silhouette plot and a scatter plot of the data colored by silhouette score.
    Args:
        X (np.ndarray): The input data array of shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels for each data point.
        centroids (np.ndarray): Coordinates of cluster centroids, shape (n_clusters, n_features).
        silhouette_scores (float): The average silhouette score for all samples.
        silhouette_samples (np.ndarray): Silhouette score for each sample.
    Returns:
        None: This function displays the plots and does not return any value.
    """
    # Create silhouette plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Silhouette plot
    y_lower = 10
    num_labels = len(np.unique(labels))
    colors = cm.viridis(np.linspace(0, 1, num_labels))

    for i in range(num_labels):
        # Get silhouette scores for cluster i
        cluster_silhouette_values = silhouette_samples[labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                        0, cluster_silhouette_values,
                        facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        
        # Label clusters
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster label')
    ax1.set_title(f'Silhouette Plot\nAverage Score: {silhouette_scores:.3f}', fontweight='bold')

    # Add vertical line for average score
    ax1.axvline(x=silhouette_scores, color="red", linestyle="--", 
            label=f'Average: {silhouette_scores:.3f}')
    ax1.legend()

    # Plot 2: Clustered data with silhouette coloring
    scatter = ax2.scatter(X[:, 0], X[:, 1], c=silhouette_samples, 
                        cmap='RdYlBu', alpha=0.7, s=50)
    ax2.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=300, 
            edgecolors='black', linewidth=2, label='Centroids')
    ax2.set_title('Data Points Colored by\nSilhouette Score', fontweight='bold')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add colorbar
    plt.colorbar(scatter, ax=ax2, label='Silhouette Score')

    plt.tight_layout()
    plt.show()

def plot_elbow_method(k_range, wcss):
    """
    Plots the Elbow Method graph to help determine the optimal number of clusters for K-Means clustering.
    Args:
        k_range (iterable): A sequence of values representing the number of clusters (k) to evaluate.
        wcss (iterable): A sequence of Within-Cluster Sum of Squares (WCSS) values corresponding to each k in k_range.
    Returns:
        None: Displays the plot and does not return any value.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o', linewidth=3, markersize=10, color='blue')
    plt.title('Elbow Method for Optimal K', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def simulate_dataset_with_outliers(X_clean, y_true, n_outliers=30, outlier_range=(-30, 30)):
    """
    Adds synthetic outliers to a clean dataset and visualizes the result.
    This function appends a specified number of outlier points, generated uniformly at random within a given range,
    to the provided clean dataset. It also creates new labels for the outliers and visualizes both the original and 
    the augmented datasets side by side.
    Args:
        X_clean (np.ndarray): The original clean dataset of shape (n_samples, 2).
        y_true (np.ndarray): The true cluster labels for the clean dataset.
        n_outliers (int, optional): Number of outlier points to add. Defaults to 30.
        outlier_range (tuple, optional): The (min, max) range for generating outlier coordinates. Defaults to (-30, 30).
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - X_with_outliers: The augmented dataset including outliers, shape (n_samples + n_outliers, 2).
            - y_with_outliers: The corresponding labels, with -1 assigned to outliers.
    """
    # Add outliers to the clean dataset
    np.random.seed(42)  # Different seed for outliers

    # Generate outliers at random positions far from main clusters
    X_outliers = np.random.uniform(outlier_range[0], outlier_range[1], (n_outliers, 2))

    # Combine clean data with outliers
    X_with_outliers = np.vstack([X_clean, X_outliers])

    # Create labels (original clusters + outlier label)
    y_with_outliers = np.hstack([y_true, np.full(n_outliers, -1)])  # -1 for outliers

    # Visualize the dataset with outliers
    plt.figure(figsize=(12, 5))

    # Plot 1: Clean dataset
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(X_clean[:, 0], X_clean[:, 1], c=y_true, cmap='viridis', alpha=0.7, s=50)
    plt.title('Original Clean Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)

    # Plot 2: Dataset with outliers
    plt.subplot(1, 2, 2)
    # Plot main clusters
    scatter2 = plt.scatter(X_clean[:, 0], X_clean[:, 1], c=y_true, cmap='viridis', alpha=0.7, s=50, label='Main clusters')
    # Plot outliers
    plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', marker='x', s=100, alpha=0.8, label='Outliers', linewidths=3)
    plt.title('Dataset with Outliers Added', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    return X_with_outliers, y_with_outliers

def compare_kmeans_kmedoids(X, kmeans_labels, kmeans_centroids, kmedoids_labels, medoids):
    """
    Visualizes and compares the clustering results of K-Means and K-Medoids algorithms.
    This function creates a side-by-side plot of the clustering assignments and cluster centers/medoids
    for both K-Means and K-Medoids on the same dataset. It helps to visually assess the differences
    in clustering behavior, especially in the presence of outliers.
    Args:
        X (np.ndarray): The input data array of shape (n_samples, 2).
        kmeans_labels (np.ndarray): Cluster labels assigned by K-Means for each sample.
        kmeans_centroids (np.ndarray): Coordinates of K-Means cluster centroids of shape (n_clusters, 2).
        kmedoids_labels (np.ndarray): Cluster labels assigned by K-Medoids for each sample.
        medoids (np.ndarray): Coordinates of K-Medoids cluster centers (medoids) of shape (n_clusters, 2).
    Returns:
        None: This function displays the plots and does not return any value.
    """
    # Visualize K-Medoids results
    plt.figure(figsize=(12, 5))

    # Plot 1: K-Means on data with outliers (for comparison)
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, 
                        cmap='viridis', alpha=0.7, s=60)
    plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], 
            c='red', marker='X', s=100, edgecolors='black', linewidth=2, label='Centroids')
    
    plt.title('K-Means', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: K-Medoids results
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(X[:, 0], X[:, 1], c=kmedoids_labels, 
                        cmap='viridis', alpha=0.7, s=60)
    plt.scatter(medoids[:, 0], medoids[:, 1], 
            c='red', marker='s', s=100, edgecolors='black', linewidth=2, label='Medoids')
    plt.title('K-Medoids', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_hierarchical_clustering(X, linkage_matrix, cluster_labels, title):
    """
    Visualizes hierarchical clustering results using a dendrogram and a scatter plot.
    This function creates a figure with two subplots:
    - The left subplot displays the dendrogram of the hierarchical clustering, highlighting the cut threshold for the specified number of clusters.
    - The right subplot shows a scatter plot of the clustered data points, colored by cluster assignment.
    Args:
        X (np.ndarray): The data matrix of shape (n_samples, 2), where each row represents a data point with two features.
        linkage_matrix (np.ndarray): The linkage matrix produced by hierarchical clustering algorithms (e.g., scipy's `linkage` function).
        cluster_labels (np.ndarray): Array of cluster labels for each data point, typically obtained from `fcluster`.
        title (str): The overall title for the figure.
    Returns:
        None: This function displays the plots and does not return any value.
    """
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    num_clusters = len(np.unique(cluster_labels))
    cluster_colors = ['red', 'blue', 'green', 'orange']
    
    # Left: Average Linkage Dendrogram
    ax1 = axes[0]
    threshold = linkage_matrix[-(num_clusters-1), 2]  # For 4 clusters

    dend = dendrogram(linkage_matrix, 
                            #truncate_mode='level',
                            p=10,  # Show more nodes
                            leaf_rotation=90,
                            leaf_font_size=6,
                            show_leaf_counts=True,  # Show cluster sizes
                            color_threshold=threshold,
                            ax=ax1)

    ax1.axhline(y=threshold, color='blue', linestyle='--', linewidth=3,
            label=f'Cut for {num_clusters} clusters')
    ax1.set_title('Dendrogram', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Data Samples', fontsize=12)
    ax1.set_ylabel('Distance (Average Linkage)', fontsize=12)
    ax1.legend()
    ax1.set_xticks([])

    # Right: Average Linkage Clustering Visualization
    ax2 = axes[1]

    for cluster_id in range(num_clusters):
        mask = cluster_labels == cluster_id
        cluster_size = np.sum(mask)
        ax2.scatter(X[mask, 0], X[mask, 1], 
                c=cluster_colors[cluster_id], s=50, alpha=0.7,
                label=f'Cluster {cluster_id} (n={cluster_size})')

    ax2.set_title('Clustering Results', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.show()

def generate_moons_data(n_samples=300, noise=0.2, random_state=42, plot=True):
    """
    Generate a synthetic dataset of two interleaving half circles (moons).
    
    Parameters:
    - n_samples: Number of samples to generate.
    - noise: Standard deviation of Gaussian noise added to the data.
    - random_state: Seed for reproducibility.
    
    Returns:
    - X: Feature matrix of shape (n_samples, 2).
    - y: Labels of shape (n_samples,).
    """
    # Create the moons dataset
    X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=42)

    if plot:
        # Plot the moons data
        plt.figure(figsize=(8, 6))
        colors = ['red', 'blue']
        labels = ['Class 0', 'Class 1']

        for i in range(2):
            mask = y_moons == i
            plt.scatter(X_moons[mask, 0], X_moons[mask, 1], 
                    c=colors[i], alpha=0.7, s=50, label=labels[i])

        plt.xlabel('X₁', fontsize=14)
        plt.ylabel('X₂', fontsize=14)
        plt.title('Moons Dataset: A Non-Linear Classification Problem', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"Dataset shape: {X_moons.shape}")
        print(f"Class distribution: {np.bincount(y_moons)}")

    return X_moons, y_moons

def generate_non_spherical_data():
    """
    Generates and visualizes two non-spherical synthetic datasets (moons and circles) and applies multiple clustering algorithms.
    The function creates two datasets using `make_moons` and `make_circles`, then visualizes the true clusters and the results of K-Means, K-Medoids, and Hierarchical clustering for each dataset in a 2x4 subplot grid.
    Returns:
        tuple: A tuple containing:
            - X_moons (ndarray): Feature matrix for the moons dataset.
            - y_moons (ndarray): True labels for the moons dataset.
            - X_circles (ndarray): Feature matrix for the circles dataset.
            - y_circles (ndarray): True labels for the circles dataset.
    """
    # Generate challenging datasets
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    cmap = plt.get_cmap('Set1')

    # Dataset 1: Moons
    X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)

    # Dataset 2: Circles  
    X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)

    datasets = [
        (X_moons, y_moons, "Moons"),
        (X_circles, y_circles, "Circles")
    ]

    for row, (X, y_true, name) in enumerate(datasets):
        
        # Original data
        axes[row, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap=cmap, alpha=0.7)
        axes[row, 0].set_title(f'{name} Dataset\n(True Clusters)', fontweight='bold')
        axes[row, 0].grid(True, alpha=0.3)
        
        # K-Means
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        axes[row, 1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap=cmap, alpha=0.7)
        axes[row, 1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                            c='red', marker='X', s=200, edgecolors='black', linewidth=2)
        axes[row, 1].set_title('K-Means Result', fontweight='bold')
        axes[row, 1].grid(True, alpha=0.3)
        
        # K-Medoids
        euclidean_dist = pairwise_distances(X, metric='euclidean')
        kmed = kmedoids.KMedoids(n_clusters=2, random_state=42)
        kmed.fit(euclidean_dist)
        medoids = X[kmed.medoid_indices_]
        axes[row, 2].scatter(X[:, 0], X[:, 1], c=kmed.labels_, cmap=cmap, alpha=0.7)
        axes[row, 2].scatter(medoids[:, 0], medoids[:, 1], 
                            c='red', marker='s', s=200, edgecolors='black', linewidth=2)
        axes[row, 2].set_title('K-Medoids Result', fontweight='bold')
        axes[row, 2].grid(True, alpha=0.3)
        
        # Hierarchical Clustering
        linkage_matrix = linkage(X, method='average')
        hierarchical_labels = fcluster(linkage_matrix, 2, criterion='maxclust') - 1
        axes[row, 3].scatter(X[:, 0], X[:, 1], c=hierarchical_labels, cmap=cmap, alpha=0.7)
        axes[row, 3].set_title('Hierarchical Result', fontweight='bold')
        axes[row, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return X_moons, y_moons, X_circles, y_circles

def plot_k_distance_graph(X, k=4, dataset_name="Dataset"):
    """
    Plots the k-distance graph to assist in determining the optimal epsilon value for DBSCAN clustering.
    The function computes the distance to the k-th nearest neighbor for each point in the dataset,
    sorts these distances in descending order, and plots them. The "elbow" in the resulting plot
    can be used as a heuristic to select the epsilon parameter for DBSCAN.
    Args:
        X (array-like): The input dataset of shape (n_samples, n_features).
        k (int, optional): The k-th nearest neighbor to consider (default is 4). 
            Typically, k = minPts - 1 for DBSCAN.
        dataset_name (str, optional): Name of the dataset to display in the plot title (default is "Dataset").
    Returns:
        np.ndarray: Array of sorted k-th nearest neighbor distances for each point in the dataset.
    """
    
    # Calculate k-distance for each point
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    # Get k-th distance (last column) and sort in descending order
    k_distances = distances[:, k-1]
    k_distances = np.sort(k_distances)[::-1]
    
    # Plot k-distance graph
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(k_distances)), k_distances, 'b-', linewidth=2, alpha=0.7)
    plt.xlabel('Points sorted by distance', fontsize=12)
    plt.ylabel(f'{k}-distance', fontsize=12)
    plt.title(f'{k}-Distance Graph for {dataset_name}\n(Look for the "elbow" to find optimal ε)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return k_distances

def create_datasets():
    """
    Generates a dictionary of synthetic datasets for clustering demonstrations.
    This helper function creates a variety of 2D datasets with different characteristics,
    such as simple blobs, outliers, noise, varying densities, circles, and moons.
    It is intended for internal use and is not used directly in the notebooks.
    Returns:
        dict: A dictionary where keys are descriptive dataset names (str) and values are
            NumPy arrays of shape (n_samples, 2) representing the data points.
    Note:
        This function is a utility for generating example datasets and is not called
        directly from the notebooks.
    """
    datasets = {}
    
    # 1. Simple blobs (baseline)
    X_blobs, _ = make_blobs(n_samples=300, centers=4, n_features=2, 
                           random_state=42, cluster_std=1.0)
    datasets['Simple Blobs'] = X_blobs
    
    # 2. Blobs with extreme outliers
    X_blobs_out, _ = make_blobs(n_samples=280, centers=4, n_features=2, 
                               random_state=42, cluster_std=0.8)
    # Add extreme outliers
    outliers = np.random.uniform(-30, 30, (20, 2))
    X_outliers = np.vstack([X_blobs_out, outliers])
    datasets['Extreme Outliers'] = X_outliers
    
    # 3. Very noisy data
    X_blobs_noisy, _ = make_blobs(n_samples=200, centers=3, n_features=2, 
                                 random_state=42, cluster_std=0.6)
    # Add lots of noise
    noise = np.random.uniform(X_blobs_noisy.min()-2, X_blobs_noisy.max()+2, (100, 2))
    X_noisy = np.vstack([X_blobs_noisy, noise])
    datasets['Noisy Data'] = X_noisy
    
    # 4. Varying densities
    # Dense cluster
    dense_cluster = np.random.multivariate_normal([2, 2], [[0.1, 0], [0, 0.1]], 100)
    # Medium density cluster  
    medium_cluster = np.random.multivariate_normal([-2, 2], [[0.5, 0], [0, 0.5]], 80)
    # Sparse cluster
    sparse_cluster = np.random.multivariate_normal([0, -3], [[1.5, 0], [0, 1.5]], 60)
    X_varying = np.vstack([dense_cluster, medium_cluster, sparse_cluster])
    datasets['Varying Densities'] = X_varying
    
    # 5. Circles
    X_circles, _ = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
    datasets['Circles'] = X_circles
    
    # 6. Moons
    X_moons, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    datasets['Moons'] = X_moons
    
    return datasets

def apply_clustering_algorithms(X, dataset_name):
    """
    Apply multiple clustering algorithms to a given dataset.
    This helper function is not used in the notebooks. It applies K-Means, K-Medoids, Hierarchical Clustering (Ward), 
    and DBSCAN to the input data, using dataset-specific parameters for the number of clusters and DBSCAN's `eps`.
    Args:
        X (array-like or ndarray): Feature matrix of shape (n_samples, n_features) representing the dataset to cluster.
        dataset_name (str): Name of the dataset, used to determine clustering parameters.
    Returns:
        dict: A dictionary mapping algorithm names to their predicted cluster labels (ndarray of shape (n_samples,)).
    """
    results = {}
    
    # Determine number of clusters (except for DBSCAN)
    if dataset_name in ['Circles', 'Moons']:
        n_clusters = 2
    elif dataset_name in ['Simple Blobs', 'Extreme Outliers']:
        n_clusters = 4
    elif dataset_name == 'Noisy Data':
        n_clusters = 3
    else:
        n_clusters = 3
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    results['K-Means'] = kmeans.fit_predict(X)
    
    # K-Medoids
    kmed = kmedoids.KMedoids(n_clusters=n_clusters, random_state=42)
    euclidean_dist = pairwise_distances(X, metric='euclidean')
    kmed.fit(euclidean_dist)
    results['K-Medoids'] = kmed.labels_
    
    # Hierarchical Clustering (Ward)
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    results['Hierarchical\n(Ward)'] = ward.fit_predict(X)
    
    # DBSCAN - parameter selection based on dataset
    eps_params = {
        'Simple Blobs': 0.8,
        'Extreme Outliers': 0.8,
        'Noisy Data': 0.6,
        'Varying Densities': 0.8,
        'Circles': 0.2,
        'Moons': 0.2
    }
    
    dbscan = DBSCAN(eps=eps_params[dataset_name], min_samples=5)
    results['DBSCAN'] = dbscan.fit_predict(X)
    
    return results

def plot_clustering_comparison():
    """
    Visualizes and compares the results of multiple clustering algorithms across different synthetic datasets.
    This function generates a grid of scatter plots, where each row corresponds to a different dataset and each column
    corresponds to a different clustering algorithm (K-Means, K-Medoids, Hierarchical (Ward), and DBSCAN). Each subplot
    displays the clustering assignments, with different colors representing different clusters. For DBSCAN, noise points
    are marked in black with 'x' markers.
    The function assumes the existence of `create_datasets()` to generate datasets and `apply_clustering_algorithms()`
    to apply clustering algorithms to the data.
    Returns:
        None: Displays the comparison plot using matplotlib.
    """
    np.random.seed(42)
    
    datasets = create_datasets()
    algorithms = ['K-Means', 'K-Medoids', 'Hierarchical\n(Ward)', 'DBSCAN']
    
    # Create the plot
    fig, axes = plt.subplots(len(datasets), len(algorithms), 
                            figsize=(16, 18))
    
    # Color maps
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    
    for row, (dataset_name, X) in enumerate(datasets.items()):
        # Apply all clustering algorithms
        results = apply_clustering_algorithms(X, dataset_name)
        
        for col, algorithm in enumerate(algorithms):
            ax = axes[row, col]
            labels = results[algorithm]
            
            # Handle DBSCAN noise points (labeled as -1)
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                if label == -1:  # Noise points in DBSCAN
                    mask = labels == label
                    ax.scatter(X[mask, 0], X[mask, 1], c='black', 
                              marker='x', s=20, alpha=0.6, label='Noise')
                else:
                    mask = labels == label
                    color_idx = label % len(colors)
                    ax.scatter(X[mask, 0], X[mask, 1], c=colors[color_idx], 
                              s=30, alpha=0.7)
            
            # Formatting
            ax.set_title(f'{algorithm}', fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add dataset name on the left
            if col == 0:
                ax.set_ylabel(dataset_name, fontweight='bold', fontsize=11, rotation=90)
    
    plt.tight_layout()
    plt.suptitle('Clustering Algorithm Comparison Across Different Dataset Types', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.95)
    plt.show()

def plot_2cluster_comparison(X, labels1, labels2, title1, title2, centroids1=None, centroids2=None):
    """
    Plots a side-by-side comparison of two clustering solutions on 2D data.
    This function visualizes two different cluster labelings of the same dataset in two subplots.
    Optionally, cluster centroids can be displayed. Supports both numeric and string cluster labels.
    Args:
        X (np.ndarray): 2D array of shape (n_samples, 2) containing the data points to plot.
        labels1 (array-like): Cluster labels for the first clustering solution (numeric or string).
        labels2 (array-like): Cluster labels for the second clustering solution (numeric or string).
        title1 (str): Title for the first subplot.
        title2 (str): Title for the second subplot.
        centroids1 (np.ndarray, optional): Array of shape (n_clusters1, 2) with centroids for the first clustering solution.
        centroids2 (np.ndarray, optional): Array of shape (n_clusters2, 2) with centroids for the second clustering solution.
    Returns:
        None: Displays the comparison plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Convert string labels to numeric for plotting
    def convert_labels_for_plotting(labels):
        if isinstance(labels[0], str):
            unique_labels = list(set(labels))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            return [label_map[label] for label in labels], unique_labels
        return labels, None
    
    # Convert labels if they are strings
    plot_labels1, unique_labels1 = convert_labels_for_plotting(labels1)
    plot_labels2, unique_labels2 = convert_labels_for_plotting(labels2)
    
    # Plot first clustering solution
    scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=plot_labels1, cmap='viridis', alpha=0.7, s=50)
    if centroids1 is not None:
        axes[0].scatter(centroids1[:, 0], centroids1[:, 1], c='red', marker='X', s=300, 
                       edgecolors='black', linewidth=2, label='Centroids')
    
    # Add legend for string labels
    if unique_labels1 is not None:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i/len(unique_labels1)), 
                             markersize=8, label=label) for i, label in enumerate(unique_labels1)]
        if centroids1 is not None:
            handles.append(plt.Line2D([0], [0], marker='X', color='red', markersize=12, 
                                    markeredgecolor='black', linewidth=0, label='Centroids'))
        axes[0].legend(handles=handles)
    elif centroids1 is not None:
        axes[0].legend()
    
    axes[0].set_title(title1, fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].grid(True, alpha=0.3)
    
    # Plot second clustering solution
    scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=plot_labels2, cmap='viridis', alpha=0.7, s=50)
    if centroids2 is not None:
        axes[1].scatter(centroids2[:, 0], centroids2[:, 1], c='red', marker='X', s=300, 
                       edgecolors='black', linewidth=2, label='Centroids')
    
    # Add legend for string labels
    if unique_labels2 is not None:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i/len(unique_labels2)), 
                             markersize=8, label=label) for i, label in enumerate(unique_labels2)]
        if centroids2 is not None:
            handles.append(plt.Line2D([0], [0], marker='X', color='red', markersize=12, 
                                    markeredgecolor='black', linewidth=0, label='Centroids'))
        axes[1].legend(handles=handles)
    elif centroids2 is not None:
        axes[1].legend()
    
    axes[1].set_title(title2, fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def gene_info(gene_data):
    """
    Displays basic information about a gene expression dataset and visualizes the distribution of cancer types.
    This function prints the number of samples (patients), features (genes), and the overall data size.
    It also generates a bar plot showing the distribution of cancer types present in the dataset.
    Args:
        gene_data (pandas.DataFrame): A DataFrame containing gene expression data with a column named 'Cancer_Type'
            indicating the cancer type for each sample.
    Returns:
        None: This function displays information and a plot but does not return any value.
    """
    # Basic dataset information
    print(f"Dataset Overview:")
    print(f"   - Samples (patients): {gene_data.shape[0]}")
    print(f"   - Features (genes): {gene_data.shape[1]}")
    print(f"   - Data size: {gene_data.shape[0]} × {gene_data.shape[1]}")

    # Create bar plot of cancer type distribution
    gene_labels = gene_data[['Cancer_Type']]
    label_counts = gene_labels.iloc[:, 0].value_counts()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(label_counts)), label_counts.values, 
                color=plt.cm.Set1(np.linspace(0, 1, len(label_counts))))

    plt.title('Cancer Type Distribution in Dataset', fontweight='bold', fontsize=14)
    plt.xlabel('Cancer Types', fontweight='bold')
    plt.ylabel('Number of Patients', fontweight='bold')
    plt.xticks(range(len(label_counts)), label_counts.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on top of bars
    for i, (cancer_type, count) in enumerate(label_counts.items()):
        plt.text(i, count + 1, str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_dbscan_grid(dataset, eps_values, min_samples_values):
    """
    Plots a grid of DBSCAN clustering results for different combinations of epsilon and min_samples parameters.
    Each subplot in the grid corresponds to a unique (eps, min_samples) pair, showing the clustering result
    on the provided dataset. Points are colored by their assigned cluster, and the epsilon neighborhood for
    each point is visualized as a circle.
    Args:
        dataset (np.ndarray): The input data to cluster, expected shape (n_samples, 2).
        eps_values (list or array-like): List of epsilon values to use for DBSCAN.
        min_samples_values (list or array-like): List of min_samples values to use for DBSCAN.
    Returns:
        None: This function displays the plot and does not return any value.
    """
    
    fig = plt.figure(figsize=(16, 20))
    plt.subplots_adjust(left=.02, right=.98, bottom=0.001, top=.96, wspace=.05,
                        hspace=0.25)


    plot_num = 1

    for i, min_samples in enumerate(min_samples_values):
        for j, eps in enumerate(eps_values):
            ax = fig.add_subplot( len(min_samples_values) , len(eps_values), plot_num)

            dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
            y_pred_2 = dbscan.fit_predict(dataset)

            colors = np.array(list(islice(cycle(['#df8efd', '#78c465', '#ff8e34',
                                                 '#f65e97', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred_2) + 1))))
            colors = np.append(colors, '#BECBD6')


            for point in dataset:
                circle1 = plt.Circle(point, eps, color='#666666', fill=False, zorder=0, alpha=0.3)
                ax.add_artist(circle1)

            ax.text(0, -0.03, 'Epsilon: {} \nMin_samples: {}'.format(eps, min_samples), transform=ax.transAxes, fontsize=16, va='top')
            ax.scatter(dataset[:, 0], dataset[:, 1], s=50, color=colors[y_pred_2], zorder=10, edgecolor='black', lw=0.5)


            plt.xticks(())
            plt.yticks(())

            # Calculate the limits for the plot
            x_min, x_max = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
            y_min, y_max = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            #plt.xlim(-14, 5)
            #plt.ylim(-12, 7)

            plot_num = plot_num + 1

    plt.show()

def compare_all_Algorithms(X_raw_scaled, X_pca, y_true, pca, labels_pca):
    """
    Compare multiple clustering algorithms on raw and PCA-reduced data, print performance metrics, and visualize results.
    This function applies K-Means, K-Medoids, Hierarchical Clustering (Ward), and DBSCAN to both the original scaled data and PCA-reduced data.
    It computes Adjusted Rand Index (ARI) and Silhouette scores for each algorithm, prints the results, and visualizes the clustering assignments
    in a 2x5 grid of scatter plots (using the first two principal components for visualization).
    Args:
        X_raw_scaled (np.ndarray): The original data, scaled (e.g., via StandardScaler), shape (n_samples, n_features).
        X_pca (np.ndarray): The PCA-reduced data, shape (n_samples, n_components).
        y_true (np.ndarray): Ground truth class labels, shape (n_samples,).
        pca (sklearn.decomposition.PCA): Fitted PCA object, used for explained variance ratios.
        labels_pca (np.ndarray): Cluster labels from K-Means on PCA data, shape (n_samples,).
    Returns:
        None: This function prints metrics and displays a matplotlib figure with clustering visualizations.
    """
    # First, apply algorithms on raw data
    results_raw = {}
    unique_true = np.unique(y_true)
    n_clusters = len(unique_true)
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # K-Means on raw data
    kmeans_raw = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    results_raw['K-Means'] = kmeans_raw.fit_predict(X_raw_scaled)

    # K-Medoids on raw data (using sample for computational efficiency)
    kmed_raw = kmedoids.KMedoids(n_clusters=n_clusters, random_state=42)
    euclidean_dist_raw = pairwise_distances(X_raw_scaled, metric='euclidean')
    kmed_raw.fit(euclidean_dist_raw)
    # Predict for all data points
    results_raw['K-Medoids'] = kmed_raw.labels_

    # Hierarchical Clustering on raw data
    ward_raw = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    results_raw['Hierarchical\n(Ward)'] = ward_raw.fit_predict(X_raw_scaled)

    # DBSCAN on raw data
    dbscan_raw = DBSCAN(eps=90, min_samples=3)
    results_raw['DBSCAN'] = dbscan_raw.fit_predict(X_raw_scaled)

    # Apply algorithms on PCA-reduced data
    results_pca = {}

    # K-Means (already computed)
    results_pca['K-Means'] = labels_pca

    # K-Medoids
    kmed_pca = kmedoids.KMedoids(n_clusters=n_clusters, random_state=42)
    euclidean_dist_pca = pairwise_distances(X_pca, metric='euclidean')
    kmed_pca.fit(euclidean_dist_pca)
    results_pca['K-Medoids'] = kmed_pca.labels_

    # Hierarchical Clustering (Ward)
    ward_pca = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    results_pca['Hierarchical\n(Ward)'] = ward_pca.fit_predict(X_pca)

    # DBSCAN - use a reasonable eps value for PCA space
    dbscan_pca = DBSCAN(eps=35, min_samples=3)
    results_pca['DBSCAN'] = dbscan_pca.fit_predict(X_pca)

    # Calculate and print performance metrics
    algorithms = ['K-Means', 'K-Medoids', 'Hierarchical\n(Ward)', 'DBSCAN']

    print("Performance on Raw Data (10,000 features):")
    for alg in algorithms:
        ari = adjusted_rand_score(y_true, results_raw[alg])
        # Handle DBSCAN noise points for silhouette score
        if -1 in results_raw[alg]:
            mask = results_raw[alg] != -1
            if mask.sum() > 1:
                sil = silhouette_score(X_raw_scaled[mask], results_raw[alg][mask])
            else:
                sil = -1
        else:
            sil = silhouette_score(X_raw_scaled, results_raw[alg])
        print(f"   {alg.replace(chr(10), ' '):<15}: ARI = {ari:.3f}, Silhouette = {sil:.3f}")

    print("\nPerformance on PCA Data (10 components):")
    for alg in algorithms:
        ari = adjusted_rand_score(y_true, results_pca[alg])
        # Handle DBSCAN noise points for silhouette score
        if -1 in results_pca[alg]:
            mask = results_pca[alg] != -1
            if mask.sum() > 1:
                sil = silhouette_score(X_pca[mask], results_pca[alg][mask])
            else:
                sil = -1
        else:
            sil = silhouette_score(X_pca, results_pca[alg])
        print(f"   {alg.replace(chr(10), ' '):<15}: ARI = {ari:.3f}, Silhouette = {sil:.3f}")

    # Create 2x5 visualization: upper row = raw data, lower row = PCA data
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))

    # Color schemes
    color_maps = [plt.cm.Set2, plt.cm.Set3, plt.cm.tab10, plt.cm.Paired]

    # Upper row: Raw data results (visualized in PCA space)
    axes[0, 0].set_title('True Cancer Types\n(Raw Data Visualization)', fontweight='bold', fontsize=12)
    for i, cancer_type in enumerate(unique_true):
        mask = y_true == cancer_type
        axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                        c=[colors[i]], alpha=0.7, s=30, label=cancer_type)
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot raw data clustering results
    for idx, algorithm in enumerate(algorithms):
        ax = axes[0, idx + 1]
        labels = results_raw[algorithm]
        
        # Calculate metrics for title
        ari = adjusted_rand_score(y_true, labels)
        if -1 in labels:
            mask = labels != -1
            if mask.sum() > 1:
                sil = silhouette_score(X_raw_scaled[mask], labels[mask])
            else:
                sil = -1
        else:
            sil = silhouette_score(X_raw_scaled, labels)
        
        # Plot clusters (visualized in PCA space)
        unique_labels = np.unique(labels)
        colors_alg = color_maps[idx](np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise points in DBSCAN
                mask = labels == label
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                        c='black', marker='x', s=20, alpha=0.6)
            else:
                mask = labels == label
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                        c=[colors_alg[i]], alpha=0.7, s=30)
        
        ax.set_title(f'{algorithm} (Raw Data)\nARI: {ari:.3f}, Sil: {sil:.3f}', 
                    fontweight='bold', fontsize=12)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.grid(True, alpha=0.3)

    # Lower row: PCA data results
    axes[1, 0].set_title('True Cancer Types\n(PCA Data Visualization)', fontweight='bold', fontsize=12)
    for i, cancer_type in enumerate(unique_true):
        mask = y_true == cancer_type
        axes[1, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                        c=[colors[i]], alpha=0.7, s=30, label=cancer_type)
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot PCA data clustering results
    for idx, algorithm in enumerate(algorithms):
        ax = axes[1, idx + 1]
        labels = results_pca[algorithm]
        
        # Calculate metrics for title
        ari = adjusted_rand_score(y_true, labels)
        if -1 in labels:
            mask = labels != -1
            if mask.sum() > 1:
                sil = silhouette_score(X_pca[mask], labels[mask])
            else:
                sil = -1
        else:
            sil = silhouette_score(X_pca, labels)
        
        # Plot clusters
        unique_labels = np.unique(labels)
        colors_alg = color_maps[idx](np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise points in DBSCAN
                mask = labels == label
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                        c='black', marker='x', s=20, alpha=0.6)
            else:
                mask = labels == label
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                        c=[colors_alg[i]], alpha=0.7, s=30)
        
        ax.set_title(f'{algorithm} (PCA Data)\nARI: {ari:.3f}, Sil: {sil:.3f}', 
                    fontweight='bold', fontsize=12)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

### Feature Selection

def generate_regression(n_samples=12, n_features=15, n_informative=1, coef=[2], noise=1.5, test_data=True, plot=True, corr=False):
    """
    Generate a regression dataset with specified parameters.
    
    Parameters:
    n_samples (int): Number of samples to generate.
    n_features (int): Total number of features.
    n_informative (int): Number of informative features.
    noise (float): Standard deviation of Gaussian noise added to the output.
    
    Returns:
    X (ndarray): Feature matrix.
    y (ndarray): Target vector.
    coef (ndarray): Coefficients of the informative features.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate features: all features are random noise
    X = np.random.randn(n_samples, n_features)
    if corr:
        # Add correlation structure
        for i in range(0, n_features, 5):  # Every 5 features are correlated
            end_idx = min(i+5, n_features)
            group_size = end_idx - i
            # Make features within group correlated
            base_feature = X[:, i]
            for j in range(i+1, end_idx):
                X[:, j] = 0.7 * base_feature + 0.3 * X[:, j]  # 70% correlation

    # Create target: only depends on n_ifnormative features
    if coef is None or len(coef) != n_informative:
        coef = np.random.randn(n_informative,) + np.zeros((n_features - n_informative,))
    else:
        coef = np.append(coef,np.zeros((n_features - n_informative,)))
        
    y = X @ coef + np.random.normal(0, noise, n_samples)

    # Generate test set
    X_test = np.random.randn(n_samples, n_features)
    if corr:
        # Add correlation structure
        for i in range(0, n_features, 5):  # Every 5 features are correlated
            end_idx = min(i+5, n_features)
            group_size = end_idx - i
            # Make features within group correlated
            base_feature = X_test[:, i]
            for j in range(i+1, end_idx):
                X_test[:, j] = 0.7 * base_feature + 0.3 * X[:, j]  # 70% correlation
                
    y_test = X_test @ coef + np.random.normal(0, noise, n_samples)

    if plot and n_informative == 1:
        
        true_coefficient = coef[0]
        x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        true_y = true_coefficient * x1_range

        # Quick visualization of the true relationship
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], y, alpha=0.7, s=50)
        plt.xlabel('X1 (Signal Feature)')
        plt.ylabel('Y (Target)')
        plt.title(f'True Relationship: Y ~ {true_coefficient:.2f} * X1 + noise')
        plt.grid(True, alpha=0.3)

        # Add true relationship line
        plt.plot(x1_range, true_y, 'r--', linewidth=2, label='True relationship')
        plt.legend()
        plt.show()

    else:
        # Print informative features and coefficients
        print(f"Informative features: {n_informative}, Coefficients: {coef.flatten()}")

    if test_data:
        return X, y, X_test, y_test, coef
    
    return X, y, coef

def fit_different_models(X, y, coef, feature_counts):
    """Fits linear regression models with different numbers of features and visualizes results.
    
    This function fits separate linear regression models using increasing numbers of features,
    visualizes how predictions compare to the true relationship, and prints performance metrics.
    
    Args:
        X (np.ndarray): Feature matrix with features in columns.
        y (np.ndarray): Target values.
        coef (list): List containing the coefficient for the true relationship.
        feature_counts (list): List of integers specifying how many features to use in each model.
    
    Returns:
        None: Function creates and displays visualizations directly.
    """
    # Create subplots
    fig, axes = plt.subplots(1, len(feature_counts), figsize=(len(feature_counts)*5, 5))

    for i, p in enumerate(feature_counts):
        ax = axes[i]
        
        # Use first p features
        X_p = X[:, :p]
        
        # Fit model with p features
        model_p = LinearRegression()
        model_p.fit(X_p, y)
        
        # Get predictions
        y_pred_p = model_p.predict(X_p)
        
        # Plot real Y values as points
        ax.scatter(X[:, 0], y, alpha=0.7, s=50, color='blue', label='Real Y')
        
        # Plot predicted Y values as crosses
        ax.scatter(X[:, 0], y_pred_p, marker='+', s=100, color='red', linewidth=2, label='Predicted Y')
        
        # Plot true relationship as dotted line
        x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        true_y = coef[0] * x1_range
        ax.plot(x1_range, true_y, 'k:', linewidth=2, label='True relationship')
        
        # Formatting
        ax.set_xlabel('X1')
        ax.set_ylabel('Y')
        ax.set_title(f'p={p} features')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate and print performance metrics
        mse_p = mean_squared_error(y, y_pred_p)

    plt.tight_layout()
    plt.show()

def plot_coefficients(X, y, coef, feature_counts):
    """
    Plots the coefficients of linear regression models fitted with increasing numbers of features, 
    alongside the true coefficients, to visualize the effect of adding noise features on parameter estimates.
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target vector of shape (n_samples,).
        coef (np.ndarray): Array of true coefficients for the features (excluding intercept).
        feature_counts (list of int): List specifying the number of features to include in each model.
    Returns:
        None: Displays a matplotlib bar plot comparing true and estimated coefficients for each model.
    """
    # Fit models with different numbers of features and extract coefficients
    all_coefficients = {}

    for p in feature_counts:
        # Use first p features
        X_p = X[:, :p]
        
        # Fit model
        model_p = LinearRegression()
        model_p.fit(X_p, y)
        
        # Store coefficients (pad with zeros for missing features)
        coeffs = np.zeros(15)
        coeffs[:p] = model_p.coef_
        all_coefficients[f'p={p}'] = coeffs
        
    plt.figure(figsize=(14, 8))

    # Define colors for each model
    colors = sns.color_palette("bright", len(feature_counts) + 1)
    model_names = ['true'] + [f'p={p}' for p in feature_counts]

    # Position parameters for grouped bars
    x_positions = np.arange(16)  # 15 features + intercept
    bar_width = 0.75/len(feature_counts)  # Width of each bar
    x_offset = np.arange(len(model_names)) * bar_width

    # Plot true coefficients (intercept = 0, X1 = 0.2, rest = 0)
    true_all = np.append([0], coef)
    plt.bar(x_positions + x_offset[0], true_all, bar_width, 
            color=colors[0], label='true', alpha=0.8)

    # Plot model coefficients
    for i, p in enumerate(feature_counts):
        model_coeffs = np.zeros(16)
        model_coeffs[0] = LinearRegression().fit(X[:, :p], y).intercept_  # Intercept
        model_coeffs[1:p+1] = LinearRegression().fit(X[:, :p], y).coef_  # Coefficients
        
        plt.bar(x_positions + x_offset[i+1], model_coeffs, bar_width,
                color=colors[i+1], label=f'p={p}', alpha=0.8)

    # Formatting
    plt.xlabel('Features')
    plt.ylabel('beta')
    plt.title('Model Coefficients: How Adding Noise Features Affects Parameter Estimates')
    plt.xticks(x_positions + bar_width * 2, 
            ['(Intercept)'] + [f'X{i}' for i in range(1, 16)], 
            rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_overfitting(X, y, X_test, y_test):
    """
    Fit a linear regression model with increasing numbers of features and visualize overfitting.
    
    This function fits a linear regression model using different numbers of features,
    visualizes how the model's performance changes on both training and test sets,
    and highlights the true number of informative features.
    
    Args:
        X (np.ndarray): Feature matrix for training data.
        y (np.ndarray): Target values for training data.
        X_test (np.ndarray): Feature matrix for test data.
        y_test (np.ndarray): Target values for test data.
        coef (list): List containing the coefficient for the true relationship.
    
    Returns:
        None: Function creates and displays visualizations directly.
    """

    # Calculate train and test performance for all feature counts (1 to 15)
    feature_range = range(1, 9)
    train_mse_list = []
    test_mse_list = []
    train_r2_list = []
    test_r2_list = []

    for p in feature_range:
        # Use first p features
        X_train_p = X[:, :p]
        X_test_p = X_test[:, :p]
        
        # Fit model
        model = LinearRegression()
        model.fit(X_train_p, y)
        
        # Predictions
        y_train_pred = model.predict(X_train_p)
        y_test_pred = model.predict(X_test_p)
        
        # Calculate MSE
        train_mse = mean_squared_error(y, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Calculate R²
        train_r2 = model.score(X_train_p, y)
        test_r2 = model.score(X_test_p, y_test)
        
        # Store results
        train_mse_list.append(train_mse)
        test_mse_list.append(test_mse)
        train_r2_list.append(train_r2)
        test_r2_list.append(test_r2)

    # Create the plots with better scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # MSE plot - use linear scale for both axes
    ax1.plot(feature_range, train_mse_list, 'o-', color='green', linewidth=2, markersize=8, label='Training MSE')
    ax1.plot(feature_range, test_mse_list, 's-', color='red', linewidth=2, markersize=8, label='Test MSE')
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Progressive Overfitting: MSE vs Features')
    #ax1.set_yscale('log')  # Use log scale for better visibility
    ax1.axvline(x=1, color='blue', linestyle='--', alpha=0.7)  # Mark the true number of relevant features
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # R² plot - use linear scale for both axes
    ax2.plot(feature_range, train_r2_list, 'o-', color='green', linewidth=2, markersize=8, label='Training R²')
    ax2.plot(feature_range, test_r2_list, 's-', color='red', linewidth=2, markersize=8, label='Test R²')
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Progressive Overfitting: R² vs Features')
    ax2.axvline(x=1, color='blue', linestyle='--', alpha=0.7)  # Mark the true number of relevant features
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def cross_validate_feature_range(X, y, num_coef=4, feature_counts=None):
    """
    Performs cross-validation to evaluate model performance as a function of the number of features used.
    Standardizes the input features, then iteratively fits a linear regression model using the first `p` features
    (where `p` varies according to `feature_counts`). For each value of `p`, computes the mean and standard deviation
    of the negative mean squared error (MSE) using 5-fold cross-validation. Plots the cross-validation MSE against
    the number of features used.
    Args:
        X (np.ndarray or pd.DataFrame): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray or pd.Series): Target vector of shape (n_samples,).
        feature_counts (int, iterable, or None, optional): Specifies the range of feature counts to evaluate.
            If int, evaluates from 1 to `feature_counts` features.
            If iterable, uses the provided sequence of feature counts.
            If None, evaluates from 1 to the total number of features in `X`.
    Returns:
        None: This function generates a plot and does not return any value.
    Raises:
        ValueError: If `feature_counts` is not an int, iterable, or None.
    """
    
    if isinstance(feature_counts, int):
        feature_counts = range(1, feature_counts + 1)
    elif feature_counts is None:
        feature_counts = range(1, X.shape[1] + 1)
    else:
        feature_counts = np.array(feature_counts)
    np.random.seed(42)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation for different numbers of features
    cv_scores = []
    cv_stds = []

    for p in feature_counts:
        # Use first p features
        X_subset = X_scaled[:, :p]
        
        # 5-fold cross-validation
        scores = cross_val_score(LinearRegression(), X_subset, y, 
                            cv=5, scoring='neg_mean_squared_error')
        
        # Convert to positive MSE and store
        cv_scores.append(-scores.mean())
        cv_stds.append(scores.std())

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(feature_counts, cv_scores, yerr=cv_stds, 
                marker='o', capsize=5, capthick=2)
    if num_coef is not None:
        plt.axvline(x=num_coef, color='red', linestyle='--', alpha=0.7, 
                    label='True number of features (p=4)')
    plt.xlabel('Number of Features Used')
    plt.ylabel('Cross-Validation MSE')
    plt.title('Cross-Validation: Finding Optimal Model Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_pca_cumvar(pca):
    """
    Plots the cumulative explained variance of a fitted PCA object and indicates the number of components required to reach common variance thresholds.
    Args:
        pca (sklearn.decomposition.PCA): A fitted PCA object with the `explained_variance_ratio_` attribute.
    Returns:
        None: This function displays a matplotlib plot and does not return any value.
    Notes:
        The plot includes horizontal lines at 80%, 90%, and 95% explained variance, and annotates these thresholds for visual reference.
    """
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find number of components for different variance thresholds
    thresholds = [0.8, 0.9, 0.95]
    components_needed = {}
    for threshold in thresholds:
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        components_needed[threshold] = n_components

    # Visualize cumulative explained variance only
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', linewidth=2, color='red')
    for threshold in [0.8, 0.9, 0.95]:
        plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.7)
        plt.text(len(cumulative_variance) * 0.8, threshold + 0.01, f'{threshold*100}%', fontsize=10)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance Explained')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_coefs(coefs, title='True Coefficient Values'):
    """
    Plots the values of coefficients as a stem plot.

    Args:
        coefs (array-like): The coefficient values to plot. Should be a 1D array or list of numerical values.
        title (str, optional): The title of the plot. Defaults to 'True Coefficient Values'.

    Returns:
        None: This function displays a plot and does not return any value.
    """
  # Show true coefficients
    plt.figure(figsize=(10, 6))
    plt.stem(range(len(coefs)), coefs, basefmt=' ')
    plt.title(title)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.show()

def cross_validate_regularization(X, y, num_coef=8, alphas=np.logspace(-3, 3, 50), model_name="Ridge"):
    """
    Performs cross-validation for regularized linear regression models (Ridge, Lasso, ElasticNet) 
    over a range of regularization strengths (alphas), plots the cross-validation error, 
    and returns the optimal alpha.
    Args:
        X (array-like): Feature matrix of shape (n_samples, n_features).
        y (array-like): Target vector of shape (n_samples,).
        num_coef (int, optional): Number of coefficients to display or consider (not directly used in function). Defaults to 8.
        alphas (array-like, optional): Sequence of regularization strengths to evaluate. Defaults to np.logspace(-3, 3, 50).
        model_name (str, optional): Name of the regularization model to use. Must be one of "Ridge", "Lasso", or "ElasticNet". Defaults to "Ridge".
    Returns:
        float: The value of alpha that yields the lowest cross-validated mean squared error.
    Raises:
        ValueError: If `model_name` is not one of "Ridge", "Lasso", or "ElasticNet".
    Notes:
        - The function plots mean squared error (with standard deviation) as a function of log(alpha).
        - The optimal alpha is highlighted on the plot.
    """
  # Store cross-validation scores
    cv_scores = []
    cv_stds = []

    if model_name == "Ridge": 
        model_fam = Ridge
    elif model_name == "Lasso":
        model_fam = Lasso
    elif model_name == "ElasticNet":
        model_fam = ElasticNet

    # Cross-validation for each alpha
    for alpha in alphas:
        model = model_fam(alpha=alpha)
        scores = cross_val_score(model, X, y, 
                            cv=5, scoring='neg_mean_squared_error')
        
        # Convert to positive MSE
        cv_scores.append(-scores.mean())
        cv_stds.append(scores.std())

    # Convert to numpy arrays for easier manipulation
    cv_scores = np.array(cv_scores)
    cv_stds = np.array(cv_stds)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.errorbar(np.log10(alphas), cv_scores, yerr=cv_stds, 
                marker='o', markersize=4, capsize=3, capthick=1,
                color='red', ecolor='gray', alpha=0.8)

    # Find optimal alpha
    optimal_idx = np.argmin(cv_scores)
    optimal_alpha = alphas[optimal_idx]
    optimal_score = cv_scores[optimal_idx]

    # Mark optimal point
    plt.axvline(x=np.log10(optimal_alpha), color='black', linestyle='--', alpha=0.7)
    plt.plot(np.log10(optimal_alpha), optimal_score, 'ko', markersize=8, 
            label=f'Optimal α = {optimal_alpha:.3f}')

    # Add text annotations for degrees of freedom
    plt.xlabel('Log(α)', fontsize=12)
    plt.ylabel('Mean-Squared Error', fontsize=12)
    plt.title(f'{model_name} Regression: Cross-Validation Performance vs Regularization Strength', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return optimal_alpha

def regularization_coef_progression(X, y, num_coef=8, alphas=np.logspace(-3, 3, 50), optimal_alpha=None, model_name="Ridge"):
    """
    Plots the progression of regression coefficients as a function of the regularization parameter (alpha)
    for Ridge, Lasso, or ElasticNet regression models.
    This function visualizes how the coefficients of each feature change as the regularization strength varies.
    Signal features (first `num_coef` features) are highlighted with thick lines, while noise features are shown
    with thin gray lines. Optionally, the optimal alpha can be indicated with a vertical dashed line.
    Args:
        X (np.ndarray or pd.DataFrame): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray or pd.Series): Target vector of shape (n_samples,).
        num_coef (int, optional): Number of signal features to highlight. Defaults to 8.
        alphas (array-like, optional): Sequence of alpha values to use for regularization. Defaults to np.logspace(-3, 3, 50).
        optimal_alpha (float, optional): Value of the optimal alpha to highlight on the plot. Defaults to None.
        model_name (str, optional): Type of regression model to use. Must be one of "Ridge", "Lasso", or "ElasticNet". Defaults to "Ridge".
    Returns:
        None: This function displays a matplotlib plot and does not return any value.
    Raises:
        ValueError: If `model_name` is not one of "Ridge", "Lasso", or "ElasticNet".
    Example:
        >>> regularization_coef_progression(X, y, num_coef=5, model_name="Lasso", optimal_alpha=0.1)
    """
    # Plot coefficient paths for Ridge regression
    plt.figure(figsize=(12, 8))

    if model_name == "Ridge": 
        model_fam = Ridge
    elif model_name == "Lasso":
        model_fam = Lasso
    elif model_name == "ElasticNet":
        model_fam = ElasticNet

    # Store coefficients for each alpha
    coefficients = []

    for alpha in alphas:
        ridge = model_fam(alpha=alpha)
        ridge.fit(X, y)
        coefficients.append(ridge.coef_)

    coefficients = np.array(coefficients)

    # Plot coefficient paths
    signal_features = range(num_coef)  # Assume first num_coef features are signal features

    for feature_idx in range(X.shape[1]):
        if feature_idx in signal_features:
            # Thick lines for signal features
            plt.plot(np.log10(alphas), coefficients[:, feature_idx], 
                    linewidth=3, alpha=0.8, 
                    label=f'Signal Feature {feature_idx}' if feature_idx in signal_features[:4] else "")
        else:
            # Thin lines for noise features
            plt.plot(np.log10(alphas), coefficients[:, feature_idx], 
                    linewidth=1, alpha=0.6, color='gray')

    # Add vertical line at optimal alpha
    if optimal_alpha is not None:
        plt.axvline(x=np.log10(optimal_alpha), color='black', linestyle='--', alpha=0.7,
                label=f'Optimal α = {optimal_alpha:.3f}')

    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.xlabel('Log(α)', fontsize=12)
    plt.ylabel('Coefficients', fontsize=12)
    plt.title(f'{model_name} Regression: Coefficient Shrinkage Paths', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Create custom legend
    signal_line = mlines.Line2D([], [], color='C0', linewidth=3, label='Signal Features (thick)')
    noise_line = mlines.Line2D([], [], color='gray', linewidth=1, label='Noise Features (thin)')
    handles = [signal_line, noise_line]
    if optimal_alpha is not None:
        optimal_line = mlines.Line2D([], [], color='black', linestyle='--', label=f'Optimal α = {optimal_alpha:.3f}')
        handles.append(optimal_line)
    plt.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.show()

def features_selected(X, y, n_features, n_informative, optimal_alpha, model_name="Lasso"):
    """
    Fits a regularized linear model (Lasso, Ridge, or ElasticNet) to the data and reports feature selection results.
    The function fits the specified model with the given regularization strength (alpha), identifies selected features
    (non-zero coefficients), and compares them to the true informative features. It prints a summary of selected features,
    their coefficients, whether they are true signal features, and feature selection performance metrics.
    Args:
        X (np.ndarray or pd.DataFrame): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray or pd.Series): Target vector of shape (n_samples,).
        n_features (int): Total number of features in the dataset.
        n_informative (int): Number of true informative (signal) features (assumed to be the first n_informative features).
        optimal_alpha (float): Regularization strength (alpha) to use for the model.
        model_name (str, optional): Name of the model to use. Must be one of "Lasso", "Ridge", or "ElasticNet". Defaults to "Lasso".
    Raises:
        ValueError: If an unsupported model name is provided.
    Prints:
        - The total number of selected features and their indices.
        - The coefficient values for selected features.
        - Whether each selected feature is a true signal feature.
        - Feature selection performance metrics: true positives, false positives, false negatives, precision, and recall.
    """

    if model_name == "Lasso":
        model_fam = Lasso
    elif model_name == "Ridge":
        model_fam = Ridge
    elif model_name == "ElasticNet":
        model_fam = ElasticNet
    else:
        raise ValueError("Unsupported model name. Use 'Lasso', 'Ridge', or 'ElasticNet'.")
    
    # Fit Lasso with optimal alpha to get selected features
    lasso_optimal = model_fam(alpha=optimal_alpha)
    lasso_optimal.fit(X, y)

    signal_features = range(n_informative)  # Assume first features are signal features

    # Get non-zero coefficients (selected features)
    selected_features = np.where(lasso_optimal.coef_ != 0)[0]
    selected_coeffs = lasso_optimal.coef_[selected_features]

    print(f"Lasso Feature Selection Results (α = {optimal_alpha:.4f}):")
    print(f"Total features selected: {len(selected_features)} out of {n_features}")
    print("\nSelected Features:")
    print("Feature Index | Coefficient | True Signal?")
    print("-" * 40)

    for feat_idx, coeff in zip(selected_features, selected_coeffs):
        is_signal = "✓" if feat_idx in signal_features else "✗"
        print(f"{feat_idx:12d} | {coeff:10.4f} | {is_signal}")

    # Calculate accuracy of feature selection
    true_positives = len(set(selected_features) & set(signal_features))
    false_positives = len(set(selected_features) - set(signal_features))
    false_negatives = len(set(signal_features) - set(selected_features))

    precision = true_positives / len(selected_features) if len(selected_features) > 0 else 0
    recall = true_positives / len(signal_features)

    print(f"\nFeature Selection Performance:")
    print(f"True Positives: {true_positives}/{n_informative}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

def cross_validate_lambda_en(X, y, alphas=np.logspace(-5, 0, 50)):
    """
    Perform cross-validation to select the optimal regularization strength (alpha) and L1/L2 mixing ratio (l1_ratio) for ElasticNet regression.
    This function evaluates ElasticNet models with different combinations of alpha (regularization strength) and l1_ratio (mixing between L1 and L2 penalties)
    using cross-validation. It plots the mean squared error (MSE) for each combination and highlights the optimal alpha for each l1_ratio. The function also
    prints a summary of the optimal parameters and their corresponding cross-validated MSE for each method (Ridge, ElasticNet, Lasso).
    Args:
        X (array-like or pd.DataFrame): Feature matrix of shape (n_samples, n_features).
        y (array-like or pd.Series): Target vector of shape (n_samples,).
        alphas (array-like, optional): Sequence of alpha values to test. Default is np.logspace(-5, 0, 50).
    Returns:
        tuple:
            best_l1_ratio (float): The l1_ratio value with the lowest cross-validated MSE.
            best_alpha (float): The alpha value corresponding to the best_l1_ratio with the lowest cross-validated MSE.
    Raises:
        ValueError: If X or y have incompatible shapes.
    Displays:
        - A plot of cross-validated MSE vs. log(alpha) for each l1_ratio.
        - Prints a summary table of optimal alpha and MSE for each method.
    """
    # Define range of alpha values and l1_ratio values
    l1_ratios = [0, 0.25, 0.5, 0.75, 1.0]  # Different mixing ratios

    # Colors for different l1_ratios
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    labels = ['Ridge (λ=0)', 'ElasticNet (λ=0.25)', 'ElasticNet (λ=0.5)', 
            'ElasticNet (λ=0.75)', 'Lasso (λ=1)']

    plt.figure(figsize=(12, 8))

    # Store optimal results for each l1_ratio
    optimal_results = []

    for i, l1_ratio in enumerate(l1_ratios):
        cv_scores = []
        
        # Cross-validation for each alpha
        for alpha in alphas:
            elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            scores = cross_val_score(elastic, X, y, 
                                cv=5, scoring='neg_mean_squared_error')
            cv_scores.append(-scores.mean())
        
        cv_scores = np.array(cv_scores)
        
        # Plot line for this l1_ratio
        plt.plot(np.log10(alphas), cv_scores, 
                color=colors[i], linewidth=2, label=labels[i])
        
        # Find optimal alpha for this l1_ratio
        optimal_idx = np.argmin(cv_scores)
        optimal_alpha = alphas[optimal_idx]
        optimal_score = cv_scores[optimal_idx]
        optimal_results.append((l1_ratio, optimal_alpha, optimal_score))
        
        # Mark optimal point
        plt.plot(np.log10(optimal_alpha), optimal_score, 'o', 
                color=colors[i], markersize=8, markeredgecolor='black', markeredgewidth=1)

    plt.xlabel('Log(α)', fontsize=12)
    plt.ylabel('Mean-Squared Error', fontsize=12)
    plt.title('ElasticNet: Cross-Validation Performance vs Regularization Strength\nfor Different L1/L2 Mixing Ratios', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    # Print optimal results
    print("Optimal Results for Each Method:")
    print("Method | L1 Ratio | Optimal α | CV MSE")
    print("-" * 45)
    for l1_ratio, opt_alpha, opt_score in optimal_results:
        method_name = labels[l1_ratios.index(l1_ratio)]
        print(f"{method_name:20s} | {l1_ratio:8.2f} | {opt_alpha:9.4f} | {opt_score:.4f}")

    # Find overall best method
    best_idx = np.argmin([result[2] for result in optimal_results])
    best_method = labels[best_idx]
    best_l1_ratio, best_alpha, best_score = optimal_results[best_idx]

    return best_l1_ratio, best_alpha

def compare_coefficients_mult_methods(X, y, pca_n=4, reg_results=None, feature_names=None):
    """
    Compare and visualize regression coefficients from multiple regularization methods.
    This function fits Principal Component Regression (PCR), Ridge, Lasso, and ElasticNet models
    to the provided data using optimal hyperparameters (as specified in `reg_results`), extracts
    their coefficients, and plots a grouped bar chart to compare the coefficients for each feature
    across the different methods.
    Args:
        X (array-like or pd.DataFrame): Feature matrix of shape (n_samples, n_features).
        y (array-like): Target vector of shape (n_samples,).
        pca_n (int, optional): Number of principal components to use in PCR. Defaults to 4.
        reg_results (dict): Dictionary containing optimal hyperparameters for each method.
            Expected keys: 'Ridge', 'Lasso', 'ElasticNet', each mapping to a dict with key 'optimal_alpha'.
        feature_names (list of str): List of feature names corresponding to columns in X.
    Raises:
        ValueError: If `reg_results` is not provided.
    Returns:
        None: Displays a matplotlib plot comparing coefficients across methods.
    """
    if reg_results is None:
        raise ValueError("Regularization results must be provided.")
    
    # Get optimal coefficients for each method
    methods_coef = {}

    # PCR coefficients (at optimal number of components)
    pcr_pipeline = Pipeline([
        ('pca', PCA(n_components=pca_n)),
        ('regression', LinearRegression())
    ])
    pcr_pipeline.fit(X, y)
    # Transform PCA coefficients back to original feature space
    pca_components = pcr_pipeline.named_steps['pca'].components_
    pcr_coef_original = pcr_pipeline.named_steps['regression'].coef_ @ pca_components
    methods_coef['PCR'] = pcr_coef_original

    # Ridge coefficients (at optimal alpha)
    ridge_optimal = Ridge(alpha=reg_results['Ridge']['optimal_alpha'])
    ridge_optimal.fit(X, y)
    methods_coef['Ridge'] = ridge_optimal.coef_

    # Lasso coefficients (at optimal alpha)
    lasso_optimal = Lasso(alpha=reg_results['Lasso']['optimal_alpha'])
    lasso_optimal.fit(X, y)
    methods_coef['Lasso'] = lasso_optimal.coef_

    # ElasticNet coefficients (at optimal alpha)
    elastic_optimal = ElasticNet(alpha=reg_results['ElasticNet']['optimal_alpha'], l1_ratio=0.5)
    elastic_optimal.fit(X, y)
    methods_coef['ElasticNet'] = elastic_optimal.coef_

    # Create DataFrame for plotting
    coef_data = []

    for method, coefficients in methods_coef.items():
        for i, (feature, coef) in enumerate(zip(feature_names, coefficients)):
            coef_data.append({
                'feature': feature,
                'beta': coef,
                'method': method,
                'feature_idx': i
            })

    coef_df = pd.DataFrame(coef_data)

    # Create the plot
    plt.figure(figsize=(16, 8))

    # Define colors for methods
    method_colors = {'PCR': 'orange', 'Ridge': 'blue', 'Lasso': 'purple', 'ElasticNet': 'red'}

    # Create grouped bar plot
    x_pos = np.arange(len(feature_names))
    width = 0.2

    for i, method in enumerate(['PCR', 'Ridge', 'Lasso', 'ElasticNet']):
        method_coefs = coef_df[coef_df['method'] == method]['beta'].values
        plt.bar(x_pos + i*width, method_coefs, width, 
                label=method, color=method_colors[method], alpha=0.8)

    # Customize plot
    plt.xlabel('Features')
    plt.ylabel('β (Coefficient Value)')
    plt.title('Coefficient Comparison Across Regularization Methods')
    plt.xticks(x_pos + width*1.5, feature_names, rotation=45, ha='right')
    plt.legend(title='Method')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.show()

def regularization_analysis(X, y, alphas_ridge=np.logspace(-1, 6, 50), alphas_lasso=np.logspace(-3, 2, 50), alphas_elastic=np.logspace(-3, 2, 50)):
    """
    Perform comparative analysis of Ridge, Lasso, and ElasticNet regularization on a dataset.
    This function evaluates the effect of different regularization strengths (alpha) for Ridge, Lasso, 
    and ElasticNet regression using cross-validation. It visualizes cross-validation MSE and coefficient 
    shrinkage paths for each method, and prints a summary of optimal parameters and feature selection.
    Args:
        X (array-like or pd.DataFrame): Feature matrix of shape (n_samples, n_features).
        y (array-like or pd.Series): Target vector of shape (n_samples,).
        alphas_ridge (array-like, optional): Array of alpha values to test for Ridge regression. 
            Defaults to np.logspace(-1, 6, 50).
        alphas_lasso (array-like, optional): Array of alpha values to test for Lasso regression. 
            Defaults to np.logspace(-3, 2, 50).
        alphas_elastic (array-like, optional): Array of alpha values to test for ElasticNet regression. 
            Defaults to np.logspace(-3, 2, 50).
    Returns:
        dict: A dictionary with keys 'Ridge', 'Lasso', and 'ElasticNet', each containing:
            - 'alphas': Array of alpha values tested.
            - 'cv_scores': Cross-validation mean squared errors for each alpha.
            - 'cv_stds': Standard deviation of CV scores for each alpha.
            - 'coefficients': Coefficient values for each alpha.
            - 'optimal_alpha': Alpha value with the lowest CV MSE.
            - 'optimal_score': Best (lowest) CV MSE.
    Displays:
        - Matplotlib figure with two rows:
            - Top row: Cross-validation MSE vs log(alpha) for each method.
            - Bottom row: Coefficient shrinkage paths vs log(alpha) for each method.
        - Prints a summary of optimal alpha, best CV MSE, and number of features selected for each method.
    """
    # Store results for reuse
    results = {}

    # Function to perform cross-validation and get coefficients
    def analyze_regularization(model_class, alphas, model_name, **kwargs):
        cv_scores = []
        cv_stds = []
        coefficients = []
        
        for alpha in alphas:
            # Cross-validation
            model = model_class(alpha=alpha, **kwargs)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            cv_scores.append(-scores.mean())
            cv_stds.append(scores.std())
            
            # Fit for coefficients
            model.fit(X, y)
            coefficients.append(model.coef_)
        
        # Find optimal alpha
        optimal_idx = np.argmin(cv_scores)
        optimal_alpha = alphas[optimal_idx]
        
        return {
            'alphas': alphas,
            'cv_scores': np.array(cv_scores),
            'cv_stds': np.array(cv_stds),
            'coefficients': np.array(coefficients),
            'optimal_alpha': optimal_alpha,
            'optimal_score': cv_scores[optimal_idx]
        }

    # Analyze all methods
    results['Ridge'] = analyze_regularization(Ridge, alphas_ridge, 'Ridge')
    results['Lasso'] = analyze_regularization(Lasso, alphas_lasso, 'Lasso')
    results['ElasticNet'] = analyze_regularization(ElasticNet, alphas_elastic, 'ElasticNet', l1_ratio=0.5)

    # Create the comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    methods = ['Ridge', 'Lasso', 'ElasticNet']
    colors = ['blue', 'purple', 'red']

    # Top row: CV scores
    for i, (method, color) in enumerate(zip(methods, colors)):
        ax = axes[0, i]
        result = results[method]
        
        ax.errorbar(np.log10(result['alphas']), result['cv_scores'], 
                    yerr=result['cv_stds'], marker='o', markersize=4, 
                    capsize=3, capthick=1, color=color, ecolor='gray', alpha=0.8)
        
        # Mark optimal point
        ax.axvline(x=np.log10(result['optimal_alpha']), color='black', 
                linestyle='--', alpha=0.7)
        ax.plot(np.log10(result['optimal_alpha']), result['optimal_score'], 
                'ko', markersize=8, label=f'Optimal α = {result["optimal_alpha"]:.3f}')
        
        ax.set_xlabel('Log(α)')
        ax.set_ylabel('Cross-Validation MSE')
        ax.set_title(f'{method}: CV Performance vs Regularization')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Bottom row: Coefficient paths
    for i, (method, color) in enumerate(zip(methods, colors)):
        ax = axes[1, i]
        result = results[method]
        
        # Plot all coefficient paths
        for feature_idx in range(X.shape[1]):
            ax.plot(np.log10(result['alphas']), result['coefficients'][:, feature_idx], 
                    linewidth=1.5, alpha=0.7)
        
        # Mark optimal alpha
        ax.axvline(x=np.log10(result['optimal_alpha']), color='black', 
                linestyle='--', alpha=0.7, label=f'Optimal α = {result["optimal_alpha"]:.3f}')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Log(α)')
        ax.set_ylabel('Coefficients')
        ax.set_title(f'{method}: Coefficient Shrinkage Paths')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Print summary
    print("Regularization Methods Comparison:")
    print("="*50)
    for method in methods:
        result = results[method]
        print(f"{method}:")
        print(f"  Optimal α: {result['optimal_alpha']:.4f}")
        print(f"  Best CV MSE: {result['optimal_score']:.2f}")
        
        # Count non-zero coefficients at optimal alpha
        optimal_idx = np.argmin(result['cv_scores'])
        non_zero_coefs = np.sum(np.abs(result['coefficients'][optimal_idx]) > 1e-6)
        print(f"  Features selected: {non_zero_coefs}/{X.shape[1]}")
        print()
    return results

### Decision trees

def plot_random_forest_accuracy(X_train, y_train, X_test, y_test, max_trees=100):
    """
    Plots the accuracy of a Random Forest classifier as the number of trees increases.
    This function trains Random Forest classifiers with varying numbers of trees and plots
    the ensemble accuracy on the test set as a function of the number of trees. It also
    highlights the accuracy of a single decision tree for comparison.
    Args:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target labels.
        X_test (array-like): Test feature data.
        y_test (array-like): Test target labels.
        max_trees (int, optional): Maximum number of trees to evaluate. Defaults to 100.
    Returns:
        None: Displays a matplotlib plot of accuracy vs. number of trees.
    """

    # Show how ensemble accuracy improves with more trees
    plt.figure(figsize=(10, 6))

    n_trees_range = range(1, max_trees, 5)
    ensemble_scores = []

    for n_trees in n_trees_range:
        rf_temp = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
        rf_temp.fit(X_train, y_train)
        ensemble_scores.append(rf_temp.score(X_test, y_test))
        if n_trees == 1:
            single_tree_accuracy = rf_temp.score(X_test, y_test)

    plt.plot(n_trees_range, ensemble_scores, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Number of Trees')
    plt.ylabel('Ensemble Accuracy')
    plt.title('Accuracy vs Number of Trees')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=single_tree_accuracy, color='red', linestyle='--', 
                alpha=0.7, label=f'Single Tree: {single_tree_accuracy:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def simple_feature_importance(X, tree):
    """
    Displays and visualizes the feature importances from a fitted decision tree model.
    Args:
        X (pd.DataFrame): The input feature set used to train the decision tree. Must have column names.
        tree (sklearn.tree.DecisionTreeClassifier or DecisionTreeRegressor): 
            A fitted decision tree model with the `feature_importances_` attribute.
    Returns:
        None: This function displays a bar plot of feature importances and does not return any value.
    Raises:
        AttributeError: If `tree` does not have the `feature_importances_` attribute.
        AttributeError: If `X` does not have a `columns` attribute.
    Example:
        >>> simple_feature_importance(X_train, clf)
    """
    # Show feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': tree.feature_importances_
    }).sort_values('importance', ascending=False)

    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance in Heart Disease Prediction')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

def compare_feature_importance(X, tree1, tree1_name, tree2, tree2_name):
    """
    Compare and visualize feature importances from two tree-based models.
    This function creates side-by-side bar plots of feature importances for two models
    (e.g., Random Forest and Gradient Boosting) and a combined comparison plot.
    It helps to visually assess which features are considered important by each model.
    Args:
        X (pd.DataFrame): The input feature DataFrame used to train the models.
        tree1 (sklearn.base.BaseEstimator): The first fitted tree-based model with `feature_importances_` attribute.
        tree1_name (str): Name of the first model (used for labeling plots).
        tree2 (sklearn.base.BaseEstimator): The second fitted tree-based model with `feature_importances_` attribute.
        tree2_name (str): Name of the second model (used for labeling plots).
    Returns:
        None: This function displays matplotlib figures and does not return any value.
    Raises:
        AttributeError: If either model does not have the `feature_importances_` attribute.
    """
  
    # Get feature importances for Random Forest and Gradient Boosting
    importance_comparison = pd.DataFrame({
        'feature': X.columns,
        tree1_name: tree1.feature_importances_,
        tree2_name: tree2.feature_importances_
    }).sort_values(tree1_name, ascending=False)

    # Visualize feature importance comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x_max = max(importance_comparison[tree1_name].max(),
                importance_comparison[tree2_name].max())
    
    sns.barplot(data=importance_comparison, x=tree1_name, y='feature', 
            palette='viridis', ax=axes[0])
    axes[0].set_title(f'{tree1_name} Feature Importance')
    axes[0].set_xlabel('Importance Score')
    axes[0].set_xlim(0, x_max)

    sns.barplot(data=importance_comparison, x=tree2_name, y='feature', 
            palette='cividis', ax=axes[1])
    axes[1].set_title(f'{tree2_name} Feature Importance')
    axes[1].set_xlabel('Importance Score')
    axes[1].set_xlim(0, x_max)

    # Comparison of both methods
    importance_melted = importance_comparison.melt(
        id_vars=['feature'], 
        value_vars=[tree1_name, tree2_name],
        var_name='model', value_name='importance'
    )

    sns.barplot(data=importance_melted, x='importance', y='feature', 
            hue='model', ax=axes[2])
    axes[2].set_title('Feature Importance Comparison')
    axes[2].set_xlabel('Importance Score')
    axes[2].legend(title='Model', labels=[tree1_name, tree2_name])
    axes[2].set_xlim(0, x_max)

    plt.tight_layout()
    plt.show()

# Select top 4 most important features for PDP analysis
def plot_partial_dependence(X, rf, n_features=4, title='Partial Dependence Plots'):
    """
    Plots partial dependence plots for the top N most important features of a fitted Random Forest model.
    Args:
        X (pandas.DataFrame): The input feature data used for training the model.
        rf (sklearn.ensemble.RandomForestClassifier or RandomForestRegressor): 
            The fitted Random Forest model.
        n_features (int, optional): Number of top features (by importance) to plot. Defaults to 4.
        title (str, optional): Title for the entire plot figure. Defaults to 'Partial Dependence Plots'.
    Returns:
        None: Displays the partial dependence plots using matplotlib.
    Notes:
        - The function prints the names of the top features being analyzed.
        - The y-axis label is set to 'Heart Disease Probability' by default.
        - Requires matplotlib, numpy, and sklearn.inspection.PartialDependenceDisplay.
    """

    importances = rf.feature_importances_
    # Create a list of feature names sorted by importance (descending)
    top_features = [X.columns[i] for i in importances.argsort()[::-1][:n_features]]
    print(f"Analyzing partial dependence for: {top_features}")

    # Create partial dependence plots
    n_rows = int(np.ceil(n_features / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 6 * n_rows))
    axes = axes.ravel()

    for i, feature in enumerate(top_features):
        # Calculate partial dependence
        PartialDependenceDisplay.from_estimator(
            rf, X, features=[feature], 
            ax=axes[i], kind='average'
        )
        axes[i].set_title(f'Partial Dependence: {feature}')
        axes[i].set_ylabel('Heart Disease Probability')  # Change y-axis label

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def compare_gb_rf_log(X_train, y_train, X_test, y_test, show_boosting=True, cv=False):
    """
    Compare the performance of Gradient Boosting, Random Forest, and Logistic Regression classifiers.
    Trains and evaluates three classifiers (Gradient Boosting, Random Forest, and Logistic Regression)
    on the provided training and test data. Optionally, displays the training curve for Gradient Boosting
    and supports cross-validation for model evaluation. Visualizes the results using bar plots and, if requested,
    the error reduction curve for Gradient Boosting.
    Args:
        X_train (array-like): Training feature matrix.
        y_train (array-like): Training target vector.
        X_test (array-like): Test feature matrix.
        y_test (array-like): Test target vector.
        show_boosting (bool, optional): If True, plots the training curve for Gradient Boosting. Defaults to True.
        cv (bool, optional): If True, evaluates models using 5-fold cross-validation on the training set. 
            If False, evaluates using the provided test set. Defaults to False.
    Returns:
        dict: Dictionary of fitted model instances with model names as keys.
    """
    
    # Initialize gradient boosting and comparison models
    models = {
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
        )
    }

    # Train all models and compare performance
    results = {}
    training_curves = {}

    for name, model in models.items():
        
        if cv:
            # Use cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            accuracy = cv_scores.mean()
            results[name] = {'mean': accuracy, 'std': cv_scores.std()}
        else:
            # Use train/test split
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
        
        # Get training curve for Gradient Boosting (only if showing boosting and not CV)
        if name == 'Gradient Boosting' and show_boosting and not cv:
            # Ensure model is fitted
            if not hasattr(model, 'estimators_'):
                model.fit(X_train, y_train)
            # Get staged predictions to track improvement over iterations
            staged_pred_proba = list(model.staged_predict_proba(X_test))
            training_curves[name] = [log_loss(y_test, pred[:, 1]) for pred in staged_pred_proba]

    # Determine subplot layout
    if show_boosting and training_curves:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(8, 5))

    # Training curve for Gradient Boosting (only if requested and available)
    if show_boosting and training_curves:
        for name, curve in training_curves.items():
            ax1.plot(curve, label=name, linewidth=2, color='darkgreen')
        ax1.set_xlabel('Boosting Iterations')
        ax1.set_ylabel('Log Loss')
        ax1.set_title('Gradient Boosting: Error Reduction Over Iterations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Performance comparison
    names = list(results.keys())
    colors = ['darkgreen', 'skyblue', 'lightcoral']

    if cv:
        # Show cross-validation results with error bars
        accuracies = [results[name]['mean'] for name in names]
        std_errors = [results[name]['std'] for name in names]
        
        bars = ax2.bar(range(len(names)), accuracies, color=colors, alpha=0.7, 
                      yerr=std_errors, capsize=5)
        ax2.set_ylabel('CV Accuracy (±std)')
        ax2.set_title('Model Performance Comparison (5-Fold CV)')
        
        # Add value labels on bars with std
        for bar, acc, std in zip(bars, accuracies, std_errors):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                    f'{acc:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        # Show regular test results
        accuracies = list(results.values())
        bars = ax2.bar(range(len(names)), accuracies, color=colors, alpha=0.7)
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Model Performance Comparison')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.show()

    return models

def compare_gb_rf_linear(X_train, y_train, X_test, y_test, show_boosting=False, cv=True):
    """
    Compare the performance of Gradient Boosting, Random Forest, and Linear Regression models on a regression task.
    This function trains and evaluates three regression models (Gradient Boosting, Random Forest, and Linear Regression)
    using either cross-validation or a train-test split. It visualizes the performance comparison and, optionally,
    the error reduction curve for Gradient Boosting.
    Args:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target values.
        X_test (array-like): Test feature data.
        y_test (array-like): Test target values.
        show_boosting (bool, optional): If True, plots the error reduction curve for Gradient Boosting. Default is False.
        cv (bool, optional): If True, uses 5-fold cross-validation for evaluation. If False, uses train-test split. Default is True.
    Returns:
        dict: A dictionary containing the fitted model objects for each algorithm, with model names as keys.
    """

    # Initialize gradient boosting and comparison models
    models = {
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
        'Linear Regression': LinearRegression()
    }

    # Train all models and compare performance
    results = {}
    training_curves = {}

    if cv:
        # Cross-validation approach
        from sklearn.model_selection import cross_val_score
        
        for name, model in models.items():
            # Use R² score for regression
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            results[name] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            
        # For display purposes, fit the models on full training data
        for name, model in models.items():
            model.fit(X_train, y_train)
            
    else:
        # Train-test split approach
        for name, model in models.items():
            
            model.fit(X_train, y_train)
            
            # Get training curve for Gradient Boosting
            if name == 'Gradient Boosting' and show_boosting:
                # Get staged predictions to track improvement over iterations
                staged_predictions = list(model.staged_predict(X_test))
                training_curves[name] = [mean_squared_error(y_test, pred) for pred in staged_predictions]
            
            # Get predictions and R² score
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            results[name] = r2

    # Visualize training curves and performance comparison
    if show_boosting and not cv:
        plt.figure(figsize=(12, 5))
        
        # Training curve for Gradient Boosting
        plt.subplot(1, 2, 1)
        if training_curves:
            for name, curve in training_curves.items():
                plt.plot(curve, label=name, linewidth=2, color='darkgreen')
            plt.xlabel('Boosting Iterations')
            plt.ylabel('Mean Squared Error')
            plt.title('Gradient Boosting: Error Reduction Over Iterations')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Performance comparison
        plt.subplot(1, 2, 2)
    else:
        plt.figure(figsize=(8, 5))

    # Performance comparison subplot
    if show_boosting and not cv:
        ax = plt.subplot(1, 2, 2)
    else:
        ax = plt.gca()

    names = list(results.keys())
    colors = ['darkgreen', 'skyblue', 'lightcoral']

    if cv:
        # Cross-validation results with error bars
        means = [results[name]['mean'] for name in names]
        stds = [results[name]['std'] for name in names]
        
        bars = ax.bar(range(len(names)), means, yerr=stds, color=colors, alpha=0.7, capsize=5)
        
        # Add value labels on bars with std
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Model Performance Comparison (5-Fold CV)')
        ax.set_ylabel('R² Score (mean ± std)')
        
    else:
        # Regular train-test results
        scores = list(results.values())
        bars = ax.bar(range(len(names)), scores, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('R² Score')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.show()

    return models

def gb_hyperparam_analysis(X_train, y_train, X_test, y_test):
    """
    Analyzes the effect of key hyperparameters on the performance of a Gradient Boosting Classifier.
    This function trains multiple GradientBoostingClassifier models on the provided training data,
    varying the number of trees (`n_estimators`), learning rate, and maximum tree depth (`max_depth`).
    For each hyperparameter, it evaluates the model's accuracy on the test set and visualizes the results
    in three subplots.
    Args:
        X_train (array-like or pd.DataFrame): Training feature data.
        y_train (array-like or pd.Series): Training target labels.
        X_test (array-like or pd.DataFrame): Test feature data.
        y_test (array-like or pd.Series): Test target labels.
    Returns:
        None: This function displays plots but does not return any value.
    Raises:
        None
    Example:
        gb_hyperparam_analysis(X_train, y_train, X_test, y_test)
    """
    # 1. Number of trees analysis
    n_trees_to_show = [1, 5, 10, 25, 50, 100]
    tree_results = []

    for n_trees in n_trees_to_show:
        temp_model = GradientBoostingClassifier(
            n_estimators=n_trees,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        temp_model.fit(X_train, y_train)
        accuracy = temp_model.score(X_test, y_test)
        tree_results.append(accuracy)

    # 2. Learning rate analysis
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
    lr_results = []

    for lr in learning_rates:
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=lr,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        lr_results.append(accuracy)

    # 3. Max depth analysis
    max_depths = [1, 2, 3, 4, 5, 6]
    depth_results = []

    for depth in max_depths:
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        depth_results.append(accuracy)

    # Visualize all hyperparameter effects
    plt.figure(figsize=(15, 5))

    # Number of trees plot
    plt.subplot(1, 3, 1)
    plt.plot(n_trees_to_show, tree_results, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Number of Trees vs Accuracy')
    plt.ylim(0.55, 0.75)
    plt.grid(True, alpha=0.3)

    # Add accuracy values as labels
    for x, y in zip(n_trees_to_show, tree_results):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')

    # Learning rate plot
    plt.subplot(1, 3, 2)
    plt.plot(learning_rates, lr_results, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Learning Rate vs Accuracy')
    plt.ylim(0.55, 0.75)
    plt.grid(True, alpha=0.3)

    # Max depth plot
    plt.subplot(1, 3, 3)
    plt.plot(max_depths, depth_results, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Tree Depth vs Accuracy')
    plt.ylim(0.55, 0.75)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

### Classification

def generate_simple_classification_data(separable=False, scale=True, plot=True):
    """
    Generates a simple 2D synthetic classification dataset for visualization and experimentation.
    The function creates a dataset with two features and two classes, with options to control class separability,
    feature scaling, and plotting. Useful for demonstrating classification algorithms and visualizations.
    Args:
        separable (bool, optional): If True, generates well-separated classes with no label noise.
            If False, generates classes with lower separation and some label noise. Defaults to False.
        scale (bool, optional): If True, standardizes features using StandardScaler. Defaults to True.
        plot (bool, optional): If True, plots the generated dataset using matplotlib. Defaults to True.
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - X (np.ndarray): Feature matrix of shape (n_samples, 2).
            - y (np.ndarray): Target labels of shape (n_samples,).
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    if separable:
        # For separable classes, increase class separation and reduce noise
        class_sep = 4
        flip_y = 0
    else:
        # For non-separable classes, use lower class separation and add some noise
        class_sep = 0.8
        flip_y = 0.05

    # Generate a 2D classification dataset
    X, y = make_classification(
        n_samples=300,           # Total number of samples
        n_features=2,            # Two features for easy visualization
        n_redundant=0,           # No redundant features
        n_informative=2,         # Both features are informative
        n_clusters_per_class=1,  # One cluster per class
        class_sep=class_sep,     # Separation between classes
        flip_y=flip_y,           # Add 5% label noise
        random_state=42
    )

    # Standardize features for better visualization and algorithm performance
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Visualize the standardized dataset
    if plot:
        plt.figure(figsize=(8, 6))
        colors = ['red', 'blue']
        for class_val in [0, 1]:
            mask = y == class_val
            plt.scatter(X[mask, 0], X[mask, 1], c=colors[class_val], 
                        label=f'Class {class_val}', alpha=0.7, s=50)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('2D Classification Dataset')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return X, y

def evaluateModel(X, y, model, model_name='Logistic Regression', ax=None):
    """
    Evaluates a classification model using 5-fold cross-validation and visualizes the results.
    Performs cross-validation on the provided model using the given features and labels,
    computes accuracy, precision, recall, and F1-score, and displays a bar plot with error bars
    representing the mean and standard deviation of each metric.
    Args:
        X (array-like or pd.DataFrame): Feature matrix for model evaluation.
        y (array-like or pd.Series): Target labels corresponding to X.
        model (estimator object): Scikit-learn compatible classification model to evaluate.
        model_name (str, optional): Name of the model for plot title. Defaults to 'Logistic Regression'.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. If None, a new figure and axis are created.
    Returns:
        None: This function displays a plot and does not return any value.
    """
    # Define scoring metrics for cross-validation
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    # Perform 5-fold cross-validation with multiple metrics
    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

    # Calculate means and standard deviations
    metrics_summary = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        scores = cv_results[f'test_{metric}']
        metrics_summary[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }

    # Display results
    #print("CLASSIFICATION METRICS SUMMARY (5-Fold Cross-Validation)")
    #print("=" * 60)
    #for metric, results in metrics_summary.items():
    #    print(f"{metric.capitalize():10}: {results['mean']:.3f} (±{results['std']:.3f})")

    # Visualize metrics with error bars
    metrics_names = [metric.capitalize() for metric in metrics_summary.keys()]
    means = [results['mean'] for results in metrics_summary.values()]
    stds = [results['std'] for results in metrics_summary.values()]

    # Create figure/axis if not provided
    if ax is None:
        plt.figure(figsize=(6, 4))
        ax = plt.gca()

    bars = ax.bar(metrics_names, means, yerr=stds, 
                color=['skyblue', 'lightgreen', 'lightcoral', 'gold'], 
                alpha=0.8, capsize=10, error_kw={'elinewidth': 2})

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_title(f'Classification Metrics - {model_name} (5-Fold Cross-Validation)')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Only show if we created our own figure
    if ax == plt.gca():
        plt.tight_layout()
        plt.show()

def evaluate_roc(X, y, model=None, model_name=None, models=None, model_names=None, ax=None):
    """
    Plots ROC curves and evaluates ROC-AUC performance for one or multiple classification models using cross-validation.
    This function computes ROC curves and AUC scores for the provided model(s) using cross-validation,
    plots the ROC curves, and prints summary statistics. It supports both single and multiple models.
    If no matplotlib axis is provided, a new figure is created.
    Args:
        X (array-like or pd.DataFrame): Feature matrix of shape (n_samples, n_features).
        y (array-like or pd.Series): Target vector of shape (n_samples,).
        model (sklearn.base.BaseEstimator, optional): A single fitted classification model.
        model_name (str, optional): Name for the single model (used in plot legend and reporting).
        models (list of sklearn.base.BaseEstimator, optional): List of fitted classification models.
        model_names (list of str, optional): List of names for the models.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. If None, a new figure is created.
    Raises:
        ValueError: If neither `model` nor `models` is provided.
        ValueError: If the length of `models` and `model_names` does not match.
    Prints:
        ROC-AUC performance summary for each model, including mean and standard deviation of cross-validated AUC,
        and the ROC AUC computed from the ROC curve.
    Returns:
        None: The function displays the ROC plot and prints results, but does not return any value.
    """
    
    # Handle single model vs multiple models
    if models is None:
        if model is None:
            raise ValueError("Either 'model' or 'models' must be provided")
        models = [model]
        model_names = [model_name]
    else:
        if model_names is None:
            model_names = [f'Model {i+1}' for i in range(len(models))]
        elif len(models) != len(model_names):
            raise ValueError("Length of models and model_names must match")
    
    # Create figure/axis if not provided
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        show_plot = True
    else:
        show_plot = False
    
    # Colors for multiple models
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Store results for reporting
    all_results = {}
    
    # Process each model
    for i, (model, name) in enumerate(zip(models, model_names)):
        # Get probability predictions using cross-validation
        y_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Also get AUC scores from cross-validation
        auc_scores = cross_validate(model, X, y, cv=5, scoring='roc_auc')['test_score']
        
        # Store results
        all_results[name] = {
            'cv_auc_mean': auc_scores.mean(),
            'cv_auc_std': auc_scores.std(),
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        }
        
        # Plot ROC curve
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2, 
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Plot random classifier line
    ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
            label='Random Classifier', alpha=0.7)
    
    # Set plot properties
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    
    # Set title based on number of models
    if len(models) == 1:
        ax.set_title(f'ROC Curve - {model_names[0]}')
    else:
        ax.set_title('ROC Curves Comparison')
    
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Print results
    print("ROC-AUC PERFORMANCE")
    print("=" * 50)
    for name, results in all_results.items():
        print(f"{name}:")
        print(f"  Cross-validation AUC: {results['cv_auc_mean']:.3f} (±{results['cv_auc_std']:.3f})")
        print(f"  ROC AUC: {results['roc_auc']:.3f}")
        print()
    
    # Show plot only if we created our own figure
    if show_plot:
        plt.tight_layout()
        plt.show()
    
    # return all_results

def evaluate_confusion_matrix(X, y, model, model_name='Logistic Regression', ax=None):
    """
    Evaluates and visualizes the confusion matrix for a classification model using cross-validation.
    This function performs 5-fold cross-validation to obtain out-of-fold predictions, computes the confusion matrix,
    prints detailed confusion matrix statistics (including error rates), and visualizes the matrix using matplotlib.
    Optionally, an existing matplotlib axis can be provided for plotting.
    Args:
        X (array-like or pd.DataFrame): Feature matrix for the input data.
        y (array-like or pd.Series): True labels for the input data.
        model (estimator object): Scikit-learn compatible classification model to evaluate.
        model_name (str, optional): Name of the model to display in the plot title. Defaults to 'Logistic Regression'.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. If None, a new figure and axis are created.
    Prints:
        - Confusion matrix with counts of true negatives, false positives, false negatives, and true positives.
        - Overall error rate, Type I error (false positive rate), and Type II error (false negative rate).
    Displays:
        - Confusion matrix plot using matplotlib.
    Returns:
        None: The function prints and plots results, but does not return any value.
        (Commented-out code is provided for returning metrics if needed.)
    """
    # Get predictions using cross-validation (at default threshold 0.5)
    y_pred_cv = cross_val_predict(model, X, y, cv=5)

    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred_cv)

    # Display confusion matrix
    print("CONFUSION MATRIX")
    print("=" * 30)
    print("Actual vs Predicted:")
    print(f"True Negatives (TN):  {cm[0,0]}")
    print(f"False Positives (FP): {cm[0,1]}")
    print(f"False Negatives (FN): {cm[1,0]}")
    print(f"True Positives (TP):  {cm[1,1]}")

    # Calculate error rates
    total = cm.sum()
    error_rate = (cm[0,1] + cm[1,0]) / total
    print(f"\nOverall Error Rate: {error_rate:.1%}")
    print(f"Type I Error (False Positive Rate): {cm[0,1]/cm[0].sum():.1%}")
    print(f"Type II Error (False Negative Rate): {cm[1,0]/cm[1].sum():.1%}")

    # Create figure/axis if not provided
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        show_plot = True
    else:
        show_plot = False

    # Visualize confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}\n(5-Fold Cross-Validation Predictions)')
    
    # Only show if we created our own figure
    if show_plot:
        plt.tight_layout()
        plt.show()

    # Return confusion matrix and metrics for further analysis
    # return {
    #    'confusion_matrix': cm,
    #    'error_rate': error_rate,
    #    'type_i_error': cm[0,1]/cm[0].sum(),
    #    'type_ii_error': cm[1,0]/cm[1].sum(),
    #    'predictions': y_pred_cv
    #}

def simple_knn_example(X, y, points_per_class=20, test_point=None):
    """
    Visualizes a simple k-Nearest Neighbors (k-NN) classification example using a 2D dataset.
    This function selects a subset of points from each class for visualization, plots the data along with a test point,
    and demonstrates the k-NN classification process (with k=5) by highlighting the nearest neighbors and showing the
    predicted class for the test point. The function also prints details about the nearest neighbors and the prediction.
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, 2), where each row is a data point with two features.
        y (np.ndarray): Array of class labels (0 or 1) of shape (n_samples,).
        points_per_class (int, optional): Number of points to sample from each class for visualization. Defaults to 20.
        test_point (np.ndarray, optional): A 2D array of shape (1, 2) representing the test point. If None, defaults to [[0, 0]].
    Returns:
        None: This function displays plots and prints information but does not return any value.
    """
    # First, let's create a subset for cleaner visualization
    np.random.seed(42)
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    # Sample `points_per_class` points from each class
    subset_0 = np.random.choice(class_0_indices, points_per_class, replace=False)
    subset_1 = np.random.choice(class_1_indices, points_per_class, replace=False)
    subset_indices = np.concatenate([subset_0, subset_1])

    X_subset = X[subset_indices]
    y_subset = y[subset_indices]

    # Choose a test point for demonstration
    if test_point is None:
        test_point = np.array([[0, 0]])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Show the subset with test point
    ax1 = axes[0]
    scatter1 = ax1.scatter(X_subset[:, 0], X_subset[:, 1], c=y_subset, 
                        cmap='viridis', s=100, alpha=0.8, edgecolors='black')
    ax1.scatter(test_point[0, 0], test_point[0, 1], c='red', s=200, marker='X', 
            edgecolors='black', linewidth=2, label='Test Point')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title(f'Dataset Subset ({points_per_class} points per class) + Test Point')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Show KNN with k=5
    k = 5
    ax2 = axes[1]

    # Calculate distances from test point to subset points
    distances = np.sqrt(np.sum((X_subset - test_point)**2, axis=1))
    k_indices = np.argsort(distances)[:k]

    # Plot all points in grey first
    ax2.scatter(X_subset[:, 0], X_subset[:, 1], c='lightgrey', s=100, 
            alpha=0.5, edgecolors='black')

    # Highlight the k nearest neighbors with their true colors
    k_neighbors_classes = y_subset[k_indices]
    k_neighbors_points = X_subset[k_indices]
    ax2.scatter(k_neighbors_points[:, 0], k_neighbors_points[:, 1], 
            c=k_neighbors_classes, cmap='viridis', s=120, alpha=0.9, 
            edgecolors='red', linewidth=2)

    # Make prediction based on majority vote
    prediction = 1 if np.sum(k_neighbors_classes) > k/2 else 0
    prediction_color = 'yellow' if prediction == 1 else 'purple'

    # Plot test point with predicted color
    ax2.scatter(test_point[0, 0], test_point[0, 1], c=prediction_color, s=200, 
            marker='X', edgecolors='black', linewidth=3, label=f'Test Point (Predicted: Class {prediction})')

    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title(f'KNN with k={k}\nRed borders = {k} nearest neighbors')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Show the detailed process
    print(f"\nNearest neighbor classes: {k_neighbors_classes}")
    print(f"Class 0 votes: {np.sum(k_neighbors_classes == 0)}")
    print(f"Class 1 votes: {np.sum(k_neighbors_classes == 1)}")
    print(f"Prediction: Class {prediction} (majority vote)")
    print(f"\nDistances to {k} nearest neighbors: {np.round(np.sort(distances)[:k], 3)}")

def find_optimal_intercept_and_margin(slope, X_data, y_data):
    """
    Finds the optimal intercept and margin for a linear separator with a given slope.
    Given a slope, this function computes the intercept that maximizes the minimum margin
    between two classes in a 2D dataset. It assumes class 0 should be below the line and
    class 1 should be above the line.
    Args:
        slope (float): The slope of the separating line.
        X_data (np.ndarray): 2D array of shape (n_samples, 2) containing the data points.
        y_data (np.ndarray): 1D array of shape (n_samples,) containing class labels (0 or 1).
    Returns:
        tuple: A tuple containing:
            - optimal_intercept (float): The intercept that maximizes the minimum margin.
            - margin (float): The maximum achievable margin for the given slope.
            - max_class0_intercept (float): The largest intercept for class 0 points.
            - min_class1_intercept (float): The smallest intercept for class 1 points.
    Raises:
        ValueError: If the input arrays have incompatible shapes or if there is no valid margin.
    """
    # For each class, find the range of intercepts that would separate it
    class_intercepts = []
    
    for class_val in [0, 1]:
        mask = y_data == class_val
        class_points = X_data[mask]
        
        # For line y = slope * x + intercept, rearrange to intercept = y - slope * x
        intercepts_for_class = class_points[:, 1] - slope * class_points[:, 0]
        class_intercepts.append(intercepts_for_class)
    
    # For separation: Class 0 should be below line, Class 1 should be above line
    # So intercept should be: max(class_0_intercepts) < intercept < min(class_1_intercepts)
    max_class0_intercept = np.max(class_intercepts[0])
    min_class1_intercept = np.min(class_intercepts[1])
    
    # The optimal intercept is in the middle of this range
    optimal_intercept = (max_class0_intercept + min_class1_intercept) / 2
    
    # The margin is half the distance between these boundary intercepts
    margin = (min_class1_intercept - max_class0_intercept) / (2 * np.sqrt(1 + slope**2))
    
    return optimal_intercept, margin, max_class0_intercept, min_class1_intercept

def plot_optimal_separators(X_data, y_data, show_margins=False):
    """
    Plots optimal linear decision boundaries (separators) for a 2D dataset, optionally visualizing the margins for each separator.
    The function draws multiple separator lines with different slopes, computes their optimal intercepts and margins using
    `find_optimal_intercept_and_margin`, and displays the data points colored by class. If `show_margins` is True, the margin
    boundaries and the margin area for each separator are also visualized.
    Args:
        X_data (np.ndarray): 2D array of shape (n_samples, 2) containing the standardized feature data.
        y_data (np.ndarray): 1D array of shape (n_samples,) containing binary class labels (0 or 1).
        show_margins (bool, optional): If True, displays the margin boundaries and fills the margin area for each separator.
            Defaults to False.
    Returns:
        None: This function displays a matplotlib plot and does not return any value.
    """
    # Define slopes and colors
    slopes = [-0.35, 0., 0.3]
    separator_colors = ['blue', 'orange', 'green']
    separator_labels = ['Separator 1', 'Separator 2', 'Separator 3']
    
    # Set up the plot
    plt.figure(figsize=(8, 6))
    
    # Plot the data points
    colors = ['red', 'blue']
    for class_val in [0, 1]:
        mask = y_data == class_val
        plt.scatter(X_data[mask, 0], X_data[mask, 1], c=colors[class_val], 
                    label=f'Class {class_val}', alpha=0.7, s=50, 
                    edgecolors='black', linewidth=0.5)
    
    x_range = np.linspace(X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5, 100)
    
    # Plot each separator
    for i, (slope, color, label) in enumerate(zip(slopes, separator_colors, separator_labels)):
        
        optimal_intercept, margin, bound_0, bound_1 = find_optimal_intercept_and_margin(slope, X_data, y_data)
        
        # Update label to include margin if showing margins
        if show_margins:
            label = f'{label} (Margin: {margin:.3f})'
        
        # Plot main separator line
        y_line = slope * x_range + optimal_intercept
        plt.plot(x_range, y_line, color=color, linewidth=3, 
                 label=label, alpha=0.8)
        
        # Plot margins if requested
        if show_margins:
            # Plot margin boundaries
            y_upper = slope * x_range + bound_1
            y_lower = slope * x_range + bound_0
            
            plt.plot(x_range, y_upper, color=color, 
                     linestyle='--', alpha=0.6, linewidth=1.5)
            plt.plot(x_range, y_lower, color=color, 
                     linestyle='--', alpha=0.6, linewidth=1.5)
            
            # Fill the margin area
            plt.fill_between(x_range, y_lower, y_upper, alpha=0.1, color=color)
    
    # Set labels and title
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    
    if show_margins:
        plt.title('Optimal Decision Boundaries with Margin Visualization')
    else:
        plt.title('Optimal Decision Boundaries for Different Slopes')
    
    plt.legend()#bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_svm(X, y, ax=None, svm_model=None, kernel_name='Linear Kernel'):
    """
    Visualizes the decision boundary of a Support Vector Machine (SVM) classifier on 2D data.
    This function fits an SVM (if not provided), plots the data points, the SVM decision boundary,
    margins, and optionally the support vectors. It also displays the classification accuracy in the plot title.
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, 2).
        y (np.ndarray): Target labels of shape (n_samples,).
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        svm_model (sklearn.svm.SVC, optional): Pre-trained SVM model. If None, a linear SVM is fitted to the data.
        kernel_name (str, optional): Name of the kernel to display in the plot title. Defaults to 'Linear Kernel'.
    Returns:
        None
    """

    def plot_svc_decision_function(model, ax=None, plot_support=False):
        """Plot the decision function for a 2D SVC"""
        if ax is None:
            ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # create grid to evaluate model
        x = np.linspace(xlim[0], xlim[1], 50)
        y = np.linspace(ylim[0], ylim[1], 50)
        Y, X = np.meshgrid(y, x)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        P = model.decision_function(xy).reshape(X.shape)
        
        # plot decision boundary and margins
        ax.contour(X, Y, P, colors='k',
                levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
        
        # plot support vectors
        if plot_support:
            ax.scatter(model.support_vectors_[:, 0],
                    model.support_vectors_[:, 1],
                    s=300, linewidth=1, facecolors='none', edgecolors='black')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if svm_model is None:
        # Fit SVM with linear kernel
        svm_model = SVC(kernel='linear', C=1E10, random_state=42)
        svm_model.fit(X, y)

    # Make predictions and calculate metrics
    y_pred = svm_model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    # Create figure only if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        show_plot = True
    else:
        show_plot = False

    # Plot the data points
    colors = ['red', 'blue']
    for class_val in [0, 1]:
        mask = y == class_val
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[class_val], 
                    label=f'Class {class_val}', alpha=0.7, s=50, 
                    edgecolors='black', linewidth=0.5)

    # Plot SVM decision function
    plot_svc_decision_function(svm_model, ax=ax)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'SVM with {kernel_name}\nAccuracy: {accuracy:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Only show plot if we created the figure
    if show_plot:
        plt.show()

    # return svm_model, accuracy

def calculate_results_svm(X, y, svm_model, C_val):
    """
    Calculates SVM classification results and metrics for a given dataset and trained SVM model.
    Args:
        X (array-like): Feature matrix used for prediction.
        y (array-like): True labels corresponding to X.
        svm_model (sklearn.svm.SVC): Trained SVM model.
        C_val (float): Regularization parameter value used in the SVM.
    Returns:
        dict: A dictionary containing:
            - 'C' (float): The regularization parameter value.
            - 'accuracy' (float): Classification accuracy on the provided data.
            - 'margin' (float): The margin of the SVM decision boundary.
            - 'n_support_vectors' (int): Number of support vectors used by the model.
    """
    # Make predictions and calculate metrics
    y_pred = svm_model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    # Calculate margin
    w = svm_model.coef_[0]
    margin = 1 / np.linalg.norm(w)

    return {
        'C': C_val,
        'accuracy': accuracy,
        'margin': margin,
        'n_support_vectors': len(svm_model.support_)
    }

### Deep Learning

def interactive_relu_demo():
    """Interactive demonstration of how ReLU combinations create different functions"""
    
    # Create x-axis
    x = torch.linspace(-2, 2, 1000)
    
    def plot_relu_combination(weight1=1.0, bias1=0.0, weight2=-0.5, bias2=0.5, weight3=0.3, bias3=-1.0):
        """Plot combination of 3 ReLU functions with adjustable parameters"""
        
        # Compute individual ReLU functions
        relu1 = weight1 * torch.relu(x + bias1)
        relu2 = weight2 * torch.relu(x + bias2)  
        relu3 = weight3 * torch.relu(x + bias3)
        
        # Combine all ReLUs
        combined = relu1 + relu2 + relu3
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Plot individual ReLUs
        plt.subplot(1, 2, 1)
        plt.plot(x, relu1, 'r-', linewidth=2, label=f'ReLU1: {weight1:.1f}*ReLU(x + {bias1:.1f})')
        plt.plot(x, relu2, 'g-', linewidth=2, label=f'ReLU2: {weight2:.1f}*ReLU(x + {bias2:.1f})')
        plt.plot(x, relu3, 'b-', linewidth=2, label=f'ReLU3: {weight3:.1f}*ReLU(x + {bias3:.1f})')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.title('Individual ReLU Functions')
        plt.xlabel('x')
        plt.ylabel('ReLU(x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-3, 3)
        
        # Plot combined function
        plt.subplot(1, 2, 2)
        plt.plot(x, combined, 'purple', linewidth=3, label='Combined Function')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.title('Combined ReLU Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-3, 3)
        
        plt.tight_layout()
        plt.show()
        
    # Create interactive sliders
    weight_layout = Layout(width='300px')
    bias_layout = Layout(width='300px')
    
    interact(plot_relu_combination,
             weight1=FloatSlider(min=-2, max=2, step=0.1, value=1.0, description='Weight 1:', layout=weight_layout),
             bias1=FloatSlider(min=-2, max=2, step=0.1, value=0.0, description='Bias 1:', layout=bias_layout),
             weight2=FloatSlider(min=-2, max=2, step=0.1, value=1.0, description='Weight 2:', layout=weight_layout),
             bias2=FloatSlider(min=-2, max=2, step=0.1, value=0.5, description='Bias 2:', layout=bias_layout),
             weight3=FloatSlider(min=-2, max=2, step=0.1, value=1.0, description='Weight 3:', layout=weight_layout),
             bias3=FloatSlider(min=-2, max=2, step=0.1, value=-0.5, description='Bias 3:', layout=bias_layout))
    
def simple_sine_dataset(N=10):
    x_train = torch.linspace(0, 2*np.pi, N).view(-1, 1)
    y_train = torch.sin(x_train)    
    return x_train, y_train

def approximate_function(x_train, y_train):
    
    # Number of ReLUs needed
    n_relus = x_train.shape[0] - 1
    
    # Dense x-axis for smooth plotting
    x = torch.linspace(torch.min(x_train), torch.max(x_train), 1000)
    
    ## COMPUTE RELU ACTIVATIONS
    # Set bias terms to "activate" ReLUs at training points
    b = -x_train[:-1]
    
    # Compute ReLU activations: ReLU(x + bias)
    relu_acts = torch.zeros((n_relus, x.shape[0]))
    for i_relu in range(n_relus):
        relu_acts[i_relu, :] = torch.relu(x + b[i_relu])
    
    ## COMBINE RELU ACTIVATIONS
    # Calculate weights to match target function slopes
    combination_weights = torch.zeros((n_relus,))
    
    prev_slope = 0
    for i in range(n_relus):
        delta_x = x_train[i+1] - x_train[i]
        slope = (y_train[i+1] - y_train[i]) / delta_x
        combination_weights[i] = slope - prev_slope
        prev_slope = slope
    
    # Final approximation: weighted sum of ReLUs
    y_hat = combination_weights @ relu_acts
    
    return y_hat, relu_acts, x, combination_weights

def plot_function_approximation(x_train, y_train, x, relu_acts, y_hat, combination_weights):
    
    # Original function for comparison
    y_true = torch.sin(x)

    # Set consistent axis limits for all plots
    x_lim = [float(torch.min(x))-0.2, float(torch.max(x))+0.2]
    y_lim = [-2, 2]  # Accommodate both sine function and individual ReLUs
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Original function
    axes[0,0].plot(x, y_true, 'b-', linewidth=2, label='True sine function')
    axes[0,0].scatter(x_train.flatten(), y_train.flatten(), color='red', s=50, zorder=5, label='Training points')
    axes[0,0].set_title('Target Function: sin(x)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Individual ReLU functions
    axes[0,1].set_title('First 3 Individual ReLU Components')
    colors = plt.cm.viridis(np.linspace(0, 1, relu_acts.shape[0]))
    for i in range(min(3, relu_acts.shape[0])):  # Show first 5 ReLUs
        axes[0,1].plot(x, combination_weights[i] * relu_acts[i], color=colors[i], alpha=0.7, label=f'ReLU {i+1}')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xlim(x_lim)
    axes[0,1].set_ylim(y_lim)
    axes[0,0].set_xlim(x_lim)
    axes[0,0].set_ylim(y_lim)
    
    # Plot 3: Cumulative approximation
    axes[1,0].set_title('Building the Approximation')
    axes[1,0].plot(x, y_true, 'b-', linewidth=2, alpha=0.5, label='Target')
    
    # Show cumulative sum of ReLUs
    cumulative = torch.zeros_like(x)
    for i in range(min(3, relu_acts.shape[0])):
        cumulative += combination_weights[i] * relu_acts[i]
        axes[1,0].plot(x, cumulative, '--', alpha=0.8, label=f'Sum of first {i+1} ReLUs')
    
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xlim(x_lim)
    axes[1,0].set_ylim(y_lim)
    
    # Plot 4: Final approximation
    axes[1,1].plot(x, y_true, 'b-', linewidth=2, label='True function')
    axes[1,1].plot(x, y_hat, 'r--', linewidth=2, label='ReLU approximation')
    axes[1,1].scatter(x_train.flatten(), y_train.flatten(), color='red', s=50, zorder=5)
    axes[1,1].set_title('Final Approximation')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xlim(x_lim)
    axes[1,1].set_ylim(y_lim)
    
    plt.tight_layout()
    plt.show()

class SimpleMLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, output_size=1, num_layers=1):
        """
        Simple Neural Network with ReLU activation
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of hidden layers
            output_size (int): Number of output neurons
            num_layers (int): Number of hidden layers
        """
        super(SimpleMLP, self).__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Input to first hidden layer
        if num_layers > 0:
            self.layers.append(nn.Linear(input_size, hidden_size))
            
            # Additional hidden layers
            for i in range(1, num_layers):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            
            # Output layer
            self.layers.append(nn.Linear(hidden_size, output_size))
        else:
            # Direct input to output
            self.layers.append(nn.Linear(input_size, output_size))
        
        # Store intermediate representations
        self.representations = {}
    
    def forward(self, x):
        """Forward pass with ReLU activation"""
        # Store input representation
        self.representations['input'] = x
        
        current_x = x
        
        # Pass through hidden layers with ReLU
        for i in range(self.num_layers):
            current_x = F.relu(self.layers[i](current_x))
            self.representations[f'hidden{i+1}'] = current_x
        
        # Output layer (no activation)
        if self.num_layers > 0:
            output = self.layers[-1](current_x)
        else:
            output = self.layers[0](current_x)
        
        self.representations['output'] = output
        
        return output
    
    def get_representations(self, x):
        """Get intermediate representations for visualization"""
        with torch.no_grad():
            output = self.forward(x)
            # Convert to numpy for easy plotting
            representations = {}
            for key, value in self.representations.items():
                representations[key] = value.numpy()
            return representations

def fit_model(model, x_train, y_train, epochs=2000, lr=0.01, type='regression'):
    """Train the neural network - we'll explore this process in detail later!
    
    Args:
        type: 'classification' uses CrossEntropyLoss, 'regression' uses MSELoss
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Choose loss function based on problem type
    if type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    losses = []
    for epoch in range(epochs):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return losses

def create_nested_circles_data(n_samples=1000, noise=0.1, random_state=42):
    """Create nested circles dataset using sklearn"""
    torch.manual_seed(42)
    np.random.seed(42)
    X, y = make_circles(n_samples=n_samples, 
                        noise=noise, 
                        factor=0.3,  # ratio of inner to outer circle
                        random_state=random_state)
    
    # Visualize the data
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue']
    for i in range(2):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], 
                    c=colors[i], alpha=0.6, 
                    label=f'Class {i}')
    plt.title('Nested Circles Dataset')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()
    return torch.FloatTensor(X), torch.LongTensor(y)

def visualize_decision_boundary(model, X, y):
    """Visualize the decision boundary learned by the model"""
    # Create a mesh
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        outputs = model(mesh_points)
        predictions = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
    
    predictions = predictions.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, predictions, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='Probability of Class 1')
    
    # Plot data points
    colors = ['red', 'blue']
    for i in range(2):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], 
                    c=colors[i], alpha=0.7, 
                    label=f'Class {i}', edgecolors='black', linewidth=0.5)
    
    plt.title('Neural Network Decision Boundary')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.axis('equal')
    plt.show()

def visualize_layer_transformations(model, X, y, n_samples=500):
    """Visualize how data transforms through network layers using MDS"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    # Sample subset for clarity
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    # Get representations from each layer
    representations = model.get_representations(X_sample)
    
    # Set up the plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    layer_names = ['Input Space', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer']
    layer_keys = ['input', 'hidden1', 'hidden2', 'output']
    
    colors = ['red', 'blue']
    
    for i, (layer_name, layer_key) in enumerate(zip(layer_names, layer_keys)):
        data = representations[layer_key]
        
        # If more than 2 dimensions, use MDS to reduce to 2D
        if data.shape[1] > 2:
            mds = MDS(n_components=2, random_state=42)
            data_2d = mds.fit_transform(data)
        else:
            data_2d = data
            
        # Plot each class
        for class_idx in range(2):
            mask = y_sample == class_idx
            axes[i].scatter(data_2d[mask, 0], data_2d[mask, 1], 
                           c=colors[class_idx], alpha=0.6, 
                           label=f'Class {class_idx}', s=20)
        
        axes[i].set_title(f'{layer_name}\n({data.shape[1]}D → 2D via MDS)' if data.shape[1] > 2 
                         else f'{layer_name}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Make axes equal for input space
        if i == 0:
            axes[i].axis('equal')
    
    plt.suptitle('How Neural Networks Transform Data Through Layers', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_multiple_epochs(X, y_true, epoch_params):
    """Plot multiple epochs side by side"""
    n_plots = len(epoch_params)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    for i, (epoch, weight, bias, loss) in enumerate(epoch_params):
        axes[i].scatter(X, y_true, alpha=0.6, color='blue', label='Data')
        x_line = np.linspace(X.min(), X.max(), 100)
        y_line = weight * x_line + bias
        axes[i].plot(x_line, y_line, 'r-', linewidth=2, 
                    label=f'Model (epoch {epoch})')
        
        axes[i].set_title(f'Epoch {epoch}\nm={weight:.2f}, b={bias:.2f}\nLoss={loss:.2f}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Continued Gradient Descent Training', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_single_epoch(X, y_true, epoch, weight, bias, loss):
    """Plot the line fit for a single epoch"""
    plt.figure(figsize=(6, 6))
    
    # Plot data and current line
    plt.scatter(X, y_true, alpha=0.6, color='blue', label='Data', s=50)
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = weight * x_line + bias
    plt.plot(x_line, y_line, 'r-', linewidth=3, label=f'Model (epoch {epoch})')
    
    # When plotting each epoch
    y_pred_current = bias + weight * X
    current_loss_display = np.mean((y_true - y_pred_current) ** 2)  # MSE for display

    plt.title(f'Epoch {epoch}\nSlope (m) = {weight:.3f}, Intercept (b) = {bias:.3f}\nLoss = {current_loss_display:.3f}', 
              fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Simple linear model
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        # Initialize with random weights - this is where we start!
        nn.init.uniform_(self.linear.weight, -3, -1)
        nn.init.uniform_(self.linear.bias, 5, 6)
    
    def forward(self, x):
        return self.linear(x)
    
def plot_decision_boundary_evolution(model, snapshots, epochs_to_show, X_data, y_data):
    """
    Plot how decision boundary evolves during training
    """
    fig, axes = plt.subplots(1, len(epochs_to_show), figsize=(4*len(epochs_to_show), 4))
    
    # Handle case where there's only one subplot
    if len(epochs_to_show) == 1:
        axes = [axes]
    
    for idx, epoch in enumerate(epochs_to_show):
        ax = axes[idx]
        
        # Save current model state
        original_state = model.state_dict()
        
        # Load weights from snapshot
        model_state = {}
        for name, param in model.named_parameters():
            model_state[name] = snapshots[epoch]['weights'][name]
        
        # Update model with snapshot weights
        model.load_state_dict(model_state, strict=False)
        
        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
        y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Get predictions on mesh
        mesh_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        model.eval()
        with torch.no_grad():
            raw_output = model(mesh_points)
            # Convert to probabilities
            if raw_output.shape[1] == 1:  # Binary classification with single output
                probs = torch.sigmoid(raw_output.squeeze())
            else:  # Multi-class with softmax
                probs = torch.softmax(raw_output, dim=1)[:, 1]  # Probability of class 1
            
            Z = probs.numpy()
        
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        contour = ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        
        # Plot data points
        colors = ['red', 'blue']
        for i in range(2):
            mask = y_data == i
            ax.scatter(X_data[mask, 0], X_data[mask, 1], 
                      c=colors[i], alpha=0.8, s=30, edgecolors='black')
        
        ax.set_xlabel('X₁', fontsize=12)
        ax.set_ylabel('X₂', fontsize=12)
        ax.set_title(f'Epoch {epoch}\nLoss = {snapshots[epoch]["loss"]:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Restore original model state
        model.load_state_dict(original_state)
    
    plt.tight_layout()
    plt.suptitle('Neural Network Training: Decision Boundary Evolution', 
                 fontsize=16, fontweight='bold', y=1.05)
    plt.show()

def plot_training_loss(losses, test_losses=None, epochs_to_show=None, title='Neural Network Training: Loss Decreases Over Time'):

    # Plot the training loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(losses)), losses, 'b-', linewidth=2, label='Training Loss')
    if test_losses is not None:
        plt.plot(range(len(test_losses)), test_losses, 'r--', linewidth=2, label='Test Loss')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (Binary Cross-Entropy)', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Add vertical lines at the epochs we visualized
    if epochs_to_show is not None:
        colors_lines = ['orange', 'green', 'purple', 'red']
        for epoch, color in zip(epochs_to_show, colors_lines):
            if epoch < len(losses):
                plt.axvline(x=epoch, color=color, linestyle='--', alpha=0.7, linewidth=2)
                plt.text(epoch-5, losses[epoch], f'Epoch {epoch}', 
                        rotation=90, fontsize=10, ha='center', color=color, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

def compare_learning_rates(dataloader, learning_rates, total_epochs):
    """
    Compare different learning rates by training a model for each rate
    and storing the losses.
    """
    # Dictionary to store losses for each learning rate
    all_losses = {}

    for lr in learning_rates:
        
        # Reinitialize model for each learning rate
        model = SimpleMLP(input_size=2, hidden_size=20, output_size=1, num_layers=2)
        
        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # Storage for this learning rate
        losses = []
        
        # Training loop
        for epoch in range(total_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0
            
            # Process all batches in this epoch
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimization
                if epoch > 0:  # Skip optimization for epoch 0
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Average loss for this epoch
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
        
        # Store results
        all_losses[lr] = losses

    # Plot comparison of learning rates
    plt.figure(figsize=(12, 6))

    colors = ['blue', 'green', 'orange', 'red']
    labels = [f'LR = {lr}' for lr in learning_rates]

    for i, lr in enumerate(learning_rates):
        plt.plot(range(len(all_losses[lr])), all_losses[lr], 
                color=colors[i], linewidth=2, label=labels[i])

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (Binary Cross-Entropy)', fontsize=14)
    plt.title('Learning Rate Comparison: How Step Size Affects Convergence', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, total_epochs)
    plt.tight_layout()
    plt.show()
    
    return all_losses

def prepare_mnist(mnist, subset_size=10000):
    # Use a subset of MNIST for faster training
    X, y = mnist.data, mnist.target.astype(int)

    # Take only 10,000 samples (about 14% of full dataset)
    subset_size = 10000
    indices = np.random.choice(len(X), subset_size, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]

    # Split the subset
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset, test_size=0.2, random_state=42, stratify=y_subset
    )

    # Reshape for CNN (batch_size, channels, height, width)
    # MNIST is grayscale so channels = 1
    X_train_cnn = X_train.reshape(-1, 1, 28, 28) / 255.0  # Normalize to [0,1]
    X_test_cnn = X_test.reshape(-1, 1, 28, 28) / 255.0

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_cnn)
    X_test_tensor = torch.FloatTensor(X_test_cnn)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    print(f"Using subset: {subset_size:,} total samples")
    print(f"Training set: {X_train_tensor.shape[0]:,} images")
    print(f"Test set: {X_test_tensor.shape[0]:,} images")
    print(f"Reduction: {subset_size/len(X)*100:.1f}% of original dataset")

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_conv_layers=2, conv_channels=[32, 64], 
                 kernel_size=3, padding=1, pool_size=2, pool_stride=2,
                 num_fc_layers=2, fc_hidden_size=128, output_size=10, 
                 input_height=28, input_width=28):
        """
        Simple CNN with configurable architecture
        
        Args:
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            num_conv_layers (int): Number of convolutional layers
            conv_channels (list): Number of channels for each conv layer
            kernel_size (int): Convolution kernel size
            padding (int): Padding for convolutions
            pool_size (int): Max pooling kernel size
            pool_stride (int): Max pooling stride
            num_fc_layers (int): Number of fully connected layers
            fc_hidden_size (int): Hidden size for FC layers
            output_size (int): Number of output classes
            input_height (int): Height of input images
            input_width (int): Width of input images
        """
        super(SimpleCNN, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.input_height = input_height
        self.input_width = input_width
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        # Build conv layers
        in_channels = input_channels
        for i in range(num_conv_layers):
            out_channels = conv_channels[i] if i < len(conv_channels) else conv_channels[-1]
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            self.pool_layers.append(nn.MaxPool2d(pool_size, pool_stride))
            in_channels = out_channels
        
        # Calculate the size after conv layers
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        
        if num_fc_layers > 0:
            # First FC layer (after flattening)
            self.fc_layers.append(nn.Linear(self.conv_output_size, fc_hidden_size))
            
            # Additional hidden FC layers
            for i in range(1, num_fc_layers):
                self.fc_layers.append(nn.Linear(fc_hidden_size, fc_hidden_size))
            
            # Output layer
            self.fc_layers.append(nn.Linear(fc_hidden_size, output_size))
        else:
            # Direct conv to output
            self.fc_layers.append(nn.Linear(self.conv_output_size, output_size))
        
        # Store intermediate representations
        self.representations = {}
    
    def _calculate_conv_output_size(self):
        """Calculate the output size after all conv and pooling layers"""
        with torch.no_grad():
            # Create a dummy input to calculate output size
            dummy_input = torch.zeros(1, 1 if not hasattr(self, 'input_channels') else self.input_channels, 
                                    self.input_height, self.input_width)
            
            x = dummy_input
            for i in range(self.num_conv_layers):
                x = self.pool_layers[i](F.relu(self.conv_layers[i](x)))
            
            return x.view(1, -1).size(1)
    
    def forward(self, x):
        """Forward pass with ReLU activation"""
        # Store input representation
        self.representations['input'] = x
        
        current_x = x
        
        # Convolutional layers
        for i in range(self.num_conv_layers):
            current_x = F.relu(self.conv_layers[i](current_x))
            self.representations[f'conv{i+1}'] = current_x
            
            current_x = self.pool_layers[i](current_x)
            self.representations[f'pool{i+1}'] = current_x
        
        # Flatten for FC layers
        flattened = current_x.view(current_x.size(0), -1)
        self.representations['flattened'] = flattened
        
        current_x = flattened
        
        # Fully connected layers (except output)
        for i in range(self.num_fc_layers):
            current_x = F.relu(self.fc_layers[i](current_x))
            self.representations[f'fc{i+1}'] = current_x
        
        # Output layer (no activation)
        if self.num_fc_layers > 0:
            output = self.fc_layers[-1](current_x)
        else:
            output = self.fc_layers[0](current_x)
        
        self.representations['output'] = output
        
        return output
    
    def get_representations(self, x):
        """Get intermediate representations for visualization"""
        with torch.no_grad():
            output = self.forward(x)
            # Convert to numpy for easy plotting
            representations = {}
            for key, value in self.representations.items():
                if value.dim() > 2:  # For conv layers, keep shape info
                    representations[key] = value.cpu().numpy()
                else:  # For FC layers, flatten if needed
                    representations[key] = value.cpu().numpy()
            return representations

    def get_feature_maps(self, x, layer_name):
        """Get feature maps from a specific convolutional layer"""
        with torch.no_grad():
            self.forward(x)
            if layer_name in self.representations:
                return self.representations[layer_name].cpu().numpy()
            else:
                available_layers = list(self.representations.keys())
                raise ValueError(f"Layer {layer_name} not found. Available layers: {available_layers}")

    def summary(self):
        """Print network architecture summary"""
        print("SimpleCNN Architecture:")
        print(f"Input shape: ({self.input_height}, {self.input_width})")
        print("\nConvolutional Layers:")
        for i, conv in enumerate(self.conv_layers):
            print(f"  Conv{i+1}: {conv}")
            print(f"  Pool{i+1}: {self.pool_layers[i]}")
        
        print(f"\nFlattened size: {self.conv_output_size}")
        print("\nFully Connected Layers:")
        for i, fc in enumerate(self.fc_layers):
            print(f"  FC{i+1}: {fc}")

def extract_and_plot_feature_maps(model, sample_image, sample_label, category_name='Digit'):

    # Get feature maps using built-in method
    model.eval()
    with torch.no_grad():
        conv1_features = model.get_feature_maps(sample_image, 'conv1')  # Shape: (1, 32, 28, 28)
        conv2_features = model.get_feature_maps(sample_image, 'conv2')  # Shape: (1, 64, 14, 14)

    # Convert for plotting
    conv1_maps = conv1_features[0]  # (32, 28, 28)
    conv2_maps = conv2_features[0]  # (64, 14, 14)
    original_img = sample_image[0, 0].cpu().numpy()  # (28, 28)

    print(f"Analyzing {category_name}: {sample_label}")
    print(f"Conv1 feature maps shape: {conv1_maps.shape}")
    print(f"Conv2 feature maps shape: {conv2_maps.shape}")

    # Plot original image and feature maps in 2 levels
    fig = plt.figure(figsize=(15, 8))

    # TOP LEVEL: Original + Conv1 feature maps
    # Original image
    plt.subplot(2, 6, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title(f'Original\n{category_name}: {sample_label}', fontsize=12, fontweight='bold')
    plt.axis('off')

    # First conv layer feature maps (show first 5)
    for i in range(5):
        plt.subplot(2, 6, i + 2)
        plt.imshow(conv1_maps[i], cmap='viridis')
        plt.title(f'Conv1\nFilter {i+1}', fontsize=10)
        plt.axis('off')

    # BOTTOM LEVEL: Conv2 feature maps after pooling
    for i in range(5):
        plt.subplot(2, 6, i + 8)
        plt.imshow(conv2_maps[i], cmap='plasma')
        plt.title(f'Conv2\nFilter {i+1}', fontsize=10)
        plt.axis('off')

    # Add separator text in the remaining spot
    plt.subplot(2, 6, 7)
    plt.text(0.5, 0.5, 'After\nPooling', ha='center', va='center', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.suptitle('Feature Maps: How CNN "Sees" the Image at Different Layers', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_cnn_representations(model, X_test_tensor, y_test_tensor, num_samples=500):
    # Use a subset of test data for visualization (500 samples for speed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_samples = 500
    indices = np.random.choice(len(X_test_tensor), n_samples, replace=False)
    sample_data = X_test_tensor[indices].to(device)
    sample_labels = y_test_tensor[indices].cpu().numpy()

    # Get representations
    representations = model.get_representations(sample_data)

    print(f"Analyzing {n_samples} test samples")
    print(f"Flattened conv features: {representations['flattened'].shape}")
    print(f"FC1 representations: {representations['fc1'].shape}")
    print(f"FC2 representations: {representations['fc2'].shape}")

    # Standardize and apply MDS to all three representations
    scaler_flat = StandardScaler()
    flat_scaled = scaler_flat.fit_transform(representations['flattened'])
    mds_flat = MDS(n_components=2, random_state=42, normalized_stress='auto')
    flat_2d = mds_flat.fit_transform(flat_scaled)

    scaler_fc1 = StandardScaler()
    fc1_scaled = scaler_fc1.fit_transform(representations['fc1'])
    mds_fc1 = MDS(n_components=2, random_state=42, normalized_stress='auto')
    fc1_2d = mds_fc1.fit_transform(fc1_scaled)

    scaler_fc2 = StandardScaler()
    fc2_scaled = scaler_fc2.fit_transform(representations['fc2'])
    mds_fc2 = MDS(n_components=2, random_state=42, normalized_stress='auto')
    fc2_2d = mds_fc2.fit_transform(fc2_scaled)

    # Create three-panel visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    colors = plt.cm.tab10(np.arange(10))

    # Flattened conv representations
    for digit in range(10):
        mask = sample_labels == digit
        if np.any(mask):
            ax1.scatter(flat_2d[mask, 0], flat_2d[mask, 1], 
                    c=[colors[digit]], s=20, alpha=0.6, 
                    label=f'Digit {digit}')

    ax1.set_xlabel('MDS Dimension 1', fontsize=12)
    ax1.set_ylabel('MDS Dimension 2', fontsize=12)
    ax1.set_title('Flattened Conv Features\n(3136D → 2D, Raw Conv Output)', 
                fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # FC1 representations
    for digit in range(10):
        mask = sample_labels == digit
        if np.any(mask):
            ax2.scatter(fc1_2d[mask, 0], fc1_2d[mask, 1], 
                    c=[colors[digit]], s=20, alpha=0.6, 
                    label=f'Digit {digit}')

    ax2.set_xlabel('MDS Dimension 1', fontsize=12)
    ax2.set_ylabel('MDS Dimension 2', fontsize=12)
    ax2.set_title('FC1 Layer Representations\n(128D → 2D, Intermediate Features)', 
                fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # FC2 representations (pre-softmax logits)
    for digit in range(10):
        mask = sample_labels == digit
        if np.any(mask):
            ax3.scatter(fc2_2d[mask, 0], fc2_2d[mask, 1], 
                    c=[colors[digit]], s=20, alpha=0.6, 
                    label=f'Digit {digit}')

    ax3.set_xlabel('MDS Dimension 1', fontsize=12)
    ax3.set_ylabel('MDS Dimension 2', fontsize=12)
    ax3.set_title('FC2 Layer Representations\n(128D → 2D, Second Intermediate Features)', 
                fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Single legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc='center left')

    plt.tight_layout()
    plt.show()