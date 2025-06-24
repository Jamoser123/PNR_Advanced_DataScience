import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, islice
from sklearn import cluster

from IPython.display import display, HTML

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import statsmodels.formula.api as smf

import seaborn as sns
from matplotlib import cm

from sklearn.metrics import pairwise_distances
import gower
from sklearn.datasets import make_blobs, make_moons, make_circles

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import kmedoids



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
    Args:
        X_norm (np.ndarray): Normalized input feature array.
        y (np.ndarray): Target variable array.
        training_params (tuple): Tuple containing (w0_hist, w1_hist, rmse_hist), which are the histories of the intercept,
            slope, and RMSE values during training (e.g., from gradient descent).
        w0_range_offset (float, optional): Range offset for the intercept axis around the closed-form optimum. Default is 70.
        w1_range_offset (float, optional): Range offset for the slope axis around the closed-form optimum. Default is 70.
        grid_points (int, optional): Number of points in the grid for each parameter axis. Default is 100.
    Displays:
        - A 3D surface plot of the RMSE loss landscape.
        - The closed-form solution as a reference point.
        - The optimization path if provided.
        - A contour plot of the RMSE loss surface.
    """
    
    # Unpack training parameters
    w0_hist, w1_hist, rmse_hist = training_params
    
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
    ax.set_title('RMSE Loss Surface for Therapy Intensity')

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
    plt.title('RMSE Loss Contour (Therapy Intensity)')
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
