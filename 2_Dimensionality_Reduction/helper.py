import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, HTML

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.metrics import pairwise_distances
import gower


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