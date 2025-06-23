import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, HTML

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import statsmodels.formula.api as smf

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