import numpy as np
import matplotlib.pyplot as plt

def normalize(data):
    # Normalize using training set statistics
    mean = data.mean()
    std = data.std()
    return (data - mean) / std

def train_linear_regression(X_train_norm, y_train, X_val_norm, y_val, learning_rate=0.01, n_iter=120):
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

    return w0, w1, w0_history, w1_history, rmse_train_history

def plot_rmse_loss_surface_with_arrow(X_norm, y, w0_hist, w1_hist, rmse_hist, w0_range_offset=70, w1_range_offset=70, grid_points=100):
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

def create_formula(degree):
    terms = ["Age_std"] + [f"I(Age_std**{i})" for i in range(2, degree + 1)]
    return "Ferritin ~ " + " + ".join(terms)

def downsample_history(*histories, k=5):
    """
    Downsample multiple histories (lists/arrays) to every k-th point.
    Returns the downsampled histories as a tuple.
    """
    
    return tuple([h[::k] for h in histories])

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