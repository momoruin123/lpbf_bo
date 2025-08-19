import os
from datetime import datetime

import pandas as pd
import torch
from botorch.utils import draw_sobol_samples

from models import black_box
from optimization import qLogEHVI, qEI


def normalize_tensor(y: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor to the [0, 1] range using min-max normalization.

    This transformation scales the input tensor such that the minimum value becomes 0
    and the maximum value becomes 1, with all other values proportionally scaled in between.
    A small epsilon (1e-8) is added to the denominator to avoid division by zero.

    Parameters:
        y: Input tensor to be normalized. Can be of any shape.

    Returns:
        torch.Tensor: Normalized tensor with the same shape as input, in [0, 1] range.
    """
    return (y - y.min()) / (y.max() - y.min() + 1e-8)


def save_data(
        x,
        file_path: str = "./result",
        file_name: str = "data",
        column_name: dict = None
) -> str:
    """
    Save data to a CSV file with a timestamp in the filename for uniqueness.

    Supports both PyTorch tensors and other array-like structures (converted to NumPy).
    Automatically creates the output directory if it doesn't exist.

    Parameters:
        x: Data to save. Can be a PyTorch tensor, NumPy array, or other array-like structure.
        file_path: Directory path where the CSV file will be saved (default: "./result").
        file_name: Base name for the CSV file (default: "data").
        column_name: Dictionary mapping column names to data columns. If provided,
            keys become CSV column names and values specify the corresponding data columns.
            Example: {"param1": x[:,0], "param2": x[:,1]}

    Returns:
        str: Timestamp string used in the filename (format: "YYYYMMDD_HHMMSS").
    """
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory if it doesn't exist
    os.makedirs(file_path, exist_ok=True)

    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = x

    # Create DataFrame with specified column names (or default if None)
    x_df = pd.DataFrame(x_np, columns=column_name)

    # Save to CSV
    full_path = f"{file_path}/{timestamp}_{file_name}.csv"
    x_df.to_csv(full_path, index=False)
    print(f"Data saved to {full_path}, timestamp: {timestamp}")

    return timestamp


def read_data(x_dim: int, y_dim: int, file_path: str, x_cols=None, y_cols=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read training data (inputs and targets) from a CSV file.

    Extracts input features (X) and target values (Y) from the CSV, with flexible column selection.
    If column names are not specified, uses the first `x_dim` columns for X and the next `y_dim` columns for Y.

    Parameters:
        x_dim: Number of input dimensions (features) to read.
        y_dim: Number of target dimensions (objectives) to read.
        file_path: Path to the CSV file containing the data.
        x_cols: Column names (list) for input features. If None, uses first `x_dim` columns.
        y_cols: Column names (list) for target values. If None, uses next `y_dim` columns after X.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - X: Input features tensor with shape (n_samples, x_dim)
            - Y: Target values tensor with shape (n_samples, y_dim)
    """
    # Read CSV file into DataFrame
    df = pd.read_csv(file_path)

    # Determine columns for X and Y if not specified
    if x_cols is None:
        x_cols = df.columns[:x_dim]
    if y_cols is None:
        y_cols = df.columns[x_dim : x_dim + y_dim]

    # Extract data and convert to PyTorch tensors
    X_np = df[list(x_cols)].to_numpy()
    Y_np = df[list(y_cols)].to_numpy()

    X = torch.as_tensor(X_np)
    Y = torch.as_tensor(Y_np)

    return X, Y


def generate_initial_data(
        model_opt: int,
        bounds: torch.Tensor,
        n_init: int,
        d: int,
        device: torch.device
) -> tuple[torch.Tensor, torch.Tensor or None]:
    """
    Generate initial samples using Sobol sequences and evaluate them with a specified black-box model.

    Sobol sequences provide low-discrepancy sampling, ensuring uniform coverage of the parameter space.
    The generated samples are evaluated using one of several predefined black-box models.

    Parameters:
        model_opt: Model selector specifying which black-box function to use:
            - 0: Return only samples (no target values, for practical optimization)
            - 1: "zdt1" black-box model (2-objective optimization)
            - 2: Linear transformation of "zdt1" (f₂ = 0.6*f₁ + 10.8 + noise)
        bounds: Parameter space boundaries. Tensor of shape (2, d) where:
            - bounds[0] = lower bounds for each dimension
            - bounds[1] = upper bounds for each dimension
        n_init: Number of initial samples to generate.
        d: Number of input dimensions.
        device: Device (CPU/GPU) to store the generated tensors.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - X_init: Initial samples with shape (n_init, d)
            - Y_init: Corresponding target values with shape (n_init, 2) (or None if model_opt=0)

    Raises:
        ValueError: If model_opt is not 0, 1, or 2.
    """
    # Handle edge case with zero initial samples
    if n_init == 0:
        sobol_x = torch.zeros((n_init, d), device=device)
        y = torch.zeros((n_init, 2), device=device)
        return sobol_x, y

    # Generate Sobol samples within the specified bounds
    sobol_x = draw_sobol_samples(
        bounds=bounds,
        n=n_init,
        q=1,
        seed=123
    ).squeeze(1).to(device)

    # Evaluate samples with the selected black-box model
    if model_opt == 0:
        # Return samples without target values (for practical use)
        return sobol_x, None
    elif model_opt == 1:
        # Use "zdt1" black-box model
        y = black_box.transfer_model_1(sobol_x, d)
    elif model_opt == 2:
        # Use linear transformation of "zdt1"
        y = black_box.transfer_model_2(sobol_x, d)
    else:
        raise ValueError("model_opt must be 0, 1, or 2")

    return sobol_x, y


def run_singletask_bo(
        model,
        bounds: torch.Tensor,
        train_y: torch.Tensor,
        batch_size: int,
        mini_batch_size: int,
        device: torch.device
) -> torch.Tensor:
    """
    Run batch Bayesian Optimization for single-objective problems using qLogEI.

    Generates a batch of candidate points by iteratively optimizing the qLogExpectedImprovement
    acquisition function, accumulating points until the desired batch size is reached.

    Parameters:
        model: Trained single-objective Gaussian Process model.
        bounds: Parameter space boundaries. Tensor of shape (2, d).
        train_y: Training target values. Tensor of shape (n_samples, 1).
        batch_size: Total number of candidate points to generate.
        mini_batch_size: Number of points to generate in each internal iteration.
        device: Device (CPU/GPU) for computations.

    Returns:
        torch.Tensor: Batch of candidate points with shape (batch_size, d).
    """
    # Initialize tensor to store candidate points
    X_next_tensor = torch.empty((0, bounds.shape[1]), dtype=torch.double).to(device)
    iteration = 0

    # Accumulate candidates until reaching the desired batch size
    while X_next_tensor.shape[0] < batch_size:
        # Optimize acquisition function to get next mini-batch
        X_candidates, acq_val = qEI.optimize_acq_fun(
            model=model,
            train_y=train_y,
            bounds=bounds,
            batch_size=mini_batch_size
        )

        # Append new candidates to the batch
        X_next_tensor = torch.cat((X_next_tensor, X_candidates), dim=0)
        iteration += 1
        # Optional: Uncomment to print progress
        # print(f"[BO] Iter {iteration}: Added {X_candidates.shape[0]} → total {X_next_tensor.shape[0]}")

    # Ensure we return exactly batch_size points
    return X_next_tensor[:batch_size, :]


def run_multitask_bo(
        model,
        bounds: torch.Tensor,
        train_y: torch.Tensor,
        ref_point: list,
        batch_size: int,
        mini_batch_size: int,
        device: torch.device
) -> torch.Tensor:
    """
    Run batch Bayesian Optimization for multi-objective problems using qLogEHVI.

    Generates a batch of candidate points by iteratively optimizing the qLogExpectedHypervolumeImprovement
    acquisition function, which balances convergence and diversity in multi-objective spaces.

    Parameters:
        model: Trained multi-objective Gaussian Process model (ModelListGP).
        bounds: Parameter space boundaries. Tensor of shape (2, d).
        train_y: Training target values. Tensor of shape (n_samples, m) where m is the number of objectives.
        ref_point: Reference point for hypervolume calculation. List of length m.
        batch_size: Total number of candidate points to generate.
        mini_batch_size: Number of points to generate in each internal iteration.
        device: Device (CPU/GPU) for computations.

    Returns:
        torch.Tensor: Batch of candidate points with shape (batch_size, d).
    """
    # Initialize tensor to store candidate points
    X_next_tensor = torch.empty((0, bounds.shape[1]), device=device)
    iteration = 0

    # Accumulate candidates until reaching the desired batch size
    while X_next_tensor.shape[0] < batch_size:
        # Optimize acquisition function to get next mini-batch
        X_candidates, acq_val = qLogEHVI.optimize_acq_fun(
            model=model,
            train_y=train_y,
            bounds=bounds,
            ref_point=ref_point,
            batch_size=mini_batch_size,
            num_restarts=10,
            raw_samples=128,
        )

        # Append new candidates to the batch
        X_next_tensor = torch.cat((X_next_tensor, X_candidates), dim=0)
        iteration += 1
        # Optional: Uncomment to print progress
        # print(f"[BO] Iter {iteration}: Added {X_candidates.shape[0]} → total {X_next_tensor.shape[0]}")

    # Ensure we return exactly batch_size points
    return X_next_tensor[:batch_size, :]