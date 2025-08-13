import os
from datetime import datetime

import pandas as pd
import torch
from botorch.utils import draw_sobol_samples

from models import black_box
from optimization import qLogEHVI, qEI


def normalize_tensor(y: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to [0, 1] range using min-max normalization."""
    return (y - y.min()) / (y.max() - y.min() + 1e-8)


def save_data(
        x,
        file_path: str = "./result",
        file_name: str = "data",
        column_name: dict = None  # names of column, e.g: {"dict1": x[:,0], "dict2": x[:,1]...}
):
    """
    Save data to .csv.

    :param x: data you want to save
    :param file_path: file path
    :param file_name: file name
    :param column_name: # names of column, e.g: {"dict1": x[:,0], "dict2": x[:,1]...}
    :return: timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # guarantee path exist
    os.makedirs(file_path, exist_ok=True)
    # data type converting
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    # data saving
    x_df = pd.DataFrame(x_np, columns=column_name)
    x_df.to_csv(f"{file_path}/{timestamp}_{file_name}.csv", index=False)
    print(f"data has already saved in {file_path}/{timestamp}_{file_name}，time：{timestamp}")
    return timestamp  # return timestamp


def read_data(x_dim, y_dim, file_path: str, x_cols=None, y_cols=None):
    """
    Read training data from .csv. If no column name is given, the first column is taken by default:
        -input_dim column as X.
        -objective_dim column as Y.
    :param x_dim: dimension of X
    :param y_dim: dimension of Y
    :param file_path: file path
    :param x_cols: column names of training data
    :param y_cols: column names of training data
    :return: X, Y
    """
    df = pd.read_csv(file_path)

    if x_cols is None:
        x_cols = df.columns[:x_dim]
    if y_cols is None:
        y_cols = df.columns[x_dim:x_dim + y_dim]
    X_np = df[list(x_cols)].to_numpy()
    Y_np = df[list(y_cols)].to_numpy()
    X = torch.as_tensor(X_np)
    Y = torch.as_tensor(Y_np)
    return X, Y


def generate_initial_data(model_opt, bounds: torch.Tensor, n_init: int, d: int, device: torch.device) -> tuple:
    """
    use Sobol sequence to generate initial samples in given bounds, and use black_box func to get targets.

    model options:
        - 0: Provides random sampling for practical optimization.
        - 1: "zdt1" black-box model for 2-task BO.
        - 2: A linear transformation model of "zdt1" with formula:
           f₂ = 0.6 * f₁ + 10.8 + noise.

    :param model_opt: choose models
    :param bounds: shape [2, d]，Lower and upper
    :param n_init: number of initial samples
    :param d: number of input dimensions
    :param device: Device used for computation
    :return Tuple of tensors: (X_init, Y_init)
    """
    if n_init == 0:
        sobol_x = torch.zeros((n_init, d), device=device)
        y = torch.zeros((n_init, 2), device=device)
        return sobol_x, y
    sobol_x = draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=123).squeeze(1).to(device)
    # for using
    if model_opt == 0:
        return sobol_x, None
    # for evaluation
    if model_opt == 1:
        y = black_box.transfer_model_1(sobol_x, d)
    elif model_opt == 2:
        y = black_box.transfer_model_2(sobol_x, d)
    else:
        raise ValueError("model_opt must be 1 or 2")
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
    Run batch Bayesian Optimization using qLogEHVI.

    Args:
        model: Trained multi-objective GP model.
        bounds (torch.Tensor): Optimization variable bounds [2, d].
        train_y (torch.Tensor): Training objectives, shape [N, 2].
        batch_size (int): Target number of new samples to generate.
        mini_batch_size (int): BO internal batch size per iteration.
        device (torch.device): Target device (CPU/GPU).

    Returns:
        torch.Tensor: New candidate points, shape [batch_size, d].
    """
    X_next_tensor = torch.empty((0, bounds.shape[1]), dtype=torch.double).to(device)
    iteration = 0

    while X_next_tensor.shape[0] < batch_size:
        X_candidates, acq_val = qEI.optimize_acq_fun(
            model=model,
            train_y=train_y,
            bounds=bounds,
            batch_size=mini_batch_size
        )
        X_next_tensor = torch.cat((X_next_tensor, X_candidates), dim=0)
        iteration += 1
        # print(f"[BO] Iter {iteration}: Added {X_candidates.shape[0]} → total {X_next_tensor.shape[0]}")
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
    Run batch Bayesian Optimization using qLogEHVI.

    Args:
        model (ModelListGP): Trained multi-objective GP models.
        bounds (torch.Tensor): Optimization variable bounds [2, d].
        train_y (torch.Tensor): Training objectives, shape [N, 2].
        ref_point (list): Reference point in objective space, e.g., [0.5, 0.5].
        batch_size (int): Target number of new samples to generate.
        mini_batch_size (int): BO internal batch size per iteration.
        device (torch.device): Target device (CPU/GPU).

    Returns:
        torch.Tensor: New candidate points, shape [batch_size, d].
    """
    X_next_tensor = torch.empty((0, bounds.shape[1]), device=device)
    iteration = 0

    while X_next_tensor.shape[0] < batch_size:
        X_candidates, acq_val = qLogEHVI.optimize_acq_fun(
            model=model,
            train_y=train_y,
            bounds=bounds,
            ref_point=ref_point,
            batch_size=mini_batch_size,
            num_restarts=10,
            raw_samples=128,
        )
        X_next_tensor = torch.cat((X_next_tensor, X_candidates), dim=0)
        iteration += 1
        # print(f"[BO] Iter {iteration}: Added {X_candidates.shape[0]} → total {X_next_tensor.shape[0]}")
    return X_next_tensor[:batch_size, :]
