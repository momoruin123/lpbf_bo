import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


def build_model(x_old: torch.Tensor, y_old: torch.Tensor, x_new: torch.Tensor, y_new: torch.Tensor) -> ModelListGP:
    """
    Build a multitask Gaussian Process (GP) surrogate model for transfer learning between tasks.

    Constructs independent MultiTaskGPs for each objective metric, combining data from a source task (old)
    and a target task (new). Each model is fitted to maximize the marginal log likelihood.

    Parameters:
        x_old: Input parameters from the source task.
            Tensor of shape (K, M) where:
            - K = number of samples in the source task
            - M = number of input dimensions (process parameters)
        y_old: Output metrics from the source task.
            Tensor of shape (K, N) where:
            - N = number of target metrics (objectives)
        x_new: Input parameters from the target task.
            Tensor of shape (L, M) where:
            - L = number of samples in the target task
        y_new: Output metrics from the target task.
            Tensor of shape (L, N)

    Returns:
        ModelListGP: A collection of fitted MultiTaskGP models (one per objective metric)
    """
    input_dim = x_old.shape[1]  # Number of input parameters (M)
    target_dim = y_new.shape[1]  # Number of objective metrics (N)

    # List to store individual GP models for each objective
    models = []

    # Create a separate MultiTaskGP for each objective metric
    for i in range(target_dim):
        # Prepare source task data with task ID = 0
        x_old_processed, y_old_processed = prepare_data(x_old, y_old[:, i:i+1], task_id=0)
        # Prepare target task data with task ID = 1
        x_new_processed, y_new_processed = prepare_data(x_new, y_new[:, i:i+1], task_id=1)

        # Combine data from both tasks
        x_combined = torch.cat([x_old_processed, x_new_processed], dim=0)
        y_combined = torch.cat([y_old_processed, y_new_processed], dim=0)

        # Initialize MultiTaskGP for the current objective
        model = MultiTaskGP(
            train_X=x_combined,
            train_Y=y_combined,
            task_feature=-1,  # Last column contains task IDs
            output_tasks=[1],  # Focus on predicting the target task (ID=1)
            # Normalize input features (excluding task ID column)
            input_transform=Normalize(d=input_dim + 1, indices=list(range(input_dim))),
            # Standardize output values for stable training
            outcome_transform=Standardize(m=1)
        )
        models.append(model)

    # Combine all per-objective models into a single ModelListGP
    combined_model = ModelListGP(*models)

    # Fit each model by maximizing the exact marginal log likelihood
    marginal_log_likelihoods = [ExactMarginalLogLikelihood(m.likelihood, m) for m in combined_model.models]
    for mll in marginal_log_likelihoods:
        fit_gpytorch_mll(mll)

    return combined_model


def predict(model: ModelListGP, test_x_target: torch.Tensor) -> list[torch.Tensor]:
    """
    Generate predictions for the target task using the trained multitask GP model.

    Parameters:
        model: Trained ModelListGP containing MultiTaskGP models for each objective
        test_x_target: Input parameters for which to generate predictions.
            Tensor of shape (P, M) where:
            - P = number of test points
            - M = number of input dimensions

    Returns:
        list[torch.Tensor]: Predicted mean values for each objective.
            Each tensor has shape (P, 1) corresponding to the target task predictions.
    """
    # Augment test inputs with target task ID (1.0) as the last column
    test_x_augmented = torch.cat([test_x_target, torch.ones(test_x_target.shape[0], 1)], dim=1)

    # Set model to evaluation mode
    model.eval()

    # Generate predictions without tracking gradients
    with torch.no_grad():
        # Get posterior mean for each objective's model
        return [m.posterior(test_x_augmented).mean for m in model.models]


def prepare_data(x: torch.Tensor, y: torch.Tensor, task_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare data for multitask GP by appending task IDs and ensuring correct tensor shapes.

    Adds a task identifier column to input features and ensures output values have consistent dimensions.

    Parameters:
        x: Input parameters of shape (S, M) where S = number of samples
        y: Output values of shape (S, 1) (single objective)
        task_id: Integer identifier for the task (e.g., 0 for source, 1 for target)

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - x_with_id: Input features with task ID appended, shape (S, M+1)
            - y_reshaped: Output values with consistent dimensions, shape (S, 1, 1)
    """
    # Create tensor of task IDs with same batch size as input data
    task_identifier = torch.full(
        (x.shape[0], 1),  # Shape: (number of samples, 1 column)
        task_id,
        device=x.device,  # Match device of input data
        dtype=x.dtype     # Match data type of input data
    )

    # Append task ID as the last column to input features
    x_with_id = torch.cat([x, task_identifier], dim=1)

    # Ensure output has correct dimensions (adds a singleton dimension if needed)
    y_reshaped = y.unsqueeze(-1) if y.ndim == 1 else y

    return x_with_id, y_reshaped