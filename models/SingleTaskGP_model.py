import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


def build_model(train_x: torch.Tensor, train_y: torch.Tensor) -> ModelListGP:
    """
    Build a multi-objective Gaussian Process (GP) surrogate model for multiple target metrics.

    Constructs independent SingleTaskGP models for each objective (target metric) and combines them into a ModelListGP.
    Each model is fitted by maximizing the exact marginal log likelihood.

    Parameters:
        train_x: Input training data.
            Tensor of shape (K, M) where:
            - K = number of training samples
            - M = number of input dimensions (process parameters)
        train_y: Output training data (target metrics).
            Tensor of shape (K, N) where:
            - N = number of target metrics (objectives)

    Returns:
        ModelListGP: A collection of fitted SingleTaskGP models (one per objective)
    """
    input_dim = train_x.shape[1]  # Number of input dimensions (M)
    num_targets = train_y.shape[1]  # Number of target metrics (N)
    models = []

    # Create a separate GP model for each target metric
    for i in range(num_targets):
        # Extract data for the i-th target metric (shape: [K, 1])
        target_y = train_y[:, i:i+1]

        # Initialize SingleTaskGP for the current target
        model_i = SingleTaskGP(
            train_X=train_x,
            train_Y=target_y,
            input_transform=Normalize(d=input_dim),  # Normalize input features to [0,1] range
            outcome_transform=Standardize(m=1)       # Standardize output to zero mean and unit variance
        )
        models.append(model_i)

    # Combine all per-objective models into a single ModelListGP
    combined_model = ModelListGP(*models)

    # Fit each model by maximizing the exact marginal log likelihood
    marginal_log_likelihoods = [ExactMarginalLogLikelihood(m.likelihood, m) for m in combined_model.models]
    for mll in marginal_log_likelihoods:
        fit_gpytorch_mll(mll)

    return combined_model


def build_single_model(train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
    """
    Build a single-objective Gaussian Process (GP) surrogate model for a single target metric.

    Constructs and fits a SingleTaskGP model using the input and output training data, with input normalization
    and output standardization for stable training.

    Parameters:
        train_x: Input training data.
            Tensor of shape (K, M) where:
            - K = number of training samples
            - M = number of input dimensions (process parameters)
        train_y: Output training data (single target metric).
            Tensor of shape (K,) or (K, 1) (will be reshaped to (K, 1) if necessary)

    Returns:
        SingleTaskGP: A fitted single-task Gaussian Process model
    """
    # Ensure output data has shape (K, 1) for consistency
    if train_y.ndim == 1:
        train_y = train_y.unsqueeze(-1)

    input_dim = train_x.shape[1]  # Number of input dimensions (M)

    # Initialize SingleTaskGP with input normalization and output standardization
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=Normalize(d=input_dim),  # Normalize input features to [0,1] range
        outcome_transform=Standardize(m=1)       # Standardize output to zero mean and unit variance
    )

    # Fit the model by maximizing the exact marginal log likelihood
    marginal_log_likelihood = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(marginal_log_likelihood)

    return model


def predict(model: SingleTaskGP | ModelListGP, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate predictive mean and variance from a fitted Gaussian Process model.

    Computes the posterior mean and marginal variance for the given input points using the trained GP model.
    Works with both single-task (SingleTaskGP) and multi-task (ModelListGP) models.

    Parameters:
        model: Fitted Gaussian Process model (either SingleTaskGP or ModelListGP)
        x: Input points for prediction.
            Tensor of shape (P, M) where:
            - P = number of prediction points
            - M = number of input dimensions

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - mean: Predictive mean values. Shape:
              - (P, 1) for SingleTaskGP
              - (P, N) for ModelListGP (N = number of objectives)
            - variance: Predictive marginal variance values. Shape matches `mean`
    """
    # Set model to evaluation mode
    model.eval()

    # Generate predictions without tracking gradients
    with torch.no_grad():
        posterior = model.posterior(x)
        mean = posterior.mean          # Predictive mean
        variance = posterior.variance  # Marginal predictive variance (diagonal of covariance matrix)

        return mean, variance