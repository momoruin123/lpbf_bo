import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler


def build_acq_fun(model, train_y: torch.Tensor) -> qLogExpectedImprovement:
    """
    Build a qLogExpectedImprovement acquisition function for Bayesian Optimization.

    Constructs a log-transformed expected improvement acquisition function, which is robust
    to high noise levels and large objective values. Uses Sobol quasi-Monte Carlo sampling
    for estimating the acquisition function values.

    Parameters:
        model: Trained Gaussian Process model (e.g., SingleTaskGP) used for predictions
        train_y: Training data target values. Tensor of shape (n_samples,) or (n_samples, 1)
            Used to determine the current best objective value

    Returns:
        qLogExpectedImprovement: Initialized acquisition function object
    """
    # Ensure train_y is properly shaped for best value calculation (remove singleton dimension if present)
    if train_y.ndim == 2 and train_y.shape[1] == 1:
        train_y = train_y.squeeze(-1)  # Convert from (n, 1) to (n,)

    # Determine the best observed objective value from training data
    best_f = train_y.max().item()

    # Initialize Sobol quasi-Monte Carlo sampler for stable acquisition function estimation
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([128]),  # Number of samples for Monte Carlo estimation
        seed=42  # Fixed seed for reproducibility
    )

    # Initialize log expected improvement acquisition function
    acq_fun = qLogExpectedImprovement(
        model=model,
        best_f=best_f,  # Current best objective value
        sampler=sampler  # Sampler for estimating the expectation
    )

    return acq_fun


def optimize_acq_fun(model, train_y: torch.Tensor, bounds, batch_size: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize the acquisition function to find the next set of candidate points.

    Maximizes the pre-constructed acquisition function within the given parameter bounds
    to select the next batch of points for evaluation.

    Parameters:
        model: Trained Gaussian Process model used by the acquisition function
        train_y: Training data target values. Tensor of shape (n_samples,) or (n_samples, 1)
        bounds: Parameter space boundaries. Can be a list/tuple of arrays or a tensor of shape
            (2, input_dim), where bounds[0] = lower bounds and bounds[1] = upper bounds
        batch_size: Number of candidate points to select (default: 3)

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - candidate: Optimal candidate points with shape (batch_size, input_dim)
            - acq_value: Acquisition function values at the candidate points with shape (batch_size,)
    """
    # Convert bounds to tensor if necessary and ensure correct dtype/device
    if not torch.is_tensor(bounds):
        bounds = torch.tensor(bounds, dtype=torch.double)

    # Move bounds to the same device as the model parameters
    bounds = bounds.to(
        dtype=torch.double,
        device=next(model.parameters()).device  # Infer device from model parameters
    )

    # Build the acquisition function
    acq_func = build_acq_fun(model, train_y)

    # Optimize the acquisition function to find the best candidates
    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,  # Parameter space boundaries
        q=batch_size,  # Number of points to select in batch
        num_restarts=30,  # Number of restart points for optimization
        raw_samples=2048,  # Number of initial samples for candidate selection
        return_best_only=True,  # Return only the best candidate set
    )

    return candidate, acq_value