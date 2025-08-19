import torch
from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective, qLogExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning


def optimize_acq_fun(model, train_y: torch.Tensor, bounds, batch_size: int = 3,
                     ref_point=None, slack=None, num_restarts: int = 10, raw_samples: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize a qLogExpectedHypervolumeImprovement acquisition function for multi-objective Bayesian Optimization.

    Constructs and maximizes the log-transformed expected hypervolume improvement acquisition function,
    which balances convergence to the Pareto frontier and diversity across it. Used for selecting the next
    batch of points to evaluate in multi-objective optimization problems.

    Parameters:
        model: Fitted ModelListGP containing independent GPs for each objective (multi-objective surrogate)
        train_y: Current training target values. Tensor of shape (N, M) where:
            - N = number of training samples
            - M = number of objectives
        bounds: Parameter space boundaries. Can be a list/tuple of arrays or a tensor of shape (2, input_dim),
            where bounds[0] = lower bounds and bounds[1] = upper bounds
        batch_size: Number of candidate points to select in each iteration (default: 3)
        ref_point: Reference point for hypervolume calculation. Tensor or list of shape (M,).
            If None, will be computed using `slack`
        slack: Slack value(s) for automatic reference point calculation. Can be a scalar (same for all objectives)
            or array-like of shape (M,). Used only if `ref_point` is None
        num_restarts: Number of restart points for acquisition function optimization (default: 10).
            Helps avoid local optima by restarting optimization from different initial points
        raw_samples: Number of initial random samples for candidate selection (default: 128).
            Used to initialize the optimization with diverse starting points

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - candidate: Optimal candidate points with shape (batch_size, input_dim)
            - acq_value: Acquisition function values at the candidate points with shape (batch_size,)

    Raises:
        ValueError: If neither `ref_point` nor `slack` is provided
    """
    # Validate reference point input
    if ref_point is None and slack is None:
        raise ValueError("You must provide either a ref_point or a slack value to compute it.")

    # Compute reference point automatically if not provided
    if ref_point is None:
        ref_point = get_ref_point(train_y, slack)

    # Ensure bounds and reference point are tensors with correct type and device
    if not torch.is_tensor(bounds):
        bounds = torch.tensor(bounds, dtype=torch.double)
    if not torch.is_tensor(ref_point):
        ref_point = torch.tensor(ref_point, dtype=torch.double, device=train_y.device)

    # Build the acquisition function
    acq_func = build_acq_fun(model, ref_point, train_y)

    # Optimize the acquisition function to find next candidates
    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,  # Parameter space boundaries
        q=batch_size,   # Number of points to select in batch
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        return_best_only=True,  # Return only the best candidate set
    )

    return candidate, acq_value


def get_ref_point(train_y: torch.Tensor, slack) -> torch.Tensor:
    """
    Automatically compute a reference point for hypervolume calculation in multi-objective optimization.

    The reference point is determined as the minimum observed value for each objective minus a slack value,
    ensuring it lies below all observed points in the objective space.

    Parameters:
        train_y: Current training target values. Tensor of shape (N, M)
        slack: Slack value(s) to subtract from the minimum observed values. Can be:
            - A scalar (applied to all objectives)
            - A list or tensor of shape (M,) (per-objective slack values)

    Returns:
        torch.Tensor: Reference point with shape (M,) positioned below all observed points
    """
    # Convert slack to tensor with matching type and device
    if isinstance(slack, list):
        slack_tensor = torch.tensor(slack, dtype=train_y.dtype, device=train_y.device)
    else:
        slack_tensor = torch.full_like(train_y[0], fill_value=slack)

    # Calculate reference point as (minimum observed value per objective) - slack
    ref_point = train_y.min(dim=0).values - slack_tensor
    ref_point = ref_point.to(train_y.device)

    return ref_point


def build_acq_fun(model, ref_point: torch.Tensor, train_y: torch.Tensor) -> qLogExpectedHypervolumeImprovement:
    """
    Construct a qLogExpectedHypervolumeImprovement acquisition function for multi-objective optimization.

    This acquisition function estimates the expected improvement in hypervolume (a measure of Pareto frontier quality)
    from evaluating a batch of candidate points, using log-transformed values for numerical stability.

    Parameters:
        model: Fitted ModelListGP containing independent GPs for each objective
        ref_point: Reference point for hypervolume calculation. Tensor of shape (M,)
        train_y: Current training target values. Tensor of shape (N, M)

    Returns:
        qLogExpectedHypervolumeImprovement: Initialized acquisition function object
    """
    # Ensure reference point is a tensor with correct type
    if not torch.is_tensor(ref_point):
        ref_point = torch.tensor(ref_point, dtype=torch.double)

    # Initialize Sobol quasi-Monte Carlo sampler for expectation estimation
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([128]),  # Number of samples for Monte Carlo estimation
        seed=42  # Fixed seed for reproducibility
    )

    # Use identity objective (no transformation of model outputs)
    objective = IdentityMCMultiOutputObjective()

    # Identify Pareto-optimal points in the current training data
    y_pareto = train_y  # Note: This assumes train_y contains only non-dominated points; filter if necessary

    # Create partitioning of the objective space based on non-dominated points
    partitioning = NondominatedPartitioning(
        ref_point=ref_point,
        Y=y_pareto  # Pareto-optimal points from training data
    )

    # Initialize log-transformed expected hypervolume improvement acquisition function
    acq_func = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,          # Reference point for hypervolume calculation
        partitioning=partitioning,    # Partitioning of the objective space
        sampler=sampler,              # Sampler for expectation estimation
        objective=objective           # Objective transformation (identity here)
    )

    return acq_func