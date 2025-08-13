import torch
from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective, qLogExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning


def optimize_acq_fun(model, train_y, bounds, batch_size=3, ref_point=None, slack=None, num_restarts=10, raw_samples=128):
    """
    Build a qLogExpectedHypervolumeImprovement acquisition function for multi-objective BO.

    :param model: A fitted ModelListGP models (multi-objective surrogate).
    :param train_y: Current training targets (shape: N x M).
    :param bounds: Limit of the tasks
    :param batch_size: The size of the returned suggestion sample (default: 3).
    :param ref_point: Reference point for hyper-volume calculation (Shape: M, Tensor or list).
    :param slack: Slack to get ref_point automatically (Shape: M)
    :return: candidate (q x d), acquisition values
    """
    if ref_point is None and slack is None:
        raise ValueError("You must provide either a ref_point or a slack value to compute.")
    # If no ref_point
    if ref_point is None:
        ref_point = get_ref_point(train_y, slack)

    # Determine whether it is a tensor
    if not torch.is_tensor(bounds):
        bounds = torch.tensor(bounds, dtype=torch.double)
    if not torch.is_tensor(ref_point):
        ref_point = torch.tensor(ref_point, dtype=torch.double, device=train_y.device)

    acq_func = build_acq_fun(model, ref_point, train_y)

    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,  # Repeat optimization times with different starting points (prevent local optimum)
        raw_samples=raw_samples,  # Initial random sample number (used to find initial value)
        return_best_only=True,  # Only return optimal solution
    )
    return candidate, acq_value  # suggested samples and average acq_value


def get_ref_point(train_y, slack):
    """
    Find a reference point for hyper-volume optimization.

    :param train_y: Current training targets (shape: N x M).
    :param slack: Slack to get ref_point automatically (Shape: M or scaler)
    :return: A reference point (shape: M).
    """
    if isinstance(slack, list):
        slack_tensor = torch.tensor(slack, dtype=train_y.dtype, device=train_y.device)
    else:
        slack_tensor = torch.full_like(train_y[0], fill_value=slack)

    ref_point = train_y.min(dim=0).values - slack_tensor
    ref_point = ref_point.to(train_y.device)
    return ref_point


def build_acq_fun(model, ref_point, train_y):
    """
    Build a qLogExpectedHypervolumeImprovement acquisition function for multi-objective BO.

    :param model: A fitted ModelListGP models (multi-objective surrogate).
    :param ref_point: Reference point for hyper-volume calculation (Tensor or list).
    :param train_y: Current training targets (shape: N x M).
    :return: A qLogEHVI acquisition function object.
    """
    if not torch.is_tensor(ref_point):
        ref_point = torch.tensor(ref_point, dtype=torch.double)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=42)

    objective = IdentityMCMultiOutputObjective()

    y_pareto = train_y

    partitioning = NondominatedPartitioning(
        ref_point=ref_point,
        Y=y_pareto
    )

    acq_func = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
        objective=objective
    )

    return acq_func
