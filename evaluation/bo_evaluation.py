import torch
from botorch.utils.multi_objective import Hypervolume
from torch import Tensor


def get_hyper_volume(pareto_y: Tensor, ref_point) -> float:
    """Compute the Hypervolume metric for evaluating multi-objective optimization performance.

    Hypervolume measures the volume of the objective space dominated by the Pareto frontier
    relative to a reference point. A larger hypervolume indicates better performance as it
    captures both convergence and diversity of the solution set.

    Parameters:
        pareto_y: Tensor of shape [N, M] representing the algorithm's Pareto frontier points.
            N is the number of Pareto points, M is the number of objectives.
        ref_point: Reference point of shape [M], typically set to the worst values for all objectives.
            Serves as the lower bound for hypervolume calculation.

    Returns:
        float: Hypervolume value, where larger values indicate better performance.
    """
    # Convert reference point to tensor if needed, matching device and dtype of pareto_y
    if not torch.is_tensor(ref_point):
        ref_point = torch.tensor(ref_point, dtype=torch.double, device=pareto_y.device)
    # Initialize hypervolume calculator with reference point
    hv = Hypervolume(ref_point=ref_point)
    # Compute hypervolume and return as float
    hv_value = hv.compute(pareto_y)

    return hv_value


def get_gd(pareto_y: Tensor, true_pf: Tensor) -> float:
    """Compute Generational Distance (GD) to evaluate convergence to the true Pareto frontier.

    GD measures the average Euclidean distance from each point in the algorithm's Pareto frontier
    to the nearest point in the true Pareto frontier. A smaller GD indicates better convergence.

    Parameters:
        pareto_y: Tensor of shape [N, M] representing the algorithm's Pareto frontier points.
            N is the number of algorithm-generated Pareto points, M is the number of objectives.
        true_pf: Tensor of shape [K, M] representing the true Pareto frontier.
            K is the number of true Pareto points, M is the number of objectives.

    Returns:
        float: GD value, where smaller values indicate better convergence.
    """
    # Calculate pairwise Euclidean distances between algorithm points and true Pareto points
    dists = torch.cdist(pareto_y, true_pf)  # Shape: [N, K]

    # Find minimum distance from each algorithm point to any true Pareto point
    min_dists = torch.min(dists, dim=1).values  # Shape: [N]

    # Return average of these minimum distances
    return float(min_dists.mean().item())


def get_igd(pareto_y: Tensor, true_pf: Tensor) -> float:
    """Compute Inverted Generational Distance (IGD) to evaluate coverage of the true frontier.

    IGD measures the average Euclidean distance from each point in the true Pareto frontier
    to the nearest point in the algorithm's Pareto frontier. A smaller IGD indicates better
    coverage of the true frontier by the algorithm's solutions.

    Parameters:
        pareto_y: Tensor of shape [N, M] representing the algorithm's Pareto frontier points.
            N is the number of algorithm-generated Pareto points, M is the number of objectives.
        true_pf: Tensor of shape [K, M] representing the true Pareto frontier.
            K is the number of true Pareto points, M is the number of objectives.

    Returns:
        float: IGD value, where smaller values indicate better coverage.
    """
    # Calculate pairwise Euclidean distances between true Pareto points and algorithm points
    dists = torch.cdist(true_pf, pareto_y)  # Shape: [K, N]

    # Find minimum distance from each true Pareto point to any algorithm point
    min_dists = torch.min(dists, dim=1).values  # Shape: [K]

    # Return average of these minimum distances
    return float(min_dists.mean().item())


def get_spacing(pareto_y: Tensor) -> float:
    """Compute Spacing metric to evaluate distribution uniformity of Pareto points.

    Spacing measures the standard deviation of distances between each point and its nearest
    neighbor in the algorithm's Pareto frontier. A smaller spacing value indicates a more
    uniform distribution of points across the frontier.

    Parameters:
        pareto_y: Tensor of shape [N, M] representing the algorithm's Pareto frontier points.
            N is the number of Pareto points, M is the number of objectives.

    Returns:
        float: Spacing value, where smaller values indicate more uniform distribution.
    """
    # Calculate pairwise Euclidean distances between all Pareto points
    D = torch.cdist(pareto_y, pareto_y)  # Shape: [N, N]

    # Set self-distances (diagonal) to infinity to exclude from nearest neighbor calculation
    D.fill_diagonal_(float("inf"))

    # Find minimum distance (nearest neighbor) for each point
    nn_dists = torch.min(D, dim=0).values  # Shape: [N]

    # Return standard deviation of nearest neighbor distances
    return float(nn_dists.std(unbiased=True).item())


def get_cardinality(pareto_y: Tensor) -> int:
    """Get the cardinality (size) of the Pareto frontier.

    Cardinality represents the number of non-dominated points in the algorithm's Pareto frontier,
    providing a measure of the solution set's richness.

    Parameters:
        pareto_y: Tensor of shape [N, M] representing the algorithm's Pareto frontier points.
            N is the number of Pareto points, M is the number of objectives.

    Returns:
        int: Number of points in the Pareto frontier.
    """
    return pareto_y.size(0)