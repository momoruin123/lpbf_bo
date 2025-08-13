import torch
from botorch.utils.multi_objective import Hypervolume
from torch import Tensor


def get_hyper_volume(pareto_y: Tensor, ref_point) -> float:
    """
    Compute hyper-volume given training outputs and a reference point. The bigger, the better.

    :param pareto_y: [N, M]
    :param ref_point: [M]
    :return hv_value: scalar hyper-volume value
    """
    if not torch.is_tensor(ref_point):
        ref_point = torch.tensor(ref_point, dtype=torch.double, device=pareto_y.device)

    hv = Hypervolume(ref_point=ref_point)
    hv_value = hv.compute(pareto_y)  # to float

    return hv_value


def get_gd(pareto_y: Tensor, true_pf: Tensor) -> float:
    """
    Generational Distance:
    Average distance from each algorithm frontier point to the nearest true frontier point (the smaller, the better).
    Emphasizes the convergence of the algorithm solution set

    :param pareto_y: pareto points set [N, M]
    :param true_pf: true pareto frontier [K, M]
    :return scalar: gd value
    """
    # Calculate the minimum distance from each point to true_pf
    dists = torch.cdist(pareto_y, true_pf)  # [N, K]
    min_dists = torch.min(dists, dim=1).values  # [N]
    return float(min_dists.mean().item())


def get_igd(pareto_y: Tensor, true_pf: Tensor) -> float:
    """
    Inverted GD:
    Average distance from each true frontier point to the nearest algorithm frontier point (the smaller, the better).
    Emphasizes the coverage of the algorithm solution set

    :param pareto_y: pareto points set [N, M]
    :param true_pf: true pareto frontier [K, M]
    :return scalar: igd value
    """
    dists = torch.cdist(true_pf, pareto_y)  # [K, N]
    min_dists = torch.min(dists, dim=1).values  # [K]
    return float(min_dists.mean().item())


def get_spacing(pareto_y: Tensor) -> float:
    """
    Spacing: The standard deviation of the Euclidean distance between adjacent points on the approximate frontier (
    the smaller the value, the more uniform the distribution). First calculate the distances of all point pairs,
    then take the distance set of the nearest neighbors of each point, and finally find its standard deviation.

    :param pareto_y: pareto points set [N, M]
    :return scalar: spacing value
    """
    # Distance matrix
    D = torch.cdist(pareto_y, pareto_y)  # [N, N]
    # Set the diagonal (self-distance) to +inf, so that the nearest neighbor is another point
    D.fill_diagonal_(float("inf"))
    # The minimum value of each row is the distance from the point to the nearest neighbor.
    nn_dists = torch.min(D, dim=0).values  # [n]
    return float(nn_dists.std(unbiased=True).item())


def get_cardinality(pareto_y: Tensor) -> int:
    """Number of Pareto non-dominated frontier points"""
    return pareto_y.size(0)
