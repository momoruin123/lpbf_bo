"""
black_box.py

This module defines a collection of black-box objective functions
for Bayesian Optimization and LPBF simulations.

Sections:
1. Benchmark Test Problems (e.g., ZDT1-based)
2. LPBF Mechanical Property Models
3. Synthetic Multi-objective Transfer Tasks
"""
import numpy as np
import torch
from pymoo.problems import get_problem
from torch import Tensor


# -------------------------------
# Section 1: Benchmark Test Models
# -------------------------------

def transfer_model_1(x: Tensor, d) -> Tensor:
    """
    In transfer_model_1, "ZDT1" is used, which is a popular testing black box models for Multi-task Bayesian
    optimization. It offers a d-dimension input and 2-dimension output, with d is defined in the function.
    (X with dim=D) -> |Model| -> (Y with dim=2)

    :param x: input tensor
    :param d: dimension of test models input (Normally, d = x.shape[1])
    :return y: output tensor
    """
    problem = get_problem("zdt1", n_var=d)
    # to numpy
    x_np = x.detach().cpu().numpy()
    y_np = problem.evaluate(x_np)
    # to torch
    y = torch.tensor(y_np, device=x.device)
    return y


def transfer_model_2(x: Tensor, d, has_nonlinear=False) -> Tensor:
    """
    In transfer_model_2, the "transfer_model_1" is transformed with different ways, includes
        nonlinear transform: adding linear and nonlinear terms (y_nonlinear = y + w*x + nonlinear_func(x) ),
        linear transform: scaling and translating (y_linear = a*y + b ).

    You can choose adding nonlinear transforms or not by "has_nonlinear" parameter.

    :param x: input tensor
    :param d: dimension of test models input and d = x.shape[1]
    :param has_nonlinear: whether to use nonlinear transform (default: False)
    :return y_trans: output tensor
    """
    if d > 10:
        raise ValueError(f"d = {d} is too large; must be <= 10. High dimensional input will cause memory issues.")

    problem = get_problem("zdt1", n_var=d)
    # to numpy
    x_np = x.detach().cpu().numpy()
    y_np = problem.evaluate(x_np)
    # to torch
    y = torch.tensor(y_np, dtype=x.dtype, device=x.device)
    # final combination
    if has_nonlinear:
        # w = torch.rand(d, device=x.device)device
        w = torch.tensor([
            0.3367, 0.1288, 0.2345, 0.2303, -1.1229,
            -0.1863, 2.2082, -0.6380, -0.1800, 0.0376
        ], device=x.device)
        w = w[0:d]
        # linear item
        linear_term = 0.3 * (x @ w)
        linear_term = linear_term.unsqueeze(1).expand_as(y)
        # nonlinear item
        nonlinear_term = (0.2 * torch.sin(2 * torch.pi * (x[:, 0] + 0.5 * x[:, 1] - 0.3 * x[:, 2]))
                          + 0.1 * torch.cos(3 * torch.pi * x[:, 3]))
        nonlinear_term = nonlinear_term.unsqueeze(1).expand_as(y)
        y = 0.6 * y + linear_term + nonlinear_term + 0.05 * torch.randn_like(y)
    else:
        y = 0.6 * y + 10.8 + 0.05 * torch.randn_like(y)
    return y


# -----------------------------------------
# Section 2: LPBF Mechanical Black-box Models
# -----------------------------------------

def mechanical_model_1(x: torch.Tensor) -> Tensor:
    """
    Simulated black-box function for LPBF laser process.

    Args:
        x (np.ndarray): shape [N, 3], where columns are:
            - power (W) in [25, 300]
            - hatch_distance (mm) in [0.1, 0.6]
            - outline_power (W) in [25, 300]

    Returns:
        y (np.ndarray): shape [N, 6], columns are:
            - label_visibility [0,1]
            - surface_uniformity [0,1]
            - Young's_modulus (MPa)
            - tensile_strength (MPa)
            - Elongation (%)
            - edge_measurement (mm)
    """
    power = x[:, 0]
    hatch = x[:, 1]
    outline = x[:, 2]

    N = x.shape[0]
    noise = lambda scale: scale * torch.randn(N, dtype=x.dtype, device=x.device)

    # --- 前面部分保持不变 ---
    label_visibility = 10 * torch.exp(-((outline - 160) / 50) ** 2) + noise(1.0)
    label_visibility = torch.clamp(label_visibility.round(), 0, 10)

    base = 10 * torch.exp(-((outline - 170) / 50) ** 2)
    modulation = torch.exp(-6 * (hatch - 0.3) ** 2)
    surface_uniformity = base * modulation + noise(1.0)
    surface_uniformity = torch.clamp(surface_uniformity.round(), 0, 10)

    eff_density = (0.6 * power + 0.4 * outline) / (hatch + 1e-4)

    # --- 修改后的 Young's modulus (MPa) ---
    e1 = 800 * torch.exp(-((eff_density - 800) / 150) ** 2)
    e2 = 600 * torch.exp(-((eff_density - 1600) / 200) ** 2)
    # 新增两项：让 E 与前面两个标签相关（系数可调）
    E = (
            1200
            + e1
            + e2
            + 15.0 * label_visibility
            + 10.0 * surface_uniformity
            + noise(30)
    )
    E = torch.clamp(E, 1000, 3000)

    # --- 修改后的 tensile_strength (MPa) ---
    strength = (
            45
            + 10 * torch.exp(-((eff_density - 1000) / 150) ** 2)
            + 5 * torch.exp(-((eff_density - 1800) / 200) ** 2)
            + 5 * torch.sin((eff_density % 500) * 0.01)
            # 新增标签依赖
            + 2.0 * label_visibility
            + 1.5 * surface_uniformity
            + noise(5)
    )
    strength = torch.clamp(strength, 30, 80)

    # --- 修改后的 elongation (%) ---
    elongation = (
            2.0
            + 2.5 * torch.exp(-((eff_density - 900) / 150) ** 2)
            + 1.5 * torch.exp(-((eff_density - 1700) / 180) ** 2)
            # 新增标签依赖
            + 0.2 * label_visibility
            + 0.1 * surface_uniformity
            + noise(0.2)
    )
    elongation = torch.clamp(elongation, 2.0, 6.5)

    # --- edge_error (mm): low is better, no negative ---
    edge_error = 0.7 - 0.0006 * power + 0.12 * hatch + 0.05 * torch.sin(outline * 0.01) + noise(0.05)
    edge_error = torch.clamp(edge_error, 0.05, 1.0)

    # output
    y = torch.stack([
        label_visibility,
        surface_uniformity,
        E,
        strength,
        elongation,
        edge_error
    ], dim=1)

    return y


def mechanical_model(x: torch.Tensor) -> Tensor:
    """
    Simulated black-box function for LPBF laser process.

    Args:
        x (np.ndarray): shape [N, 3], where columns are:
            - power (W) in [25, 300]
            - hatch_distance (mm) in [0.1, 0.6]
            - outline_power (W) in [25, 300]

    Returns:
        y (np.ndarray): shape [N, 6], columns are:
            - label_visibility [0,1]
            - surface_uniformity [0,1]
            - Young's_modulus (MPa)
            - tensile_strength (MPa)
            - Elongation (%)
            - edge_measurement (mm)
    """
    power = x[:, 0]
    hatch = x[:, 1]
    outline = x[:, 2]

    N = x.shape[0]
    noise = lambda scale: scale * torch.randn(N, dtype=x.dtype, device=x.device)

    # --- label_visibility: peak near outline=160 ---
    label_visibility = 10 * torch.exp(-((outline - 160) / 50) ** 2) + noise(1.0)
    label_visibility = torch.clamp(label_visibility.round(), 0, 10)

    # --- surface_uniformity: controlled by outline and hatch ---
    base = 10 * torch.exp(-((outline - 170) / 50) ** 2)
    modulation = torch.exp(-6 * (hatch - 0.3) ** 2)
    surface_uniformity = base * modulation + noise(1.0)
    surface_uniformity = torch.clamp(surface_uniformity.round(), 0, 10)

    # --- effective energy density: no divide-by-zero ---
    eff_density = (0.6 * power + 0.4 * outline) / (hatch + 1e-4)

    # --- Young's modulus (MPa): double peak + safe clamp ---
    e1 = 800 * torch.exp(-((eff_density - 800) / 150) ** 2)
    e2 = 600 * torch.exp(-((eff_density - 1600) / 200) ** 2)
    E = 1200 + e1 + e2 + noise(30)
    E = torch.clamp(E, 1000, 3000)

    # --- tensile_strength (MPa): smooth multi-peak ---
    strength = (
            45
            + 10 * torch.exp(-((eff_density - 1000) / 150) ** 2)
            + 5 * torch.exp(-((eff_density - 1800) / 200) ** 2)
            + 5 * torch.sin((eff_density % 500) * 0.01)
            + noise(5)
    )
    strength = torch.clamp(strength, 30, 80)

    # --- elongation (%): smooth twin peaks ---
    elongation = (
            2.0
            + 2.5 * torch.exp(-((eff_density - 900) / 150) ** 2)
            + 1.5 * torch.exp(-((eff_density - 1700) / 180) ** 2)
            + noise(0.2)
    )
    elongation = torch.clamp(elongation, 2.0, 6.5)

    # --- edge_error (mm): low is better, no negative ---
    edge_error = 0.7 - 0.0006 * power + 0.12 * hatch + 0.05 * torch.sin(outline * 0.01) + noise(0.05)
    edge_error = torch.clamp(edge_error, 0.05, 1.0)

    # output
    y = torch.stack([
        label_visibility,
        surface_uniformity,
        E,
        strength,
        elongation,
        edge_error
    ], dim=1)

    return y


# -------------------------------------
# Section 3: Synthetic Multi-task Models
# -------------------------------------

def func_a(x):
    """
    Build a synthetic black box function.

    :param x: Tasks value
    :return: targets value
    """
    p, v, t, h = x.unbind(dim=1)
    # Volumetric energy density
    ev = p / (v * h * t)

    # Molten pool width & depth (simplified physical empirical formula)
    w = torch.sqrt(p * t / v)
    d = torch.sqrt(p / (v * h))

    # Target 1: Density ~ sigmoid(ev) and minus micropores caused by too wide a melt pool
    y1 = 1.0 / (1 + torch.exp(-(ev - 0.1))) * torch.exp(-0.05 * (w - d).abs())

    # Target 2: Roughness ~ increases with w and h
    y2 = 5.0 + 0.5 * w + 2.0 * h + 0.05 * (p / v)

    # Goal 3: Processing time ~ is inversely proportional to speed t, but positively correlated with thickness
    y3 = (1 / v + 2 * t + 1 * h) * 10

    return torch.stack([y1, -y2, -y3], dim=-1)


def func_b(x):
    p, v, t, h = x.unbind(dim=1)
    ev = p / (v * h * t)
    w = torch.sqrt(p * t / v)
    d = torch.sqrt(p / (v * h))
    y1 = 1.0 / (1 + torch.exp(-(ev - 0.12))) * torch.exp(-0.04 * (w - d).abs())
    y2 = 6.0 + 0.4 * w + 2.2 * h + 0.06 * (p / v)
    y3 = (1 / v + 2 * t + 2 * h) * 10
    return torch.stack([y1, -y2, -y3], dim=-1)


# Source task black box function
def func_1(x):
    """
    Build a synthetic black box function.

    :param x: Tasks value
    :return: targets value
    """
    p, v, t, h = x.unbind(dim=1)
    # Volumetric energy density
    ev = p / (v * h * t)

    # Molten pool width & depth (simplified physical empirical formula)
    w = torch.sqrt(p * t / v)
    d = torch.sqrt(p / (v * h))

    # Target 1: Density ~ sigmoid(ev) and minus micropores caused by too wide a melt pool
    y1 = 1.0 / (1 + torch.exp(-(ev - 0.08))) * torch.exp(-0.06 * (w - d).abs())

    # Target 2: Roughness ~ increases with w and h
    y2 = 5.0 + 0.6 * w + 1.8 * h + 0.04 * (p / v)

    # Goal 3: Processing time ~ is inversely proportional to speed t, but positively correlated with thickness
    y3 = (1 / v + 1.8 * t + 0.4 / h) * 5

    return torch.stack([y1, -y2, -y3], dim=-1)


# Target task black box function
def func_2(x):
    p, v, t, h = x.unbind(dim=1)
    ev = p / (v * h * t)
    w = torch.sqrt(p * t / v)
    d = torch.sqrt(p / (v * h))
    y1 = 1.0 / (1 + torch.exp(-(ev - 0.12))) * torch.exp(-0.04 * (w - d).abs())
    y2 = 6.0 + 0.4 * w + 2.2 * h + 0.06 * (p / v)
    y3 = (1 / v + 2.0 * t + 0.5 / h) * 5
    return torch.stack([y1, -y2, -y3], dim=-1)
