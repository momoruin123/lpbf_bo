import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler


def build_acq_fun(model, train_y):
    if train_y.ndim == 2 and train_y.shape[1] == 1:
        train_y = train_y.squeeze(-1)

    best_f = train_y.max().item()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=42)

    acq_fun = qLogExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=sampler
    )
    return acq_fun


def optimize_acq_fun(model, train_y, bounds, batch_size=3):
    if not torch.is_tensor(bounds):
        bounds = torch.tensor(bounds, dtype=torch.double)

    bounds = bounds.to(dtype=torch.double, device=next(model.parameters()).device)

    acq_func = build_acq_fun(model, train_y)

    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=30,
        raw_samples=2048,
        return_best_only=True,
    )
    return candidate, acq_value
