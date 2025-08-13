import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


def build_model(x_old, y_old, x_new, y_new):
    """
    Build a multitask Gaussian Process (GP) surrogate model.

    :param x_old: X from source task
                    Input tensor of shape (K, M), where
                    K is the number of samples and
                    M is the number of process parameters (e.g., power, speed, etc.).
    :param y_old: Y from source task
                    Output tensor of shape (K, N), where
                    N is the number of target metrics (e.g., density, roughness, processing time).
    :param x_new: X from target task
    :param y_new: Y from target task
    :return: A fitted ModelListGP object containing independent GPs for each objective.
    """
    input_dim = x_old.shape[1]
    target_dim = y_new.shape[1]

    # Create a MultiTaskGP for each  task
    models = []
    for i in range(target_dim):  # for [density, roughness, time]
        x_old_p, y_old_pp = prepare_data(x_old, y_old[:, i:i + 1], 0)
        x_new_p, y_new_p = prepare_data(x_new, y_new[:, i:i + 1], 1)
        x_all = torch.cat([x_old_p, x_new_p], dim=0)
        y_all = torch.cat([y_old_pp, y_new_p], dim=0)

        model = MultiTaskGP(
            train_X=x_all,
            train_Y=y_all,
            task_feature=-1,
            output_tasks=[1],  # output task_1
            input_transform=Normalize(d=input_dim+1, indices=list(range(input_dim))),
            outcome_transform=Standardize(m=1)
        )
        models.append(model)

    model = ModelListGP(*models)

    # 拟合每个模型
    mlls = [ExactMarginalLogLikelihood(m.likelihood, m) for m in model.models]
    for mll in mlls:
        fit_gpytorch_mll(mll)

    return model


def predict(model, test_x_target):
    """
    Predict on target task (task_id = 1.0)
    """
    test_x_aug = torch.cat([test_x_target, torch.ones(test_x_target.shape[0], 1)], dim=1)
    model.eval()
    with torch.no_grad():  # DO NOT compute gradients
        return [m.posterior(test_x_aug).mean for m in model.models]


def prepare_data(x, y, task_id):
    # Append task IDs
    task_tensor = torch.full((x.shape[0], 1), task_id, device=x.device, dtype=x.dtype)
    x_and_id = torch.cat([x, task_tensor], dim=1)
    y = y.unsqueeze(-1) if y.ndim == 1 else y

    return x_and_id, y
