"""
A BO class with embedding-based representations

:author: Maoyurun Mao
:affiliation: Institut für Strahlwerkzeuge (IFSW), University of Stuttgart
:date: 2025-08-18
"""
import torch

from optimization.base_bo_class import BaseBO
from optimization.utils import run_multitask_bo, run_singletask_bo
from models import SingleTaskGP_model, MultiTaskGP_model


def attach_feature_vector(x: torch, v: list):
    """
    Attach feature vectors to x

    :param x: input set, shape (n_samples, n_features)
    :param v: feature vectors, shape (1, n_embedding_features)
    :return: augmented set, shape (n_samples, n_features + n_embedding_features)
    """
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.double, device=x.device)

    v = v.repeat(x.shape[0], 1)
    x_aug = torch.cat((x, v), dim=1)
    return x_aug


class EmbeddingBO(BaseBO):
    def __init__(self, input_dim, objective_dim, vector_dim, seed=None, device=None):
        """
        :param input_dim: the dimension of input data, e.g. processing parameters(h, v,...)
        :param objective_dim: the dimension of objective function, e.g. final note of product
        :param vector_dim: the dimension of featrue vectores, e.g. number of features
        :param seed: seed for random number generator
        :param device: device for computation
        """

        super().__init__(input_dim, objective_dim, seed, device)
        # vector_dim
        self.vector_dim = vector_dim
        # source task X/Y init
        self.X = torch.empty((0, input_dim+vector_dim), device=self.device)
        self.Y = torch.empty((0, objective_dim), device=self.device)

        # Determine objective dimensions
        if objective_dim == 1:
            self._is_single_task = True
        else:
            self._is_single_task = False

    def add_augment_data(self, X_new, Y_new, v):
        """Append new augment observations"""
        X_new = torch.as_tensor(X_new, dtype=self.X.dtype, device=self.device)
        Y_new = torch.as_tensor(Y_new, dtype=self.Y.dtype, device=self.device)
        assert X_new.shape[0] == Y_new.shape[0], "X and Y batch size(Number of lines) mismatch"
        X_aug = attach_feature_vector(X_new, v)
        self.X = torch.cat([self.X, X_aug], dim=0)
        self.Y = torch.cat([self.Y, Y_new], dim=0)

    def set_bounds(self, lower, upper):
        """lower/upper: array-like length: input_dim + vector_dim"""
        lower = torch.as_tensor(lower, dtype=self.X.dtype, device=self.device).view(-1)
        upper = torch.as_tensor(upper, dtype=self.X.dtype, device=self.device).view(-1)
        assert (lower.numel() == self.input_dim+self.vector_dim and
                upper.numel() == self.input_dim+self.vector_dim), "bounds dim mismatch (input_dim+vector_dim)"
        assert torch.all(upper >= lower), "upper must be > lower"
        self.bounds = torch.stack([lower, upper], dim=0)

    def build_model(self):
        """build and return GP model with task augmented data. Make sure to call when X and Y is not empty"""
        assert self.X.numel() > 0 and self.Y.numel() > 0, \
            "X/Y are empty; add_augment_data() before run_bo()."
        model = SingleTaskGP_model.build_model(self.X, self.Y)  # build GP model
        self.model = model
        return model

    def run_bo(self):
        self.build_model()
        # set y for BO based on whether there is sample
        if self.X.nelement() == 0:  # if not, use source task samples to do BO
            y_bo = self.Y
        else:  # if yes, use target task samples to do BO
            y_bo = self.Y

        # determine whether it is single-task optimization
        if self._is_single_task:  # if yes, use qlogEI as asq. func.
            X_next = run_singletask_bo(
                model=self.model,
                bounds=self.bounds,
                train_y=y_bo,
                batch_size=self.batch_size,
                mini_batch_size=self.mini_batch_size,
                device=self.device
            )
        else:  # if no, use qlogEHVI as asq. func.
            X_next = run_multitask_bo(
                model=self.model,
                bounds=self.bounds,
                train_y=y_bo,
                ref_point=self.ref_point,
                batch_size=self.batch_size,
                mini_batch_size=self.mini_batch_size,
                device=self.device
            )
        print(X_next[:, 0:self.input_dim])
        return X_next
