"""
A general Bayesian Optimization (BO) class.

Two typical usage sequences:

- When you have initial samples:
    1) Instantiate `BaseBoClass`.
    2) Call `read_train_data`.
    3) Call `set_bounds`.
    4) Call `set_ref_point`.
    5) Call `build_model`.
    6) Call `run_bo`.

- When you don't have initial samples:
    1) Instantiate `BaseBoClass`.
    2) Call `set_bounds`.
    3) Call `run_start_sampling` to get random samples.

:note: Ensure that all objective functions are optimized in the direction of maximization

:author: Maoyurun Mao
:affiliation: Institut für Strahlwerkzeuge (IFSW), University of Stuttgart
:date: 2025-08-12
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import torch
import numpy as np

from optimization.utils import generate_initial_data, run_multitask_bo
from models import SingleTaskGP_model


class BaseBO:
    def __init__(self, input_dim, objective_dim, seed=None, device=None):
        # for device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        torch.set_default_dtype(torch.float64)  # set default data type
        self.seed = seed  # seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # dims
        self.input_dim = input_dim
        self.objective_dim = objective_dim

        # BO config
        # bounds: 2 x d tensor: [[l1,...,ld],[u1,...,ud]]
        self.bounds = torch.empty(2, input_dim, device=self.device)
        self.batch_size = 2
        self.mini_batch_size = self.batch_size
        self.ref_point = None  # reference point

        # data (empty)
        self.X = torch.empty((0, input_dim), device=self.device)  # input init
        self.Y = torch.empty((0, objective_dim), device=self.device)  # objective init

        # model holder
        self.model = None

        # # for evaluation
        # self.test_iter = 1  # Number of tests
        # self.n_iter = 2  # Number of iterations
        # self.n_init_samples = 0  # Number of initial samples of new task
        #
        # # log
        # self.X_log = torch.empty((0, self.input_dim), device=self.device)
        # self.Y_log = torch.empty((0, self.objective_dim), device=self.device)

    # ---------- data ops ----------
    def add_data(self, X_new, Y_new):
        """Append new observations"""
        X_new = torch.as_tensor(X_new, dtype=self.X.dtype, device=self.device).view(-1, self.input_dim)
        Y_new = torch.as_tensor(Y_new, dtype=self.Y.dtype, device=self.device).view(-1, self.objective_dim)
        assert X_new.shape[0] == Y_new.shape[0], "X/Y batch size mismatch"
        self.X = torch.cat([self.X, X_new], dim=0)
        self.Y = torch.cat([self.Y, Y_new], dim=0)

    def clear_data(self):
        """reset X and Y to empty tensor"""
        self.X = torch.empty(0, self.input_dim, dtype=self.X.dtype, device=self.device)
        self.Y = torch.empty(0, self.objective_dim, dtype=self.Y.dtype, device=self.device)

    # ---------- set ----------
    def set_batch_size(self, batch_size, mini_batch_size):
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

    def set_bounds(self, lower, upper):
        """lower/upper: array-like length d"""
        lower = torch.as_tensor(lower, dtype=self.X.dtype, device=self.device).view(-1)
        upper = torch.as_tensor(upper, dtype=self.X.dtype, device=self.device).view(-1)
        assert lower.numel() == self.input_dim and upper.numel() == self.input_dim, "bounds dim mismatch"
        assert torch.all(upper > lower), "upper must be > lower"
        self.bounds = torch.stack([lower, upper], dim=0)

    # def set_ref_point(self, ref_point):
    #     """Reference point for multi-objective optimization. Length = objective_dim"""
    #     rp = torch.as_tensor(ref_point, device=self.device).view(-1)
    #     assert rp.numel() == self.objective_dim, "ref_point dim mismatch"
    #     self.ref_point = rp

    def set_ref_point(self, slack):
        """
        Find a reference point for hyper-volume optimization.

        :param slack: Slack to get ref_point automatically (Shape: M or scaler)
        :return: A reference point (shape: M).
        """
        if isinstance(slack, list):
            slack_tensor = torch.tensor(slack, dtype=self.Y.dtype, device=self.device)
        else:
            slack_tensor = torch.full_like(self.Y[0], fill_value=slack)

        ref_point = self.Y.min(dim=0).values - slack_tensor
        ref_point = ref_point.to(self.device)
        self.ref_point = ref_point
        return ref_point

    # ---------- get ----------
    def get_train_data(self):
        return self.X, self.Y

    def get_ref_point(self):
        return self.ref_point

    def get_bounds(self):
        return self.bounds

    # ---------- model ----------
    def build_model(self, **kwargs):
        """build and return GP model. Make sure to call when XY is not empty"""
        assert self.X.numel() > 0 and self.Y.numel() > 0, "X/Y are empty; add_data() before build_model()."
        model = SingleTaskGP_model.build_model(self.X, self.Y)
        self.model = model.to(self.device)
        return self.model

    # ---------- BO process ----------
    def run_start_sampling(self):
        X_next, _ = generate_initial_data(0, self.bounds, self.batch_size, self.input_dim, self.device)
        print(X_next)
        return X_next

    def run_bo(self):
        self.build_model()
        X_next = run_multitask_bo(  # run BO
            model=self.model,
            bounds=self.bounds,
            train_y=self.Y,
            ref_point=self.ref_point,
            batch_size=self.batch_size,
            mini_batch_size=self.mini_batch_size,
            device=self.device
        )
        print(X_next)
        return X_next
