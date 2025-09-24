"""
A general Bayesian Optimization (BO) class.

Two typical usage sequences:

- When you have initial samples:
    1) Instantiate `BaseBO`.
    2) Call `read_train_data` to load existing data.
    3) Call `set_bounds` to define parameter space boundaries.
    4) Call `set_ref_point` to set reference point for multi-objective optimization.
    5) Call `build_model` to construct the Gaussian Process model.
    6) Call `run_bo` to perform the optimization loop.

- When you don't have initial samples:
    1) Instantiate `BaseBO`.
    2) Call `set_bounds` to define parameter space boundaries.
    3) Call `run_start_sampling` to generate initial random samples.

Note: Ensure that all objective functions are optimized in the direction of maximization.

Author: Maoyurun Mao
Affiliation: Institut für Strahlwerkzeuge (IFSW), University of Stuttgart
Date: 2025-08-12
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import torch
import numpy as np

from optimization.utils import generate_initial_data, run_multitask_bo, run_singletask_bo
from models import SingleTaskGP_model


class BaseBO:
    """A basic class for Bayesian Optimization.

    Handles data input/output operations, parameter bounds management, reference point setting,
    Gaussian Process model construction, and execution of the Bayesian Optimization loop.
    """
    def __init__(self, input_dim: int, objective_dim: int, seed: int = None, device: torch.device = None):
        """Initialize the BaseBO class with dimensionality parameters and device configuration.

        Args:
            input_dim: Number of input dimensions (parameters to optimize)
            objective_dim: Number of objective dimensions (target functions)
            seed: Random seed for reproducibility. If None, no fixed seed is set.
            device: Torch device to use for computations. If None, automatically selects
                   CUDA if available, otherwise uses CPU.
        """
        # Device configuration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        torch.set_default_dtype(torch.float64)  # Set default dtype to double precision

        # Random seed configuration for reproducibility
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Dimensionality parameters
        self.input_dim = input_dim
        self.objective_dim = objective_dim

        # Bayesian Optimization configuration
        self.bounds = torch.empty(2, input_dim, device=self.device)  # [2, d] tensor: [[lower bounds], [upper bounds]]
        self.batch_size = 2  # Number of points to query in each BO iteration
        self.mini_batch_size = self.batch_size  # Mini-batch size for optimization
        self.ref_point = None  # Reference point for multi-objective optimization

        # Training data storage
        self.X = torch.empty((0, input_dim), device=self.device)  # Input parameters [N, input_dim]
        self.Y = torch.empty((0, objective_dim), device=self.device)  # Objective values [N, objective_dim]

        # Gaussian Process model holder
        self.model = None

        # Determine optimization type (single vs multi-objective)
        self._is_single_task = (objective_dim == 1)

    def add_data(self, X_new, Y_new) -> None:
        """Append new observations to the existing training data.

        Converts input data to appropriate tensor format and concatenates with existing data.

        Args:
            X_new: New input parameters to add. Can be array-like or tensor.
                   Should have shape [n_samples, input_dim]
            Y_new: Corresponding objective values. Can be array-like or tensor.
                   Should have shape [n_samples, objective_dim]

        Raises:
            AssertionError: If the number of samples in X_new and Y_new do not match
        """
        # Convert to tensors and ensure correct shape and device
        X_new = torch.as_tensor(X_new, dtype=self.X.dtype, device=self.device).view(-1, self.input_dim)
        Y_new = torch.as_tensor(Y_new, dtype=self.Y.dtype, device=self.device).view(-1, self.objective_dim)

        # Validate input consistency
        assert X_new.shape[0] == Y_new.shape[0], "X/Y batch size mismatch - must have same number of samples"

        # Append new data
        self.X = torch.cat([self.X, X_new], dim=0)
        self.Y = torch.cat([self.Y, Y_new], dim=0)

    def clear_data(self) -> None:
        """Reset training data (X and Y) to empty tensors while preserving dtype and device."""
        self.X = torch.empty(0, self.input_dim, dtype=self.X.dtype, device=self.device)
        self.Y = torch.empty(0, self.objective_dim, dtype=self.Y.dtype, device=self.device)

    def set_batch_size(self, batch_size: int, mini_batch_size: int) -> None:
        """Set the batch sizes for BO iterations.

        Args:
            batch_size: Number of points to query in each BO iteration
            mini_batch_size: Mini-batch size for internal optimization steps
        """
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

    def set_bounds(self, lower, upper) -> None:
        """Set the parameter space boundaries.

        Args:
            lower: Lower bounds for each input dimension. Array-like of length input_dim
            upper: Upper bounds for each input dimension. Array-like of length input_dim

        Raises:
            AssertionError: If bounds dimensions don't match input_dim or if upper <= lower for any dimension
        """
        # Convert to tensors and ensure correct shape and device
        lower = torch.as_tensor(lower, dtype=self.X.dtype, device=self.device).view(-1)
        upper = torch.as_tensor(upper, dtype=self.X.dtype, device=self.device).view(-1)

        # Validate bounds
        assert lower.numel() == self.input_dim and upper.numel() == self.input_dim, \
            f"Bounds dimension mismatch - expected {self.input_dim} dimensions"
        assert torch.all(upper > lower), "All upper bounds must be greater than lower bounds"

        self.bounds = torch.stack([lower, upper], dim=0)

    def set_ref_point(self, slack) -> torch.Tensor:
        """ref_point"""
        # Ensure we have training data to calculate minimum values
        assert self.Y.numel() > 0, "No training data available - add data before setting reference point"

        # Process slack value
        if isinstance(slack, list):
            slack_tensor = torch.tensor(slack, dtype=self.Y.dtype, device=self.device)
        else:
            slack_tensor = torch.full_like(self.Y[0], fill_value=slack)

        # Calculate reference point as (min Y values) - slack
        ref_point = self.Y.min(dim=0).values - slack_tensor
        ref_point = ref_point.to(self.device)

        self.ref_point = ref_point
        return ref_point

    def get_train_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the current training data.

        Returns:
            tuple: (X, Y) where:
                X is a tensor of input parameters with shape [N, input_dim]
                Y is a tensor of objective values with shape [N, objective_dim]
        """
        return self.X, self.Y

    def get_ref_point(self) -> torch.Tensor:
        """Get the current reference point.

        Returns:
            torch.Tensor: Reference point with shape [objective_dim], or None if not set
        """
        return self.ref_point

    def get_bounds(self) -> torch.Tensor:
        """Get the parameter space boundaries.

        Returns:
            torch.Tensor: Bounds tensor with shape [2, input_dim], where
                          bounds[0] contains lower bounds and bounds[1] contains upper bounds
        """
        return self.bounds

    def build_model(self, **kwargs) -> SingleTaskGP_model:
        """Build and return a Gaussian Process model using current training data.

        The model is constructed using the SingleTaskGP_model and moved to the configured device.

        Returns:
            SingleTaskGP_model: The constructed Gaussian Process model

        Raises:
            AssertionError: If no training data is available (X or Y is empty)
        """
        # Ensure we have training data
        assert self.X.numel() > 0 and self.Y.numel() > 0, \
            "No training data available - call add_data() before building model"

        # Build model and move to device
        model = SingleTaskGP_model.build_model(self.X, self.Y)
        self.model = model.to(self.device)
        return self.model

    def run_start_sampling(self) -> torch.Tensor:
        """Generate initial random samples within the parameter bounds.

        Used when no initial training data is available to start the BO process.

        Returns:
            torch.Tensor: Initial sample points with shape [batch_size, input_dim]
        """
        X_next, _ = generate_initial_data(
            0,
            self.bounds,
            self.batch_size,
            self.input_dim,
            self.device
        )
        print(X_next)
        return X_next

    def run_bo(self) -> torch.Tensor:
        """Run a single iteration of Bayesian Optimization.

        Builds the model (if not already built) and queries new points using the BO strategy.

        Returns:
            torch.Tensor: Next set of points to evaluate with shape [batch_size, input_dim]
        """
        # Ensure model is built
        self.build_model()

        # Determine which data to use for BO
        y_bo = self.Y

        # Run appropriate BO algorithm based on objective type
        if self._is_single_task:
            # Single-objective optimization using qlogEI acquisition
            X_next = run_singletask_bo(
                model=self.model,
                bounds=self.bounds,
                train_y=y_bo,
                batch_size=self.batch_size,
                mini_batch_size=self.mini_batch_size,
                device=self.device
            )
        else:
            # Multi-objective optimization using qlogEHVI acquisition
            X_next = run_multitask_bo(
                model=self.model,
                bounds=self.bounds,
                train_y=y_bo,
                ref_point=self.ref_point,
                batch_size=self.batch_size,
                mini_batch_size=self.mini_batch_size,
                device=self.device
            )

        print(X_next)
        return X_next
