"""
A warm start Bayesian Optimization method class that leverages source task data to accelerate convergence.

This class extends the base BO functionality by incorporating data from related source tasks,
using either single-task or multi-task Gaussian Processes depending on the availability of target task data.

:author: Maoyurun Mao
:affiliation: Institut für Strahlwerkzeuge (IFSW), University of Stuttgart
:date: 2025-08-12
"""
import torch

from models import SingleTaskGP_model, MultiTaskGP_model
from optimization.base_bo_class import BaseBO
from optimization.utils import run_multitask_bo, run_singletask_bo


class WarmStartBO(BaseBO):
    """
    Warm Start Bayesian Optimization class that utilizes source task data for accelerated optimization.

    Combines data from a related source task with target task data (when available) to build more informative
    surrogate models, enabling faster convergence compared to cold-start BO. Supports both single-objective
    and multi-objective optimization scenarios.
    """
    def __init__(self, input_dim: int, objective_dim: int, seed: int = None, device: torch.device = None):
        """
        Initialize the WarmStartBO class with source task data storage.

        Parameters:
            input_dim: Dimension of input parameters (e.g., processing variables)
            objective_dim: Dimension of objective functions (e.g., performance metrics)
            seed: Random seed for reproducibility. If None, no fixed seed is set.
            device: Torch device for computations. If None, automatically selects
                   CUDA if available, otherwise uses CPU.
        """
        # Initialize parent class with core dimensions
        super().__init__(input_dim, objective_dim, seed, device)

        # Source task data storage
        self.X_src = torch.empty((0, input_dim), device=self.device)  # Source task inputs [N_src, input_dim]
        self.Y_src = torch.empty((0, objective_dim), device=self.device)  # Source task outputs [N_src, objective_dim]

        # Determine optimization type (single vs multi-objective)
        self._is_single_task = (objective_dim == 1)

    def add_source_data(self, X_src, Y_src) -> None:
        """
        Append new observations from the source task to the existing source data.

        Converts input data to appropriate tensor format and concatenates with existing source data.

        Parameters:
            X_src: New source task input parameters. Can be array-like or tensor of shape
                   (n_samples, input_dim)
            Y_src: Corresponding source task objective values. Can be array-like or tensor of shape
                   (n_samples, objective_dim)

        Raises:
            AssertionError: If the number of samples in X_src and Y_src do not match
        """
        # Convert to tensors and ensure correct shape and device
        X_src = torch.as_tensor(X_src, dtype=self.X_src.dtype, device=self.device).view(-1, self.input_dim)
        Y_src = torch.as_tensor(Y_src, dtype=self.Y_src.dtype, device=self.device).view(-1, self.objective_dim)

        # Validate input consistency
        assert X_src.shape[0] == Y_src.shape[0], \
            "X_src and Y_src batch size mismatch - must have same number of samples"

        # Append new source data
        self.X_src = torch.cat([self.X_src, X_src], dim=0)
        self.Y_src = torch.cat([self.Y_src, Y_src], dim=0)

    def build_model(self) -> SingleTaskGP_model or MultiTaskGP_model:
        """
        Build and return a Gaussian Process model using a combination of source and target task data.

        Uses different model types based on target task data availability:
        - SingleTaskGP with source data when no target data exists
        - MultiTaskGP combining source and target data when target data is available

        Returns:
            Fitted Gaussian Process model (either SingleTaskGP or MultiTaskGP)

        Raises:
            AssertionError: If no source task data is available
        """
        # Ensure source task data is available
        assert self.X_src.numel() > 0 and self.Y_src.numel() > 0, \
            "No source task data available - call add_source_data() before building model"

        if self.X.nelement() == 0:
            # Use only source task data when no target task data exists
            model = SingleTaskGP_model.build_model(self.X_src, self.Y_src)
        else:
            # Combine source and target task data using a multi-task model
            model = MultiTaskGP_model.build_model(self.X_src, self.Y_src, self.X, self.Y)

        self.model = model
        return model

    def run_bo(self) -> torch.Tensor:
        """
        Run a single iteration of warm-start Bayesian Optimization.

        Uses appropriate acquisition functions based on objective type (single/multi) and
        selects between source and target data for acquisition function optimization based
        on target data availability.

        Returns:
            torch.Tensor: Next set of points to evaluate with shape (batch_size, input_dim)
        """
        # Build the appropriate model (single-task or multi-task)
        self.build_model()

        # Select data for acquisition function optimization
        if self.X.nelement() == 0:
            # Use source task data when no target data is available
            y_bo = self.Y_src
        else:
            # Use target task data once available
            y_bo = self.Y

        # Run appropriate BO algorithm based on objective type
        if self._is_single_task:
            # Single-objective optimization using qLogEI acquisition
            X_next = run_singletask_bo(
                model=self.model,
                bounds=self.bounds,
                train_y=y_bo,
                batch_size=self.batch_size,
                mini_batch_size=self.mini_batch_size,
                device=self.device
            )
        else:
            # Multi-objective optimization using qLogEHVI acquisition
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