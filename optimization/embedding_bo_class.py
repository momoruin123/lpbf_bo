"""
A Bayesian Optimization (BO) class with embedding-based representations.

This class extends the base BO functionality by incorporating feature vector embeddings,
allowing for augmented input spaces that combine original parameters with embedding features.

:author: Maoyurun Mao
:affiliation: Institut für Strahlwerkzeuge (IFSW), University of Stuttgart
:date: 2025-08-18
"""
import torch

from optimization.base_bo_class import BaseBO
from optimization.utils import run_multitask_bo, run_singletask_bo
from models import SingleTaskGP_model, MultiTaskGP_model


def attach_feature_vector(x: torch.Tensor, v: list | torch.Tensor) -> torch.Tensor:
    """
    Augment input data with feature vectors by appending them as additional dimensions.

    Combines original input features with embedding vectors to create an augmented input space
    for the Gaussian Process model.

    Parameters:
        x: Original input data. Tensor of shape (n_samples, n_features)
            where n_samples is the number of data points and n_features is the input dimension
        v: Feature/embedding vector to attach. Can be a list or tensor of shape (1, n_embedding_features)
            where n_embedding_features is the dimension of the embedding vector

    Returns:
        torch.Tensor: Augmented input data with shape
            (n_samples, n_features + n_embedding_features)
    """
    # Convert feature vector to tensor if needed, matching device and dtype of input
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.double, device=x.device)

    # Repeat feature vector to match number of samples and concatenate with input
    v_repeated = v.repeat(x.shape[0], 1)
    x_augmented = torch.cat((x, v_repeated), dim=1)

    return x_augmented


class EmbeddingBO(BaseBO):
    """
    Bayesian Optimization class with embedding-based input augmentation.

    Extends the BaseBO class to handle input spaces augmented with feature vectors,
    supporting both single-objective and multi-objective optimization scenarios.
    """
    def __init__(self, input_dim: int, objective_dim: int, vector_dim: int,
                 seed: int = None, device: torch.device = None):
        """
        Initialize the EmbeddingBO class with embedding dimension parameters.

        Parameters:
            input_dim: Dimension of original input parameters (e.g., processing parameters)
            objective_dim: Dimension of objective functions (e.g., product quality metrics)
            vector_dim: Dimension of the embedding feature vectors
            seed: Random seed for reproducibility. If None, no fixed seed is set.
            device: Torch device for computations. If None, automatically selects
                   CUDA if available, otherwise uses CPU.
        """
        # Initialize parent class with core dimensions
        super().__init__(input_dim, objective_dim, seed, device)

        # Embedding-specific parameters
        self.vector_dim = vector_dim

        # Initialize data storage with augmented dimensions (input + embedding)
        self.X = torch.empty((0, input_dim + vector_dim), device=self.device)
        self.Y = torch.empty((0, objective_dim), device=self.device)

        # Determine optimization type (single vs multi-objective)
        self._is_single_task = (objective_dim == 1)

    def add_augment_data(self, X_new, Y_new, v: list | torch.Tensor) -> None:
        """
        Append new observations with embedded feature vectors to training data.

        Augments the input data with the provided feature vector before adding it to
        the existing training dataset.

        Parameters:
            X_new: New input parameters to add. Can be array-like or tensor of shape
                   (n_samples, input_dim)
            Y_new: Corresponding objective values. Can be array-like or tensor of shape
                   (n_samples, objective_dim)
            v: Feature vector to embed with the input data. Shape (1, vector_dim)

        Raises:
            AssertionError: If the number of samples in X_new and Y_new do not match
        """
        # Convert inputs to tensors with correct dtype and device
        X_new = torch.as_tensor(X_new, dtype=self.X.dtype, device=self.device)
        Y_new = torch.as_tensor(Y_new, dtype=self.Y.dtype, device=self.device)

        # Validate input consistency
        assert X_new.shape[0] == Y_new.shape[0], \
            "X and Y batch size mismatch - must have same number of samples"

        # Augment input data with feature vector
        X_augmented = attach_feature_vector(X_new, v)

        # Append new data to existing training data
        self.X = torch.cat([self.X, X_augmented], dim=0)
        self.Y = torch.cat([self.Y, Y_new], dim=0)

    def set_bounds(self, lower, upper) -> None:
        """
        Set parameter bounds for the augmented input space (original + embedding features).

        Parameters:
            lower: Lower bounds for augmented input space. Array-like of length
                   (input_dim + vector_dim)
            upper: Upper bounds for augmented input space. Array-like of length
                   (input_dim + vector_dim)

        Raises:
            AssertionError: If bounds dimensions don't match augmented input space or
                           if upper <= lower for any dimension
        """
        # Convert bounds to tensors with correct dtype and device
        lower = torch.as_tensor(lower, dtype=self.X.dtype, device=self.device).view(-1)
        upper = torch.as_tensor(upper, dtype=self.X.dtype, device=self.device).view(-1)

        # Validate bounds dimensions
        expected_dim = self.input_dim + self.vector_dim
        assert (lower.numel() == expected_dim and upper.numel() == expected_dim), \
            f"Bounds dimension mismatch - expected {expected_dim} dimensions (input_dim + vector_dim)"
        assert torch.all(upper >= lower), "All upper bounds must be greater than or equal to lower bounds"

        # Store bounds as a stacked tensor [2, augmented_dim]
        self.bounds = torch.stack([lower, upper], dim=0)

    def build_model(self) -> SingleTaskGP_model:
        """
        Build and return a Gaussian Process model using the augmented training data.

        Constructs a GP model using the combined input + embedding feature data.
        Must be called after adding training data with add_augment_data().

        Returns:
            SingleTaskGP_model: Fitted Gaussian Process model using augmented data

        Raises:
            AssertionError: If no training data is available (X or Y is empty)
        """
        # Ensure training data is available
        assert self.X.numel() > 0 and self.Y.numel() > 0, \
            "No training data available - call add_augment_data() before building model"

        # Build and store the GP model
        model = SingleTaskGP_model.build_model(self.X, self.Y)
        self.model = model

        return model

    def run_bo(self) -> torch.Tensor:
        """
        Run a single iteration of Bayesian Optimization with embedded features.

        Uses either single-task or multi-task BO depending on the objective dimension,
        returning the next set of points to evaluate in the original (non-augmented)
        input space.

        Returns:
            torch.Tensor: Next set of points to evaluate with shape (batch_size, input_dim)
        """
        # Ensure model is built with current data
        self.build_model()

        # Determine which data to use for BO (falls back to existing data if none)
        y_bo = self.Y if self.X.nelement() > 0 else self.Y

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

        # Print and return only the original input dimensions (excluding embedding)
        print(X_next[:, 0:self.input_dim])
        return X_next
