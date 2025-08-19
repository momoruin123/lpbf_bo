import torch

from optimization import warm_start_bo_class
from optimization.utils import read_data, save_data, normalize_tensor


# ======== Define cost function here ========
def cost_func(y) -> torch.Tensor:
    """
    Convert raw objective values into a single cost metric for warm-start optimization.

    This function normalizes input objectives and combines them into a single cost value
    using a weighted sum. Adjust the combination logic based on specific optimization goals.

    Parameters:
        y: Raw objective values. Can be a list, array, or tensor of shape (n_samples, n_objectives)

    Returns:
        torch.Tensor: Combined cost metric with shape (n_samples, 1)
    """
    # Convert input to tensor and ensure proper dimensions
    if not torch.is_tensor(y):
        y = torch.as_tensor(y).unsqueeze(0)
    y = torch.as_tensor(y)

    # Normalize objectives to [0, 1] range for consistent weighting
    y_normalized = normalize_tensor(y)

    # -------- Modify this section for your specific cost function --------
    # Combine normalized objectives (example: equal weights for two objectives)
    f = 0.5 * y_normalized[:, 0] + 0.5 * y_normalized[:, 1]
    # ---------------------------------------------------------------------

    return f.unsqueeze(-1)  # Ensure output shape matches BO's expected format


# ======== Configuration parameters ========
# ---------------------------- Modify these parameters as needed ----------------------------
input_dim = 5  # Number of input dimensions (e.g., processing parameters to optimize)
objective_dim = 1  # Dimension of the optimization target (1 for single-objective cost function)
# Note: This matches the output dimension of the cost function, not the raw objectives

batch_size = 20  # Number of candidate points to generate in each BO iteration

# Parameter bounds for the optimization space
lower_bounds = [0, 0, 0, 0, 0]  # Lower bounds for each input dimension
upper_bounds = [1, 1, 1, 1, 1]  # Upper bounds for each input dimension

# Data dimensions (input parameters, raw objectives)
data_dim = [5, 2]  # [number of processing parameters, number of raw product indicators]

# Source task data configuration (for warm start)
source_data_path = "./data/source_task_data.csv"  # Path to related source task data
source_data_clm_name = None  # Column names for source data (None uses default ordering)

# Target task initial samples (optional)
has_sample = False  # Set to True if initial target task data is available
target_data_path = "./data/target_task_data.csv"  # Path to target task data
target_data_clm_name = None  # Column names for target data (None uses default ordering)

# ---------------------------- Optional: Mini-batch configuration ----------------------------
# Reduce if encountering memory issues (especially for multi-objective scenarios)
minibatch_size = batch_size  # Typically matches batch_size unless memory constraints exist


# ======== Initialize and configure Warm Start BO ========
# Create WarmStartBO instance with input and objective dimensions
warm_start_BO = warm_start_bo_class.WarmStartBO(
    input_dim=input_dim,
    objective_dim=objective_dim
)

# Configure optimization bounds and batch sizes
warm_start_BO.set_bounds(lower_bounds, upper_bounds)
warm_start_BO.set_batch_size(batch_size, minibatch_size)


# ======== Load initial data ========
# Load target task data if available
if has_sample:
    # Read target task data (input parameters and raw objectives)
    X_target_initial, Y_target_initial = read_data(
        x_dim=data_dim[0],
        y_dim=data_dim[1],
        file_path=target_data_path,
        x_cols=target_data_clm_name,
        y_cols=target_data_clm_name
    )

    # Convert raw objectives to cost metric using the cost function
    Y_target_cost = cost_func(Y_target_initial)

    # Add target data to BO (uses base class's add_data method)
    warm_start_BO.add_data(X_target_initial, Y_target_cost)

# Load source task data for warm start
X_src, Y_src = read_data(
    x_dim=data_dim[0],
    y_dim=data_dim[1],
    file_path=source_data_path,
    x_cols=source_data_clm_name,
    y_cols=source_data_clm_name
)

# Convert source objectives to cost metric
Y_src_cost = cost_func(Y_src)

# Add source data to warm start BO
warm_start_BO.add_source_data(X_src, Y_src_cost)


# ======== Run Warm Start Bayesian Optimization ========
# Generate next set of candidate points to evaluate
X_next = warm_start_BO.run_bo()


# ======== Save results ========
# Save the recommended candidate points to CSV
save_data(
    x=X_next,
    file_path="./result",  # Default save directory
    file_name="warm_start_bo_candidates"  # Base filename for results
)