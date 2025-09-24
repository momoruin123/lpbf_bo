import torch
from datetime import datetime

from optimization import embedding_bo_class
from optimization.utils import read_data, save_data, normalize_tensor


# ======== Define cost function here ========
def cost_func(y) -> torch.Tensor:
    """
    Convert raw objective values into a single cost metric for optimization.

    This function normalizes the input objectives and combines them into a single cost value
    using a simple weighted sum. Modify the combination logic based on specific requirements.

    Parameters:
        y: Raw objective values. Can be a list, array, or tensor of shape (n_samples, n_objectives)

    Returns:
        torch.Tensor: Combined cost metric with shape (n_samples, 1)
    """
    # Convert input to tensor and ensure proper shape
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
# Note: This represents the output dimension of the BO process (matches cost function output)

# IF objective_dim = 1:
#   we need cost function at the top of this file, but don't need slack (don't need reference point)
# ELSE objective_dim > 1:
#   need slack (don't need reference point)
#   Slack for setting reference points
slack = 0.1  # set same slack for each dimension
# or slack = [0.1, 0.2, 0.3, 0.1, 0.5]  # set different slack for each dimension

batch_size = 20  # Number of candidate points to generate in each BO iteration

# Embedding feature vectors (task-specific features)
v_src_1 = [1, 0]  # Feature vector representing source task 1
# v_src_2 = [];  # Uncomment to add more source tasks
# v_src_3 = [];
v_trg = [0.6, 10.8]  # Feature vector representing target task

# Parameter bounds for optimization (input parameters + fixed target feature vector)
# Format: [lower bounds for input params + target feature vector, upper bounds matching]
lower_bounds = [0, 0, 0, 0, 0, *v_trg]
upper_bounds = [1, 1, 1, 1, 1, *v_trg]

# Data dimensions (input parameters, raw objectives)
data_dim = [5, 2]  # [number of processing parameters, number of raw product indicators]

# Source task data configuration
source_1_data_path = "./data/source_task_data.csv"  # Path to source task 1 data
# source_2_data_path = "./data/source_task_data.csv"  # Uncomment for additional source tasks
source_data_clm_name = None  # Column names for source data (None uses default ordering)

# ---------------------------- Optional: Target task initial samples ----------------------------
has_sample = False  # Set to True if initial target task samples are available
target_data_path = "./data/target_task_data.csv"  # Path to target task data
target_data_clm_name = None  # Column names for target data (None uses default ordering)

# ---------------------------- Optional: Mini-batch configuration ----------------------------
# Reduce mini_batch_size if encountering memory issues (especially for multi-objective cases)
minibatch_size = batch_size  # Typically matches batch_size unless memory constraints exist

# ======== Initialize and configure Embedding BO ========
# Create EmbeddingBO instance with input, objective, and embedding dimensions
embedding_BO = embedding_bo_class.EmbeddingBO(
    input_dim=input_dim,
    objective_dim=objective_dim,
    vector_dim=len(v_trg)  # Dimension of embedding feature vectors
)

# Set parameter bounds and batch sizes
embedding_BO.set_bounds(lower_bounds, upper_bounds)
embedding_BO.set_batch_size(batch_size, minibatch_size)

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

    if objective_dim == 1:
        # Convert raw objectives to cost metric using the cost function
        Y_target_cost = cost_func(Y_target_initial)
        embedding_BO.add_augment_data(X_target_initial, Y_target_cost, v=v_trg)

    else:
        # Multi-task BO
        Y_target_cost = Y_target_initial
        embedding_BO.add_augment_data(X_target_initial, Y_target_cost, v=v_trg)
        embedding_BO.set_ref_point(slack)

# Load source task data
# ---------------------------- Modify for your source tasks ----------------------------
# Source task 1
X_src_1, Y_src_1 = read_data(
    x_dim=data_dim[0],
    y_dim=data_dim[1],
    file_path=source_1_data_path,
    x_cols=source_data_clm_name,
    y_cols=source_data_clm_name
)

# Convert source task objectives to cost metric
if objective_dim == 1:
    Y_src_1_cost = cost_func(Y_src_1)
else:
    Y_src_1_cost = Y_src_1

# Add augmented source data (input parameters + source feature vector)
embedding_BO.add_augment_data(
    X_new=X_src_1,
    Y_new=Y_src_1_cost,
    v=v_src_1
)

# Uncomment to add additional source tasks
# X_src_2, Y_src_2 = read_data(data_dim[0], data_dim[1], source_2_data_path)
# if objective_dim == 1:
#     Y_src_2_cost = cost_func(Y_src_2)
# else:
#     Y_src_2_cost = Y_src_2
# embedding_BO.add_augment_data(X_src_2, Y_src_2_cost, v_src_2)

# ======== Run Bayesian Optimization ========
# Generate next set of candidate points to evaluate
X_next = embedding_BO.run_bo()

# ======== Save results ========
# Save the recommended candidate points to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

save_data(
    x=X_next,
    file_path="./result",  # Default save directory
    file_name=f"{timestamp}_embedding_bo_candidates"  # Base filename for the results
)
