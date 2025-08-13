import torch

from optimization import warm_start_bo_class
from optimization.utils import read_data, save_data, normalize_tensor


# ======== define cost function here ========
def cost_func(y):
    if not torch.is_tensor(y):  # convert to torch
        y = torch.as_tensor(y).unsqueeze(0)
    y = torch.as_tensor(y)
    y = normalize_tensor(y)  # normalization

    # -------- cost func. -------- NEED TO CHANGE!!!
    f = 0.5 * y[:, 0] + 0.5 * y[:, 1]
    # ----------------------------

    return f.unsqueeze(-1)


# ======== parameters ========
# ---------------------------- NEED TO CHANGE!!!
input_dim = 5  # BO input dimension
objective_dim = 1  # BO output dimension
# notice: It only represents the output dimension of your BO optimization,
# and does not represent the dimensions of all indicators.
batch_size = 20
lower_bounds = [0, 0, 0, 0, 0]
upper_bounds = [1, 1, 1, 1, 1]
data_dim = [5, 2]  # the dimensions of [Processing parameters, Product indicators]
source_data_path = "./data/source_task_data.csv"
source_data_clm_name = None  # a directory of column name if you have to point out.
has_sample = False  # if there is samples for new task, set "True"
target_data_path = "./data/target_task_data.csv"
target_data_clm_name = None  # same
# ----------------------------
# only use it when the memory explodes, normally in multi-objective optimization (objective_dim > 2)
# default: = batch_size
minibatch_size = batch_size

# ======== data read ========
warm_start_BO = warm_start_bo_class.WarmStartBO(input_dim, objective_dim)
warm_start_BO.set_bounds(lower_bounds, upper_bounds)
warm_start_BO.set_batch_size(batch_size, minibatch_size)

# target task data
if has_sample:  # if there is sample of target task
    X, Y = read_data(data_dim[0], data_dim[1], target_data_path)
    Y = cost_func(Y)
    warm_start_BO.add_data(X, Y)

# source task data
X_src, Y_src = read_data(data_dim[0], data_dim[1], source_data_path)
Y_src = cost_func(Y_src)
warm_start_BO.add_source_data(X_src, Y_src)

# run BO process
X_next = warm_start_BO.run_bo()

# save data
save_data(X_next)
