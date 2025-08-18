import torch

from optimization import embedding_bo_class
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
# notice: It only represents the output dimension of your BO optimization, and does not represent the dimensions of all
# indicators. (if a cost function is used, it represents the output dimensions of cost func.)

batch_size = 20  # the size of BO batch

v_src_1 = [1, 0]  # feature vector of source task
# v_src_2 = []; ,,,,
v_trg = [0.6, 10.8]  # feature vector of target task

lower_bounds = [0, 0, 0, 0, 0, *v_trg]  # The bounds of optimized parameters + target feature vector
upper_bounds = [1, 1, 1, 1, 1, *v_trg]

# read data
data_dim = [5, 2]  # the dimensions of data, e.g. [Processing parameters, Product indicators]
source_1_data_path = "./data/source_task_data.csv"
# source_2_data_path = "./data/source_task_data.csv"
source_data_clm_name = None  # a directory of column name if you have to point out.

# ---------------------------- Optional_1 ----------------------------
# only use if there is samples of new task: (has_sample = True)
has_sample = False  # if there is samples for new task, set "True"
target_data_path = "./data/target_task_data.csv"
target_data_clm_name = None  # same
# ---------------------------- Optional_2 ----------------------------
# only use it when the memory explodes, normally in multi-objective optimization (objective_dim > 2)
# default: = batch_size
minibatch_size = batch_size

# ======== data read ========
embedding_BO = embedding_bo_class.EmbeddingBO(input_dim, objective_dim, len(v_trg))
embedding_BO.set_bounds(lower_bounds, upper_bounds)
embedding_BO.set_batch_size(batch_size, minibatch_size)

# target task data
if has_sample:  # if there is sample of target task
    X, Y = read_data(data_dim[0], data_dim[1], target_data_path)
    Y = cost_func(Y)
    embedding_BO.add_augment_data(X, Y, v_trg)

# source task data
# ---------------------------- NEED TO CHANGE!!!
X_src_1, Y_src_1 = read_data(data_dim[0], data_dim[1], source_1_data_path)
Y_src_1 = cost_func(Y_src_1)
embedding_BO.add_augment_data(X_src_1, Y_src_1, v_src_1)
# X_src_2, Y_src_2 = read_data(data_dim[0], data_dim[1], source_2_data_path)
# Y_src = cost_func(Y_src_2)
# embedding_BO.add_augment_data(X_src_2, Y_src_2, v_src_2)

# run BO process
X_next = embedding_BO.run_bo()

# save data
save_data(X_next)
