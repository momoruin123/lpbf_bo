from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt

from models import black_box
from optimization import warm_start_bo_class
from optimization.utils import read_data


def cost_func(y):
    if not torch.is_tensor(y):
        y = torch.as_tensor(y).unsqueeze(0)
    y = torch.as_tensor(y)
    f = 0.5 * y[:, 0] + 0.5 * y[:, 1]
    f = f.unsqueeze(-1)
    return f


# ==== cycle warm start tast
# read data
X_src, Y_src = read_data(5, 2, '../data/source_task_data.csv')
Y_src = cost_func(Y_src)
X, Y = read_data(5, 2, '../data/target_task_data.csv')
Y = cost_func(Y)
# test parameters
test_iter = 1
n_iter = 10
batch_size = 10
minibatch_size = batch_size
# init log
bsf_history = np.zeros((test_iter, n_iter))
b_history = np.zeros((test_iter, n_iter))
# start cycle

for j in range(test_iter):
    wsbo = warm_start_bo_class.WarmStartBO(5, 1)
    wsbo.set_bounds([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])
    wsbo.add_source_data(X_src, Y_src)
    wsbo.set_batch_size(batch_size, minibatch_size)

    print(f"\n========= Test {j + 1}/{test_iter} =========")
    for i in range(n_iter):
        print(f"\n========= Iteration {i + 1}/{n_iter} =========")
        X_next = wsbo.run_bo()
        Y_next = black_box.transfer_model_2(X_next, 5)
        Y_next = cost_func(Y_next)

        wsbo.add_data(X_next, Y_next)
        b = max(Y_next)
        bsf = max(wsbo.Y)
        b_history[j, i] = b
        bsf_history[j, i] = bsf

b_mean = b_history.mean(axis=0)
bsf_mean = bsf_history.mean(axis=0)  # HV_mean for all test
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
iterations = list(range(1, len(bsf_mean) + 1))
plt.figure(figsize=(8, 6))
# left Y axis
ax1 = plt.gca()
ax1.plot(iterations, bsf_mean, marker='o', label='best_so_far')
ax1.plot(iterations, b_mean, marker='x', label='best_so_far')
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Metric Value (normalized)")
ax1.legend(loc='upper left')
ax1.grid(True)
plt.title(f"batch_size={batch_size} n_iter={n_iter}")
plt.tight_layout()
plt.savefig(f"metrics_value_v4_{timestamp}.png")
plt.close()


# # ==== test warm start bo
# X_src, Y_src = read_data(5, 2, './data/source_task_data.csv')
# Y_src = cost_func(Y_src)
# X, Y = read_data(5, 2, './data/target_task_data.csv')
# Y = cost_func(Y)
#
# ref_point = [10.6221, 11.1111]
# ref_point = cost_func(ref_point)
#
# wsbo = warm_start_bo_class.WarmStartBO(5, 1)
#
# # wsbo.batch_size = 10
# # wsbo.mini_batch_size = 5
#
# wsbo.set_bounds([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])
#
# wsbo.add_source_data(X_src, Y_src)
# print(wsbo.X_src.shape)
# print(wsbo.Y_src.shape)
#
# wsbo.add_data(X, Y)
# print(wsbo.X.shape)
# print(wsbo.Y.shape)
#
# wsbo.ref_point = ref_point
# print(wsbo.get_ref_point())
#
# wsbo.run_bo()

# ====test base bo
# BO = base_bo_class.BaseBO(4, 3)
#
# BO.read_data('./result/target_task_data.csv')
# X, Y = BO.get_train_data()
# # print(X)
# # print(Y)
# BO.set_bounds([0, 0, 0, 0], [200, 1000, 0.15, 0.5])
# A = BO.set_ref_point(0.1)
# print(A)
# BO.run_bo()
