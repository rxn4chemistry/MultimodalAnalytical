#!/usr/bin/env python
from pathlib import Path

import numpy as np

with Path("list_DD6_test_selected.txt").open("r") as f:
    file_ids = [line.strip() for line in f]
prefix="DD6/"

postfix="/EVAL/t40/y_pred.npy"
file_list = [ prefix + file_id + postfix for file_id in file_ids]
y_data = [np.load(f) for f in file_list]
y_stacked = np.vstack(y_data)
np.save("y_pred_unseen_molecules_selected.npy", y_stacked)
print(y_stacked.shape, "y_pred_unseen_molecules_selected.npy")

postfix="/EVAL/t40/y_true.npy"
file_list = [ prefix + file_id + postfix for file_id in file_ids]
y_data = [np.load(f) for f in file_list]
y_stacked = np.vstack(y_data)
np.save("y_true_unseen_molecules_selected.npy", y_stacked)
print(y_stacked.shape, "y_true_unseen_molecules_selected.npy")

