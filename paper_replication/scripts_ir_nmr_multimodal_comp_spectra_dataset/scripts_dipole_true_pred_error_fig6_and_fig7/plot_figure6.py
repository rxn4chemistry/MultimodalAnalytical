#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

# Load arrays
y_pred_frames = np.load("y_pred_unseen_frames.npy")
y_true_frames = np.load("y_true_unseen_frames.npy")
y_pred_mols = np.load("y_pred_unseen_molecules.npy")
y_true_mols = np.load("y_true_unseen_molecules.npy")

# Compute norms
norm_pred_frames = np.linalg.norm(y_pred_frames, axis=1)
norm_true_frames = np.linalg.norm(y_true_frames, axis=1)

norm_pred_mols = np.linalg.norm(y_pred_mols, axis=1)
norm_true_mols = np.linalg.norm(y_true_mols, axis=1)

# Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

BIG = 22
MEDIUM = 18

# Frames
axs[0, 0].scatter(norm_true_frames, norm_pred_frames, alpha=0.5, color='darkorange', s=10)
axs[0, 0].plot([-4, 4], [-4, 4], 'k--')
axs[0, 0].set_title("Unseen Frames – Norm", fontsize=BIG)
axs[0, 0].set_xlabel("True Norm", fontsize=BIG)
axs[0, 0].set_ylabel("Predicted Norm", fontsize=BIG)
axs[0, 0].grid(True)
axs[0, 0].set_aspect('equal', adjustable='box')
axs[0, 0].set_xlim(-0.1, 3.4)
axs[0, 0].set_ylim(-0.1, 3.4)
axs[0, 0].tick_params(axis='both', labelsize=MEDIUM)

# Molecules
axs[0, 1].scatter(norm_true_mols, norm_pred_mols, alpha=0.5, color='steelblue', s=10)
axs[0, 1].plot([-4, 4], [-4, 4], 'k--')
axs[0, 1].set_title("Unseen Molecules – Norm", fontsize=BIG)
axs[0, 1].set_xlabel("True Norm", fontsize=BIG)
axs[0, 1].grid(True)
axs[0, 1].set_aspect('equal', adjustable='box')
axs[0, 1].set_xlim(-0.1, 3.4)
axs[0, 1].set_ylim(-0.1, 3.4)
axs[0, 1].tick_params(axis='both', labelsize=MEDIUM)

# === Bottom row: XYZ Components ===
# colors = ['tab:blue', 'tab:orange', 'tab:green']
colors = ["tab:red", "tab:green", "tab:blue"]
labels = ['dipole-x', 'dipole-y', 'dipole-z']

ff = 2.9
# Frames
for i in range(3):
    axs[1, 0].scatter(y_true_frames[:, i], y_pred_frames[:, i],
                      label=labels[i], alpha=0.5, s=10, color=colors[i])
plt.grid(True)
axs[1, 0].plot([-4, 4], [-4, 4], 'k--')
axs[1, 0].set_title("Unseen Frames – XYZ", fontsize=BIG)
axs[1, 0].set_xlabel("True Value", fontsize=BIG)
axs[1, 0].set_ylabel("Predicted Value", fontsize=BIG)
# axs[1, 0].legend()
axs[1, 0].legend(fontsize=MEDIUM)
axs[1, 0].grid(True)
axs[1, 0].set_aspect('equal', adjustable='box')
axs[1, 0].set_xlim(-ff, ff)
axs[1, 0].set_ylim(-ff, ff)
axs[1, 0].tick_params(axis='both', labelsize=MEDIUM)

# Molecules
for i in range(3):
    axs[1, 1].scatter(y_true_mols[:, i], y_pred_mols[:, i],
                      label=labels[i], alpha=0.5, s=10, color=colors[i])
axs[1, 1].plot([-4, 4], [-4, 4], 'k--')
axs[1, 1].set_title("Unseen Molecules – XYZ", fontsize=BIG)
axs[1, 1].set_xlabel("True Value", fontsize=BIG)
# axs[1, 1].legend()
axs[1, 1].legend(fontsize=MEDIUM)
axs[1, 1].grid(True)
axs[1, 1].set_aspect('equal', adjustable='box')
axs[1, 1].set_xlim(-ff, ff)
axs[1, 1].set_ylim(-ff, ff)
axs[1, 1].tick_params(axis='both', labelsize=MEDIUM)

plt.tight_layout()
plt.savefig("figure6.png", dpi=600)
plt.show()
plt.close()

