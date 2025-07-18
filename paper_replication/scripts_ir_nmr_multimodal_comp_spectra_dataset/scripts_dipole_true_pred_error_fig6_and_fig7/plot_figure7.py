#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

# Load arrays
y_pred_mols = np.load("y_pred_unseen_molecules_selected.npy")
y_true_mols = np.load("y_true_unseen_molecules_selected.npy")

# Plot settings
labels = ["dipole-x", "dipole-y", "dipole-z"]
colors = ["tab:red", "tab:green", "tab:blue"]
ff = 3.2  # Axis limit

# Create plot
plt.figure(figsize=(10, 10))

for i in range(3):
    plt.scatter(
        y_true_mols[:, i], y_pred_mols[:, i],
        label=labels[i],
        alpha=0.6,
        s=100,  # Big dots!
        color=colors[i]
    )

plt.plot([-ff, ff], [-ff, ff], 'k--', linewidth=2)
#plt.title("Unseen Molecules – XYZ", fontsize=36)
plt.xlabel("True Value", fontsize=38)
plt.ylabel("Predicted Value", fontsize=38)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(fontsize=32)
plt.grid(True)
plt.axis('equal')
plt.xlim(-ff, ff)
plt.ylim(-ff, ff)

plt.tight_layout()
plt.savefig("figure7_selected-ex-22447-and-11726.png", dpi=600, bbox_inches='tight')
plt.show()
plt.close()


