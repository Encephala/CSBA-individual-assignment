#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

specifier = "2023-12-14 14:29:45.986509-[1, 2, 3, 4, 6, 8, 9, 12]"
results = np.load(f"data/bootstrap {specifier}.npy")

# Average results across bootstraps
comparison_ratio = np.mean(results[:, :, 0], axis = 0)
precision_star = np.mean(results[:, :, 1], axis = 0)
recall_star = np.mean(results[:, :, 2], axis = 0)
F1_star = np.mean(results[:, :, 3], axis = 0)
precision = np.mean(results[:, :, 4], axis = 0)
recall = np.mean(results[:, :, 5], axis = 0)
F1 = np.mean(results[:, :, 6], axis = 0)


fig = plt.figure(figsize = (8, 4))
fig.suptitle(f"Performance LSH")

axes = fig.subplots(1, 2)
axes[0].plot(comparison_ratio * 100, recall_star * 100, ".-", color = "orange", label = "Recall")
axes[1].plot(comparison_ratio * 100, precision_star * 100, ".-", color = "blue", label = "Precision")
axes[1].plot(comparison_ratio * 100, F1_star * 100, ".-", color = "green", label = "F1")

axes[0].set_ylabel("Performance (%)")
for i in range(2):
    axes[i].set_xlabel("Fraction of comparisons (%)")
    axes[i].set_xticks(np.arange(0, 71, 10))
    axes[i].legend()

plt.savefig(f"images/FINAL_performance_LSH.png")
plt.savefig(f"images/FINAL_performance_LSH.svg")

# plt.figure()
# plt.title(f"Recall LSH")
# plt.xlabel("Fraction of comparisons (%)")
# plt.ylabel("Recall (%)")
# plt.plot(comparison_ratio * 100, recall_star * 100, ".-", color = "orange", label = "Recall")
# plt.legend()
# plt.savefig(f"images/FINAL_recall_LSH.png")
# plt.savefig(f"images/FINAL_recall_LSH.svg")

# plt.figure()
# plt.title(f"Precision, F1 LSH")
# plt.xlabel("Fraction of comparisons (%)")
# plt.ylabel("Performance (%)")
# plt.plot(comparison_ratio * 100, precision_star * 100, ".-", color = "blue", label = "Precision")
# plt.plot(comparison_ratio * 100, F1_star * 100, ".-", color = "green", label = "F1")
# plt.legend()
# plt.savefig(f"images/FINAL_performance_LSH.png")
# plt.savefig(f"images/FINAL_performance_LSH.svg")

plt.figure()
plt.title(f"Performance final")
plt.xlabel("Fraction of comparisons (%)")
plt.ylabel("Performance (%)")
plt.plot(comparison_ratio * 100, precision * 100, ".-", color = "blue", label = "Precision")
plt.plot(comparison_ratio * 100, recall * 100, ".-", color = "orange", label = "Recall")
plt.plot(comparison_ratio * 100, F1 * 100, ".-", color = "green", label = "F1")
plt.legend()
plt.savefig(f"images/FINAL_performance_final.png")
plt.savefig(f"images/FINAL_performance_final.svg")

plt.show()
