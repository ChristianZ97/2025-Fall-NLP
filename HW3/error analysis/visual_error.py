import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子圖 1：回歸誤差分佈
axes[0, 0].hist(df_errors["reg_error"], bins=30, edgecolor="black")
axes[0, 0].set_xlabel("Regression Error")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("Distribution of Regression Errors")

# 子圖 2：分類混淆矩陣熱力圖
import seaborn as sns

confusion_matrix = np.zeros((3, 3))
for (true_label, pred_label), count in confusion.items():
    confusion_matrix[true_label, pred_label] = count

sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt="g",
    ax=axes[0, 1],
    xticklabels=["entailment", "neutral", "contra"],
    yticklabels=["entailment", "neutral", "contra"],
)
axes[0, 1].set_title("Classification Confusion Matrix")
axes[0, 1].set_ylabel("True Label")
axes[0, 1].set_xlabel("Predicted Label")

# 子圖 3：目標值 vs 誤差
axes[1, 0].scatter(df_errors["reg_target"], df_errors["reg_error"], alpha=0.5)
axes[1, 0].set_xlabel("Target Score")
axes[1, 0].set_ylabel("Prediction Error")
axes[1, 0].set_title("Error vs Target Score")

# 子圖 4：句子長度 vs 誤差
axes[1, 1].scatter(
    df_errors["premise_len"] + df_errors["hypothesis_len"],
    df_errors["reg_error"],
    alpha=0.5,
)
axes[1, 1].set_xlabel("Total Text Length (words)")
axes[1, 1].set_ylabel("Prediction Error")
axes[1, 1].set_title("Error vs Text Length")

plt.tight_layout()
plt.savefig("saved_models/{run_id}/error_analysis.png", dpi=150)
print("✓ Saved visualization to error_analysis.png")
