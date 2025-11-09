import json
import pandas as pd
import numpy as np
from collections import Counter

# 讀取錯誤
with open("./error_analysis.json", "r", encoding="utf-8") as f:
    errors = json.load(f)

# 轉成 DataFrame
df_errors = pd.DataFrame(errors)

print("=" * 60)
print("ERROR ANALYSIS REPORT")
print("=" * 60)

# ===== 指標 1：回歸誤差分析 =====
print("\n1. REGRESSION ERROR ANALYSIS (Relatedness Prediction)")
print(f"   Total regression errors: {len(df_errors)}")
print(f"   Mean error: {df_errors['reg_error'].mean():.4f}")
print(f"   Median error: {df_errors['reg_error'].median():.4f}")
print(f"   Std dev: {df_errors['reg_error'].std():.4f}")
print(f"   Max error: {df_errors['reg_error'].max():.4f}")
print(f"   Min error: {df_errors['reg_error'].min():.4f}")

# 按誤差大小分組
print("\n   Error distribution:")
error_bins = [0.5, 1.0, 1.5, 2.0, 2.5, 5.0]
for i in range(len(error_bins) - 1):
    count = (
        (df_errors["reg_error"] >= error_bins[i])
        & (df_errors["reg_error"] < error_bins[i + 1])
    ).sum()
    pct = count / len(df_errors) * 100
    print(f"   {error_bins[i]:.1f} - {error_bins[i+1]:.1f}: {count:3d} ({pct:5.1f}%)")

# ===== 指標 2：分類誤差分析 =====
print("\n2. CLASSIFICATION ERROR ANALYSIS (Entailment)")
clf_errors = df_errors[~df_errors["clf_correct"]]
print(f"   Total classification errors: {len(clf_errors)}")
print(f"   Error rate: {len(clf_errors) / len(df_errors) * 100:.1f}%")

# 分析混淆矩陣
print("\n   Confusion matrix:")
confusion = {}
for _, row in clf_errors.iterrows():
    key = (row["clf_target"], row["clf_pred"])
    confusion[key] = confusion.get(key, 0) + 1

label_names = {0: "entailment", 1: "neutral", 2: "contradiction"}
for true_label in range(3):
    for pred_label in range(3):
        count = confusion.get((true_label, pred_label), 0)
        if count > 0:
            print(
                f"   {label_names[true_label]:15} → {label_names[pred_label]:15}: {count:3d}"
            )

# ===== 指標 3：難度分析 =====
print("\n3. DIFFICULTY ANALYSIS")

# 回歸難度（看目標值）
print("\n   Regression difficulty (by target score):")
for score in range(6):
    subset = df_errors[df_errors["reg_target"] == score]
    if len(subset) > 0:
        print(
            f"   Target score {score}: {len(subset):3d} errors, mean error={subset['reg_error'].mean():.3f}"
        )

# 分類難度（某些 label 更容易出錯嗎？）
print("\n   Classification difficulty (by true label):")
for true_label in range(3):
    subset = clf_errors[clf_errors["clf_target"] == true_label]
    total = df_errors[df_errors["clf_target"] == true_label]
    if len(total) > 0:
        error_rate = len(subset) / len(total) * 100
        print(
            f"   {label_names[true_label]:15}: {len(subset):3d} errors / {len(total):3d} total ({error_rate:5.1f}%)"
        )

# ===== 指標 4：文本特徵分析 =====
print("\n4. TEXT FEATURE ANALYSIS")


# 句子長度是否影響
def get_length(text):
    return len(text.split()) if isinstance(text, str) else 0


df_errors["premise_len"] = df_errors["premise"].apply(get_length)
df_errors["hypothesis_len"] = df_errors["hypothesis"].apply(get_length)

print(f"\n   Premise length (error cases):")
print(
    f"   Mean: {df_errors['premise_len'].mean():.1f}, Median: {df_errors['premise_len'].median():.1f}"
)

print(f"\n   Hypothesis length (error cases):")
print(
    f"   Mean: {df_errors['hypothesis_len'].mean():.1f}, Median: {df_errors['hypothesis_len'].median():.1f}"
)

# ===== 指標 5：最難的例子 =====
print("\n5. HARDEST EXAMPLES (Top 10 by error magnitude)")
top_errors = df_errors.nlargest(10, "reg_error")[
    ["premise", "hypothesis", "reg_target", "reg_pred", "reg_error"]
]
for idx, (_, row) in enumerate(top_errors.iterrows(), 1):
    print(f"\n   #{idx}")
    print(f"   Premise: {row['premise'][:80]}")
    print(f"   Hypothesis: {row['hypothesis'][:80]}")
    print(
        f"   Target: {row['reg_target']:.1f}, Pred: {row['reg_pred']:.1f}, Error: {row['reg_error']:.3f}"
    )

# 存成 CSV 便於後續檢查
df_errors.to_csv("./errors_detailed.csv", index=False, encoding="utf-8")
print("\n✓ Saved to errors_detailed.csv")


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
plt.savefig("./error_analysis.png", dpi=150)
print("✓ Saved visualization to error_analysis.png")
