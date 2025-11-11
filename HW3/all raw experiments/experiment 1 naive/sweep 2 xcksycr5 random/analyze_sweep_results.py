# analyze_sweep_results.py

import pandas as pd
import numpy as np
import yaml
from collections import Counter


def analyze_sweep_results(csv_file):
    """
    Analyze sweep results from CSV and generate refined sweep config
    只關注：adamw_lr, muon_lr, adamw_weight_decay, muon_weight_decay
    """
    # Load data
    df = pd.read_csv(csv_file)

    # Identify available columns
    print("Available columns:", df.columns.tolist())

    # 定義搜索參數
    search_params = [
        "adamw_lr",
        "muon_lr",
        "adamw_weight_decay",
        "muon_weight_decay",
    ]

    # 指標列
    metric_cols = [
        "test_combined_score",
        "test_accuracy",
        "test_pearson",
        "val_combined_score",
        "val_accuracy",
        "val_pearson",
        "train_loss",
    ]

    # 檢查哪些列存在
    available_params = [col for col in search_params if col in df.columns]
    available_metrics = [col for col in metric_cols if col in df.columns]

    print("\n✅ Available search parameters:", available_params)
    print("✅ Available metrics:", available_metrics)

    # 使用 test_combined_score 作為主指標
    primary_metric = "test_combined_score"
    if primary_metric not in df.columns:
        print(f"❌ Error: {primary_metric} not found in CSV!")
        return None

    # 移除關鍵列有缺失值的行
    dropna_cols = [primary_metric] + available_params
    df = df.dropna(subset=dropna_cols)

    # 按主指標排序
    df = df.sort_values(primary_metric, ascending=False)

    print("\n" + "=" * 70)
    print("SWEEP RESULTS ANALYSIS")
    print("=" * 70)
    print(f"\nTotal runs: {len(df)}")
    print(f"Best {primary_metric}: {df[primary_metric].max():.4f}")
    print(f"Mean {primary_metric}: {df[primary_metric].mean():.4f}")
    print(f"Median {primary_metric}: {df[primary_metric].median():.4f}")
    print(f"Std {primary_metric}: {df[primary_metric].std():.4f}")

    # 取前 20% 的表現者
    top_n = max(5, int(len(df) * 0.2))
    top_df = df.head(top_n)

    print(f"\n--- Top {top_n} Configurations ---")
    display_cols = available_params + [primary_metric]
    display_cols = [col for col in display_cols if col in df.columns]
    print(top_df[display_cols].to_string(index=False))

    # =========== AdamW LR 分析 ===========
    print("\n" + "=" * 70)
    print("ADAMW LEARNING RATE ANALYSIS")
    print("=" * 70)

    if "adamw_lr" in available_params:
        adamw_lr_top = top_df["adamw_lr"]
        adamw_lr_min = adamw_lr_top.min()
        adamw_lr_max = adamw_lr_top.max()
        adamw_lr_mean = adamw_lr_top.mean()
        adamw_lr_median = adamw_lr_top.median()

        print(f"Top configs range: [{adamw_lr_min:.8f}, {adamw_lr_max:.8f}]")
        print(f"Mean: {adamw_lr_mean:.8f}")
        print(f"Median: {adamw_lr_median:.8f}")

        # 建議的細化範圍 (±40% margin)
        adamw_lr_refined_min = max(0.00001, adamw_lr_mean * 0.6)
        adamw_lr_refined_max = min(0.001, adamw_lr_mean * 1.4)

        print(
            f"Suggested refined range for Bayes: [{adamw_lr_refined_min:.8f}, {adamw_lr_refined_max:.8f}]"
        )
    else:
        adamw_lr_refined_min = 0.00005
        adamw_lr_refined_max = 0.00025

    # =========== Muon LR 分析 ===========
    print("\n" + "=" * 70)
    print("MUON LEARNING RATE ANALYSIS")
    print("=" * 70)

    if "muon_lr" in available_params:
        muon_lr_top = top_df["muon_lr"]
        muon_lr_min = muon_lr_top.min()
        muon_lr_max = muon_lr_top.max()
        muon_lr_mean = muon_lr_top.mean()
        muon_lr_median = muon_lr_top.median()

        print(f"Top configs range: [{muon_lr_min:.8f}, {muon_lr_max:.8f}]")
        print(f"Mean: {muon_lr_mean:.8f}")
        print(f"Median: {muon_lr_median:.8f}")

        # 建議的細化範圍
        muon_lr_refined_min = max(0.0001, muon_lr_mean * 0.6)
        muon_lr_refined_max = min(0.002, muon_lr_mean * 1.4)

        print(
            f"Suggested refined range for Bayes: [{muon_lr_refined_min:.8f}, {muon_lr_refined_max:.8f}]"
        )
    else:
        muon_lr_refined_min = 0.0001
        muon_lr_refined_max = 0.0015

    # =========== AdamW Weight Decay 分析 ===========
    print("\n" + "=" * 70)
    print("ADAMW WEIGHT DECAY ANALYSIS")
    print("=" * 70)

    if "adamw_weight_decay" in available_params:
        adamw_wd_top = top_df["adamw_weight_decay"]
        adamw_wd_min = adamw_wd_top.min()
        adamw_wd_max = adamw_wd_top.max()
        adamw_wd_mean = adamw_wd_top.mean()
        adamw_wd_median = adamw_wd_top.median()

        print(f"Top configs range: [{adamw_wd_min:.8f}, {adamw_wd_max:.8f}]")
        print(f"Mean: {adamw_wd_mean:.8f}")
        print(f"Median: {adamw_wd_median:.8f}")

        # 建議的細化範圍 (×0.5 ~ ×2.0)
        adamw_wd_refined_min = max(0.0, adamw_wd_mean * 0.5)
        adamw_wd_refined_max = min(0.1, adamw_wd_mean * 2.0)

        print(
            f"Suggested refined range for Bayes: [{adamw_wd_refined_min:.8f}, {adamw_wd_refined_max:.8f}]"
        )
    else:
        adamw_wd_refined_min = 0.005
        adamw_wd_refined_max = 0.015

    # =========== Muon Weight Decay 分析 ===========
    print("\n" + "=" * 70)
    print("MUON WEIGHT DECAY ANALYSIS")
    print("=" * 70)

    if "muon_weight_decay" in available_params:
        muon_wd_top = top_df["muon_weight_decay"]
        muon_wd_min = muon_wd_top.min()
        muon_wd_max = muon_wd_top.max()
        muon_wd_mean = muon_wd_top.mean()
        muon_wd_median = muon_wd_top.median()

        print(f"Top configs range: [{muon_wd_min:.8f}, {muon_wd_max:.8f}]")
        print(f"Mean: {muon_wd_mean:.8f}")
        print(f"Median: {muon_wd_median:.8f}")

        # 建議的細化範圍
        muon_wd_refined_min = max(0.0, muon_wd_mean * 0.5)
        muon_wd_refined_max = min(0.1, muon_wd_mean * 2.0)

        print(
            f"Suggested refined range for Bayes: [{muon_wd_refined_min:.8f}, {muon_wd_refined_max:.8f}]"
        )
    else:
        muon_wd_refined_min = 0.005
        muon_wd_refined_max = 0.015

    # =========== 相關性分析 ===========
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    numeric_cols = available_params + available_metrics
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()[primary_metric].sort_values(ascending=False)
        print(f"\nCorrelation with {primary_metric}:")
        for col, corr_val in corr.items():
            print(f"  {col:.<40} {corr_val:>8.4f}")

    # =========== 生成 Bayes Sweep 配置 ===========
    print("\n" + "=" * 70)
    print("GENERATING REFINED SWEEP CONFIG FOR BAYES")
    print("=" * 70)

    config = {
        "program": "nlp_hw3.py",
        "method": "bayes",
        "metric": {"name": "test_combined_score", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 3},
        "parameters": {},
    }

    # 添加 AdamW LR
    config["parameters"]["adamw_lr"] = {
        "distribution": "log_uniform_values",
        "min": float(adamw_lr_refined_min),
        "max": float(adamw_lr_refined_max),
    }

    # 添加 Muon LR
    config["parameters"]["muon_lr"] = {
        "distribution": "log_uniform_values",
        "min": float(muon_lr_refined_min),
        "max": float(muon_lr_refined_max),
    }

    # 添加 AdamW Weight Decay
    config["parameters"]["adamw_weight_decay"] = {
        "distribution": "uniform",
        "min": float(adamw_wd_refined_min),
        "max": float(adamw_wd_refined_max),
    }

    # 添加 Muon Weight Decay
    config["parameters"]["muon_weight_decay"] = {
        "distribution": "uniform",
        "min": float(muon_wd_refined_min),
        "max": float(muon_wd_refined_max),
    }

    # 保存為 YAML
    output_file = "sweep_config_bayes.yaml"
    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Refined Bayes sweep config saved to: {output_file}")

    # 打印新配置
    print("\n📊 New Bayes Configuration:")
    print(
        f"  adamw_lr:          [{adamw_lr_refined_min:.8f}, {adamw_lr_refined_max:.8f}]"
    )
    print(
        f"  muon_lr:           [{muon_lr_refined_min:.8f}, {muon_lr_refined_max:.8f}]"
    )
    print(
        f"  adamw_weight_decay: [{adamw_wd_refined_min:.8f}, {adamw_wd_refined_max:.8f}]"
    )
    print(
        f"  muon_weight_decay:  [{muon_wd_refined_min:.8f}, {muon_wd_refined_max:.8f}]"
    )

    # =========== 建議的下一步 ===========
    print("\n" + "=" * 70)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 70)
    print("1. ✅ Review the refined config: sweep_config_bayes.yaml")
    print("2. ✅ Run: wandb sweep sweep_config_bayes.yaml")
    print("3. ✅ Run: wandb agent YOUR_USERNAME/PROJECT/SWEEP_ID")
    print("4. ✅ Run 30-50 more experiments with Bayes method")
    print("5. ✅ Monitor convergence and expected improvement > 95%")

    return config


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_sweep_results.py sweep_results.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    analyze_sweep_results(csv_file)
