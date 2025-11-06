# analyze_sweep_results.py

import pandas as pd
import numpy as np
import yaml
from collections import Counter


def analyze_sweep_results(csv_file):
    """
    Analyze sweep results from CSV and generate refined sweep config
    """
    # Load data
    df = pd.read_csv(csv_file)

    # Identify available columns
    print("Available columns:", df.columns.tolist())

    # Determine which columns to use based on what's available
    required_cols = ["test_combined_score"]
    optional_cols = [
        "adamw_lr",
        "muon_lr",
        "weight_decay",
        "dropout_rate",
        "batch_size",
    ]

    # Metric columns for display
    metric_cols = [
        "batch_perplexity",
        "raw_grad_norm",
        "test_accuracy",
        "test_combined_score",
        "test_pearson",
        "train_loss",
        "val_accuracy",
        "val_combined_score",
        "val_pearson",
    ]

    # Filter to columns that exist
    available_optional = [col for col in optional_cols if col in df.columns]
    available_metrics = [col for col in metric_cols if col in df.columns]
    dropna_cols = required_cols + available_optional

    # Use test_combined_score as primary metric for sorting
    primary_metric = "test_combined_score"

    # Remove rows with missing values in key columns
    df = df.dropna(subset=dropna_cols)

    # Sort by primary metric
    df = df.sort_values(primary_metric, ascending=False)

    print("=" * 60)
    print("SWEEP RESULTS ANALYSIS")
    print("=" * 60)
    print(f"\nTotal runs: {len(df)}")
    print(f"Best {primary_metric}: {df[primary_metric].max():.4f}")
    print(f"Mean {primary_metric}: {df[primary_metric].mean():.4f}")
    print(f"Median {primary_metric}: {df[primary_metric].median():.4f}")

    # Get top performers (top 20%)
    top_n = max(5, int(len(df) * 0.2))
    top_df = df.head(top_n)

    print(f"\n--- Top {top_n} Configurations ---")
    display_cols = available_optional + available_metrics
    display_cols = [col for col in display_cols if col in df.columns]
    print(top_df[display_cols].to_string(index=False))

    # Analyze learning rate (AdamW)
    if "adamw_lr" in df.columns:
        print("\n" + "=" * 60)
        print("LEARNING RATE (AdamW) ANALYSIS")
        print("=" * 60)
        lr_top = top_df["adamw_lr"]
        lr_min = lr_top.min()
        lr_max = lr_top.max()
        lr_mean = lr_top.mean()
        lr_median = lr_top.median()

        print(f"Top configs LR (AdamW) range: [{lr_min:.6f}, {lr_max:.6f}]")
        print(f"Mean LR: {lr_mean:.6f}")
        print(f"Median LR: {lr_median:.6f}")

        # Suggest refined LR range (with 30% margin)
        lr_refined_min = max(0.0001, lr_mean * 0.7)
        lr_refined_max = min(0.01, lr_mean * 1.3)
    else:
        lr_refined_min = 0.0005
        lr_refined_max = 0.002

    # Analyze learning rate (Muon)
    if "muon_lr" in df.columns:
        print("\n" + "=" * 60)
        print("LEARNING RATE (Muon) ANALYSIS")
        print("=" * 60)
        lr_muon_top = top_df["muon_lr"]
        lr_muon_min = lr_muon_top.min()
        lr_muon_max = lr_muon_top.max()
        lr_muon_mean = lr_muon_top.mean()
        lr_muon_median = lr_muon_top.median()

        print(f"Top configs LR (Muon) range: [{lr_muon_min:.6f}, {lr_muon_max:.6f}]")
        print(f"Mean LR: {lr_muon_mean:.6f}")
        print(f"Median LR: {lr_muon_median:.6f}")

        # Suggest refined LR range
        lr_muon_refined_min = max(0.0001, lr_muon_mean * 0.7)
        lr_muon_refined_max = min(0.01, lr_muon_mean * 1.3)
    else:
        lr_muon_refined_min = 0.0005
        lr_muon_refined_max = 0.002

    # Analyze weight decay
    if "weight_decay" in df.columns:
        print("\n" + "=" * 60)
        print("WEIGHT DECAY ANALYSIS")
        print("=" * 60)
        wd_top = top_df["weight_decay"]
        wd_min = wd_top.min()
        wd_max = wd_top.max()
        wd_mean = wd_top.mean()
        wd_median = wd_top.median()

        print(f"Top configs WD range: [{wd_min:.6f}, {wd_max:.6f}]")
        print(f"Mean WD: {wd_mean:.6f}")
        print(f"Median WD: {wd_median:.6f}")

        # Suggest refined WD range
        wd_refined_min = max(0.00001, wd_mean * 0.5)
        wd_refined_max = min(0.1, wd_mean * 2.0)
    else:
        wd_refined_min = 0.001
        wd_refined_max = 0.01

    # Analyze dropout rate
    if "dropout_rate" in df.columns:
        print("\n" + "=" * 60)
        print("DROPOUT RATE ANALYSIS")
        print("=" * 60)
        dr_stats = df.groupby("dropout_rate")[primary_metric].agg(
            ["mean", "max", "count"]
        )
        print(dr_stats)

        best_dr = dr_stats["mean"].idxmax()
        dr_counter = Counter(top_df["dropout_rate"])
        most_common_dr = dr_counter.most_common(1)[0][0]

        print(f"\nBest average performance: {best_dr}")
        print(f"Most common in top {top_n}: {most_common_dr}")

        # Decide if we should fix dropout rate
        if dr_stats.loc[best_dr, "mean"] > dr_stats["mean"].mean() + 0.01:
            use_fixed_dr = True
            fixed_dropout_rate = best_dr
            print(f"\nRecommendation: Fix dropout rate to {best_dr} (clear winner)")
        else:
            use_fixed_dr = False
            fixed_dropout_rate = None
            print(f"\nRecommendation: Continue exploring dropout rates")
    else:
        use_fixed_dr = False
        fixed_dropout_rate = None

    # Analyze batch size
    if "batch_size" in df.columns:
        print("\n" + "=" * 60)
        print("BATCH SIZE ANALYSIS")
        print("=" * 60)
        bs_stats = df.groupby("batch_size")[primary_metric].agg(
            ["mean", "max", "count"]
        )
        print(bs_stats)

        best_bs = bs_stats["mean"].idxmax()
        bs_counter = Counter(top_df["batch_size"])
        most_common_bs = bs_counter.most_common(1)[0][0]

        print(f"\nBest average performance: {best_bs}")
        print(f"Most common in top {top_n}: {most_common_bs}")

        # Decide if we should fix batch size
        if bs_stats.loc[best_bs, "mean"] > bs_stats["mean"].mean() + 0.01:
            use_fixed_bs = True
            fixed_batch_size = int(best_bs)
            print(f"\nRecommendation: Fix batch size to {fixed_batch_size}")
        else:
            use_fixed_bs = False
            fixed_batch_size = None
            print(f"\nRecommendation: Continue exploring batch sizes")
    else:
        use_fixed_bs = False
        fixed_batch_size = None

    # Generate refined sweep config
    print("\n" + "=" * 60)
    print("GENERATING REFINED SWEEP CONFIG")
    print("=" * 60)

    config = {
        "program": "nlp_hw3.py",
        "method": "bayes",
        "metric": {"name": "test_combined_score", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 3},
        "parameters": {},
    }

    # Add adamw_lr parameter
    if "adamw_lr" in df.columns:
        config["parameters"]["adamw_lr"] = {
            "distribution": "log_uniform_values",
            "min": float(lr_refined_min),
            "max": float(lr_refined_max),
        }

    # Add muon_lr parameter
    if "muon_lr" in df.columns:
        config["parameters"]["muon_lr"] = {
            "distribution": "log_uniform_values",
            "min": float(lr_muon_refined_min),
            "max": float(lr_muon_refined_max),
        }

    # Add weight decay parameter
    if "weight_decay" in df.columns:
        config["parameters"]["weight_decay"] = {
            "distribution": "log_uniform_values",
            "min": float(wd_refined_min),
            "max": float(wd_refined_max),
        }

    # Add dropout rate parameter
    if use_fixed_dr:
        config["parameters"]["dropout_rate"] = {"value": fixed_dropout_rate}
    else:
        if "dropout_rate" in df.columns:
            dr_by_perf = dr_stats.sort_values("mean", ascending=False)
            top_dropout_rates = list(dr_by_perf.head(2).index)
            config["parameters"]["dropout_rate"] = {"values": top_dropout_rates}
        else:
            config["parameters"]["dropout_rate"] = {"values": [0.1, 0.15]}

    # Add batch size parameter
    if use_fixed_bs:
        config["parameters"]["batch_size"] = {"value": fixed_batch_size}
    else:
        if "batch_size" in df.columns:
            bs_by_perf = bs_stats.sort_values("mean", ascending=False)
            top_batch_sizes = [int(x) for x in bs_by_perf.head(2).index]
            config["parameters"]["batch_size"] = {"values": top_batch_sizes}
        else:
            config["parameters"]["batch_size"] = {"values": [128, 256]}

    # Save to YAML
    output_file = "sweep_config_refined.yaml"
    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Refined sweep config saved to: {output_file}")
    print("\nNew configuration:")
    if "adamw_lr" in df.columns:
        print(f"  AdamW LR range: [{lr_refined_min:.6f}, {lr_refined_max:.6f}]")
    if "muon_lr" in df.columns:
        print(
            f"  Muon LR range: [{lr_muon_refined_min:.6f}, {lr_muon_refined_max:.6f}]"
        )
    if "weight_decay" in df.columns:
        print(f"  Weight decay range: [{wd_refined_min:.6f}, {wd_refined_max:.6f}]")

    if use_fixed_dr:
        print(f"  Dropout rate: {fixed_dropout_rate} (fixed)")
    else:
        print(f"  Dropout rates: {config['parameters']['dropout_rate']['values']}")

    if use_fixed_bs:
        print(f"  Batch size: {fixed_batch_size} (fixed)")
    else:
        print(f"  Batch sizes: {config['parameters']['batch_size']['values']}")

    # Additional insights
    print("\n" + "=" * 60)
    print("ADDITIONAL INSIGHTS")
    print("=" * 60)

    # Correlation analysis
    numeric_cols = [
        col
        for col in available_optional
        if col in df.columns and df[col].dtype in ["float64", "int64"]
    ]
    numeric_cols.extend(available_metrics)

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()[primary_metric].sort_values(ascending=False)
        print(f"\nCorrelation with {primary_metric}:")
        print(corr.to_string())

    # Suggested next steps
    print("\n" + "=" * 60)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 60)
    print("1. Review the refined config: sweep_config_refined.yaml")
    print("2. Run: wandb sweep sweep_config_refined.yaml")
    print("3. Run: wandb agent YOUR_USERNAME/PROJECT/SWEEP_ID")
    print("4. Run 20-30 more experiments with refined ranges")
    print("5. Expected improvement: Monitor for better hyperparameter convergence")

    return config


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_sweep_results.py sweep_results.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    analyze_sweep_results(csv_file)
