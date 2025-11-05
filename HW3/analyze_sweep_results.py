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
    required_cols = ["best_val_accuracy"]
    optional_cols = [
        "lr_adamw",
        "lr_muon",
        "weight_decay",
        "momentum",
        "rnn_type",
        "batch_size",
    ]

    # Filter to columns that exist
    available_optional = [col for col in optional_cols if col in df.columns]
    dropna_cols = required_cols + available_optional

    # Remove rows with missing values in key columns
    df = df.dropna(subset=dropna_cols)

    # Sort by best validation accuracy
    df = df.sort_values("best_val_accuracy", ascending=False)

    print("=" * 60)
    print("SWEEP RESULTS ANALYSIS")
    print("=" * 60)
    print(f"\nTotal runs: {len(df)}")
    print(f"Best accuracy: {df['best_val_accuracy'].max():.4f}")
    print(f"Mean accuracy: {df['best_val_accuracy'].mean():.4f}")
    print(f"Median accuracy: {df['best_val_accuracy'].median():.4f}")

    # Get top performers (top 20%)
    top_n = max(5, int(len(df) * 0.2))
    top_df = df.head(top_n)

    print(f"\n--- Top {top_n} Configurations ---")
    display_cols = [
        col for col in available_optional + ["best_val_accuracy"] if col in df.columns
    ]
    print(top_df[display_cols].to_string(index=False))

    # Analyze learning rate (AdamW)
    if "lr_adamw" in df.columns:
        print("\n" + "=" * 60)
        print("LEARNING RATE (AdamW) ANALYSIS")
        print("=" * 60)
        lr_top = top_df["lr_adamw"]
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
    if "lr_muon" in df.columns:
        print("\n" + "=" * 60)
        print("LEARNING RATE (Muon) ANALYSIS")
        print("=" * 60)
        lr_muon_top = top_df["lr_muon"]
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

    # Analyze momentum
    if "momentum" in df.columns:
        print("\n" + "=" * 60)
        print("MOMENTUM ANALYSIS")
        print("=" * 60)
        mom_top = top_df["momentum"]
        mom_min = mom_top.min()
        mom_max = mom_top.max()
        mom_mean = mom_top.mean()
        mom_median = mom_top.median()

        print(f"Top configs Momentum range: [{mom_min:.6f}, {mom_max:.6f}]")
        print(f"Mean Momentum: {mom_mean:.6f}")
        print(f"Median Momentum: {mom_median:.6f}")

        # Suggest refined momentum range
        mom_refined_min = max(0.5, mom_mean * 0.95)
        mom_refined_max = min(0.999, mom_mean * 1.05)
    else:
        mom_refined_min = 0.85
        mom_refined_max = 0.999

    # Analyze RNN type
    print("\n" + "=" * 60)
    print("RNN TYPE ANALYSIS")
    print("=" * 60)
    if "rnn_type" in df.columns:
        rnn_stats = df.groupby("rnn_type")["best_val_accuracy"].agg(
            ["mean", "max", "count"]
        )
        print(rnn_stats)

        # Get best RNN type
        best_rnn = rnn_stats["mean"].idxmax()
        rnn_counter = Counter(top_df["rnn_type"])
        most_common_rnn = rnn_counter.most_common(1)[0][0]

        print(f"\nBest average performance: {best_rnn}")
        print(f"Most common in top {top_n}: {most_common_rnn}")

        # Decide if we should fix RNN type or keep searching
        if rnn_stats.loc[best_rnn, "mean"] > rnn_stats["mean"].mean() + 0.01:
            use_fixed_rnn = True
            fixed_rnn_type = best_rnn
            print(f"\nRecommendation: Fix RNN type to {best_rnn} (clear winner)")
        else:
            use_fixed_rnn = False
            fixed_rnn_type = None
            print(f"\nRecommendation: Continue exploring RNN types")
    else:
        use_fixed_rnn = False
        fixed_rnn_type = None

    # Analyze batch size
    if "batch_size" in df.columns:
        print("\n" + "=" * 60)
        print("BATCH SIZE ANALYSIS")
        print("=" * 60)
        bs_stats = df.groupby("batch_size")["best_val_accuracy"].agg(
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
        "program": "nlp_hw2.py",
        "method": "bayes",
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 3},
        "parameters": {},
    }

    # Add lr_adamw parameter
    if "lr_adamw" in df.columns:
        config["parameters"]["lr_adamw"] = {
            "distribution": "log_uniform_values",
            "min": float(lr_refined_min),
            "max": float(lr_refined_max),
        }

    # Add lr_muon parameter
    if "lr_muon" in df.columns:
        config["parameters"]["lr_muon"] = {
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

    # Add momentum parameter
    if "momentum" in df.columns:
        config["parameters"]["momentum"] = {
            "distribution": "log_uniform_values",
            "min": float(mom_refined_min),
            "max": float(mom_refined_max),
        }

    # Add RNN type parameter
    if use_fixed_rnn:
        config["parameters"]["rnn_type"] = {"value": fixed_rnn_type}
    else:
        # Keep top 2 RNN types
        if "rnn_type" in df.columns:
            rnn_by_perf = rnn_stats.sort_values("mean", ascending=False)
            top_rnn_types = list(rnn_by_perf.head(2).index)
            config["parameters"]["rnn_type"] = {"values": top_rnn_types}
        else:
            config["parameters"]["rnn_type"] = {"values": ["LSTM", "GRU"]}

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
    if "lr_adamw" in df.columns:
        print(f"  LR (AdamW) range: [{lr_refined_min:.6f}, {lr_refined_max:.6f}]")
    if "lr_muon" in df.columns:
        print(
            f"  LR (Muon) range: [{lr_muon_refined_min:.6f}, {lr_muon_refined_max:.6f}]"
        )
    if "weight_decay" in df.columns:
        print(f"  WD range: [{wd_refined_min:.6f}, {wd_refined_max:.6f}]")
    if "momentum" in df.columns:
        print(f"  Momentum range: [{mom_refined_min:.6f}, {mom_refined_max:.6f}]")

    if use_fixed_rnn:
        print(f"  RNN type: {fixed_rnn_type} (fixed)")
    else:
        print(f"  RNN types: {config['parameters']['rnn_type']['values']}")

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
    numeric_cols.append("best_val_accuracy")

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()["best_val_accuracy"].sort_values(ascending=False)
        print("\nCorrelation with val_accuracy:")
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
