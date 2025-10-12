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

    # Remove rows with missing values
    df = df.dropna(subset=["best_val_accuracy", "lr", "weight_decay"])

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
    print(
        top_df[["lr", "weight_decay", "rnn_type", "best_val_accuracy"]].to_string(
            index=False
        )
    )

    # Analyze learning rate
    print("\n" + "=" * 60)
    print("LEARNING RATE ANALYSIS")
    print("=" * 60)
    lr_top = top_df["lr"]
    lr_min = lr_top.min()
    lr_max = lr_top.max()
    lr_mean = lr_top.mean()
    lr_median = lr_top.median()

    print(f"Top configs LR range: [{lr_min:.6f}, {lr_max:.6f}]")
    print(f"Mean LR: {lr_mean:.6f}")
    print(f"Median LR: {lr_median:.6f}")

    # Suggest refined LR range (with 20% margin)
    lr_refined_min = max(0.0001, lr_mean * 0.7)
    lr_refined_max = min(0.1, lr_mean * 1.3)

    # Analyze weight decay
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
        if rnn_stats.loc[best_rnn, "mean"] > rnn_stats["mean"].mean() + 0.1:
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

    # Generate refined sweep config
    print("\n" + "=" * 60)
    print("GENERATING REFINED SWEEP CONFIG")
    print("=" * 60)

    config = {
        "program": "nlp_hw2.py",
        "method": "bayes",
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 3},
        "parameters": {
            "lr": {
                "distribution": "log_uniform_values",
                "min": float(lr_refined_min),
                "max": float(lr_refined_max),
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": float(wd_refined_min),
                "max": float(wd_refined_max),
            },
        },
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

    # Save to YAML
    output_file = "sweep_config_refined.yaml"
    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Refined sweep config saved to: {output_file}")
    print("\nNew configuration:")
    print(f"  LR range: [{lr_refined_min:.6f}, {lr_refined_max:.6f}]")
    print(f"  WD range: [{wd_refined_min:.6f}, {wd_refined_max:.6f}]")
    if use_fixed_rnn:
        print(f"  RNN type: {fixed_rnn_type} (fixed)")
    else:
        print(f"  RNN types: {config['parameters']['rnn_type']['values']}")

    # Additional insights
    print("\n" + "=" * 60)
    print("ADDITIONAL INSIGHTS")
    print("=" * 60)

    # Correlation analysis
    numeric_cols = ["lr", "weight_decay", "best_val_accuracy"]
    if all(col in df.columns for col in numeric_cols):
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
    print("5. Expected improvement: 3-5% accuracy increase")

    return config


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_sweep_results.py sweep_results.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    analyze_sweep_results(csv_file)
