# analyze_sweep_results.py

import pandas as pd
import numpy as np
import yaml


def analyze_sweep_results(csv_file):
    """
    分析 wandb sweep 結果並生成下一階段 Bayes sweep 設定。
    新增支援: alpha (uniform), freeze_layers (categorical list)。
    在 YAML 中添加 # 註解標註最佳值 (從 top 1 配置)。
    """
    df = pd.read_csv(csv_file)
    print("Available columns:", df.columns.tolist())

    # 搜索參數
    search_params = [
        "adamw_lr",
        "muon_lr",
        "alpha",
        "freeze_layers",
    ]

    # 指標
    metric_cols = [
        "test_combined_score",
        "test_accuracy",
        "test_pearson",
        "val_combined_score",
        "val_accuracy",
        "val_pearson",
        "train_loss",
    ]

    available_params = [p for p in search_params if p in df.columns]
    available_metrics = [m for m in metric_cols if m in df.columns]

    print("\n✅ Available search parameters:", available_params)
    print("✅ Available metrics:", available_metrics)

    primary_metric = "test_combined_score"
    if primary_metric not in df.columns:
        print(f"❌ Error: {primary_metric} not found!")
        return None

    df = df.dropna(subset=[primary_metric] + available_params)
    df = df.sort_values(primary_metric, ascending=False)

    print("\n================== SWEEP RESULTS SUMMARY ==================")
    print(f"Total runs: {len(df)}")
    print(f"Best {primary_metric}: {df[primary_metric].max():.4f}")

    top_n = max(5, int(len(df) * 0.2))
    top_df = df.head(top_n)
    print(f"Top {top_n} entries preview:\n")
    print(top_df[available_params + [primary_metric]].to_string(index=False))

    # 提取最佳值 (top 1)
    best_row = top_df.iloc[0]
    best_adamw_lr = best_row.get("adamw_lr", None)
    best_muon_lr = best_row.get("muon_lr", None)
    best_alpha = best_row.get("alpha", None)
    best_freeze_layers = best_row.get("freeze_layers", None)

    # === alpha 分析 ===
    if "alpha" in available_params:
        alpha_mean = top_df["alpha"].mean()
        alpha_min = top_df["alpha"].min()
        alpha_max = top_df["alpha"].max()
        print("\nAlpha summary:")
        print(f"  top range = [{alpha_min:.4f}, {alpha_max:.4f}]")
        alpha_refined_min = max(0.0, alpha_mean * 0.6)
        alpha_refined_max = min(0.3, alpha_mean * 1.4)
        print(
            f"  suggested range for Bayes: [{alpha_refined_min:.4f}, {alpha_refined_max:.4f}]"
        )
    else:
        alpha_refined_min, alpha_refined_max = 0.0, 0.2
        best_alpha = None

    # === freeze_layers 分析 ===
    if "freeze_layers" in available_params:
        mode_value = int(top_df["freeze_layers"].mode()[0])
        counts = top_df["freeze_layers"].value_counts().to_dict()
        print("\nFreeze layers stats:", counts)
        print(f"Most common freeze_layers: {mode_value}")
        candidate_values = sorted(
            list(set(top_df["freeze_layers"].astype(int).tolist()))
        )
    else:
        candidate_values = list(range(0, 13))
        best_freeze_layers = None

    # === AdamW LR 與 Muon LR 精煉範圍 ===
    adamw_refined_min, adamw_refined_max = 0.0000035, 0.00025
    muon_refined_min, muon_refined_max = 0.0001, 0.001
    if "adamw_lr" in df.columns:
        adamw_mean = top_df["adamw_lr"].mean()
        adamw_refined_min = max(0.0000035, adamw_mean * 0.6)
        adamw_refined_max = min(0.00025, adamw_mean * 1.4)
    if "muon_lr" in df.columns:
        muon_mean = top_df["muon_lr"].mean()
        muon_refined_min = max(0.0001, muon_mean * 0.6)
        muon_refined_max = min(0.001, muon_mean * 1.4)

    # === 手動生成 YAML 內容，包含註解 ===
    yaml_content = """program: nlp_hw3.py
method: bayes
metric:
  name: test_combined_score
  goal: maximize
parameters:
"""

    # AdamW LR
    if best_adamw_lr is not None:
        yaml_content += f"  adamw_lr: # Best: {best_adamw_lr:.8f}\n"
        yaml_content += f"    distribution: log_uniform_values\n"
        yaml_content += f"    min: {adamw_refined_min:.8f}\n"
        yaml_content += f"    max: {adamw_refined_max:.8f}\n"
    else:
        yaml_content += """  adamw_lr: # Best: N/A
    distribution: log_uniform_values
    min: 0.0000035
    max: 0.00025
"""

    # Muon LR
    if best_muon_lr is not None:
        yaml_content += f"  muon_lr: # Best: {best_muon_lr:.8f}\n"
        yaml_content += f"    distribution: log_uniform_values\n"
        yaml_content += f"    min: {muon_refined_min:.8f}\n"
        yaml_content += f"    max: {muon_refined_max:.8f}\n"
    else:
        yaml_content += """  muon_lr: # Best: N/A
    distribution: log_uniform_values
    min: 0.0001
    max: 0.001
"""

    # Alpha
    if best_alpha is not None:
        yaml_content += f"  alpha: # Best: {best_alpha:.4f}\n"
        yaml_content += f"    distribution: uniform\n"
        yaml_content += f"    min: {alpha_refined_min:.4f}\n"
        yaml_content += f"    max: {alpha_refined_max:.4f}\n"
    else:
        yaml_content += """  alpha: # Best: N/A
    distribution: uniform
    min: 0.0
    max: 0.2
"""

    # Freeze layers (明確用 list)
    yaml_content += f"  freeze_layers: # Best: {best_freeze_layers}\n"
    yaml_content += (
        f"    values: [{', '.join(map(str, candidate_values))}]  # Top candidates\n"
    )

    # 儲存 YAML
    output_yaml = "sweep_config_bayes.yaml"
    with open(output_yaml, "w") as f:
        f.write(yaml_content)

    print(f"\n✅ New Bayes sweep config saved to {output_yaml}")
    print("Config preview:\n" + yaml_content)

    return yaml_content


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_sweep_results.py sweep_results.csv")
        sys.exit(1)
    analyze_sweep_results(sys.argv[1])
