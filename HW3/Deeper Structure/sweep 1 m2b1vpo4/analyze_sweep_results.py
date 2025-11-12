#!/usr/bin/env python3
# analyze_sweep_results.py

"""
Analyzes hyperparameter sweep results from a W&B CSV export to generate a
refined sweep configuration for the next iteration.
"""

import argparse
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


class FlowList(list):
    """A list subclass to signal flow-style YAML dumping."""

    pass


def flow_list_representer(dumper, data):
    """Custom representer to dump FlowList as a flow-style sequence."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(FlowList, flow_list_representer, Dumper=yaml.SafeDumper)


class SweepAnalyzer:
    """Analyzes W&B Sweep results and generates a refined configuration."""

    KNOWN_METADATA_PREFIXES = ("_", "wandb", "system.")
    KNOWN_METRIC_KEYWORDS = (
        "loss",
        "accuracy",
        "acc",
        "score",
        "f1",
        "auc",
        "roc",
        "precision",
        "recall",
        "pearson",
        "bleu",
        "rouge",
        "perplexity",
        "ppl",
        "val_",
        "valid_",
        "test_",
        "eval_",
        "train_",
        "epoch",
        "step",
        "iter",
        "duration",
        "runtime",
        "timestamp",
        "grad_norm",
    )

    LOG_SCALE_KEYWORDS = [
        r".*lr$",
        "alpha",
        "beta",
        "eps",
        "epsilon",
        r".*weight_decay$",
        "decay",
        "gamma",
        r".*momentum$",
    ]

    DISCRETE_INT_THRESHOLD = 8

    def __init__(
        self,
        csv_file: str,
        original_config: Optional[str] = None,
        primary_metric: str = "test/combined_score",
        top_percent: float = 0.2,
    ):
        self.csv_file = Path(csv_file)
        self.original_config_path = Path(original_config) if original_config else None
        self.primary_metric = primary_metric
        self.top_percent = top_percent

        self.df: Optional[pd.DataFrame] = None
        self.search_params: List[str] = []
        self.original_params: Dict[str, Any] = {}

    def run_analysis(self) -> Dict[str, Any]:
        """Executes the full analysis pipeline."""
        self.load_data()
        self.detect_parameters()
        if not self.search_params:
            raise ValueError(
                "No hyperparameters were detected. Check your CSV or provide the original sweep config."
            )
        self.preprocess_data()
        top_df = self.analyze_performance()

        param_configs = {}
        for param in self.search_params:
            config = self._analyze_single_parameter(param, top_df)
            param_configs[param] = config

        self.analyze_correlation()
        self.rank_parameter_importance()
        refined_config = self.generate_refined_config(param_configs)
        self.print_best_config()
        self.print_recommendations()

        return refined_config

    def load_data(self) -> None:
        """Loads and provides an overview of the sweep data."""
        print(f"\n{'='*70}\n1. LOADING DATA\n{'='*70}")
        if not self.csv_file.exists():
            raise FileNotFoundError(f"Error: CSV file not found at '{self.csv_file}'")

        self.df = pd.read_csv(self.csv_file)
        print(f"✓ Loaded {len(self.df)} runs from '{self.csv_file.name}'.")
        print(f"  Found {len(self.df.columns)} available columns.")

        if self.original_config_path and self.original_config_path.exists():
            with open(self.original_config_path, "r") as f:
                config = yaml.safe_load(f)
                self.original_params = config.get("parameters", {})
            print(f"✓ Loaded original config from '{self.original_config_path.name}'.")
            if self.original_params:
                print(f"  Found {len(self.original_params)} parameters in config.")

    def detect_parameters(self) -> None:
        """Automatically detects hyperparameter columns in the DataFrame."""
        print(f"\n{'='*70}\n2. DETECTING HYPERPARAMETERS\n{'='*70}")
        if self.df is None:
            return

        if self.original_params:
            print("✓ Using original config to identify hyperparameters.")
            config_params = list(self.original_params.keys())
            self.search_params = [
                p
                for p in config_params
                if p in self.df.columns and self.df[p].nunique() > 1
            ]
            missing_params = set(config_params) - set(self.search_params)
            if missing_params:
                print(
                    f"  (Note: Some params from config not in CSV or not varied: {list(missing_params)})"
                )
        else:
            print(
                "⚠ Original config not found. Using heuristics to detect hyperparameters."
            )
            potential_params = []
            for col in self.df.columns:
                if self._is_metric_or_metadata_column(col):
                    continue
                if not self._is_numeric_column(col):
                    continue
                if self.df[col].nunique() <= 1:
                    continue
                potential_params.append(col)
            self.search_params = potential_params

        print(f"✓ Detected {len(self.search_params)} hyperparameters for analysis:")
        if not self.search_params:
            print("  -> No hyperparameters found.")
        for i, param in enumerate(self.search_params, 1):
            nunique = self.df[param].nunique()
            print(f"  {i:2d}. {param:.<40} ({nunique} unique values)")

    def preprocess_data(self) -> None:
        """Cleans and prepares the data for analysis."""
        print(f"\n{'='*70}\n3. PREPROCESSING DATA\n{'='*70}")
        if self.df is None or self.primary_metric not in self.df.columns:
            raise ValueError(f"Metric '{self.primary_metric}' not found in the data!")

        original_len = len(self.df)
        for param in self.search_params:
            self.df[param] = pd.to_numeric(self.df[param], errors="coerce")
        dropna_cols = [self.primary_metric] + self.search_params
        self.df.dropna(subset=dropna_cols, inplace=True)
        if len(self.df) < original_len:
            print(f"⚠ Removed {original_len - len(self.df)} rows with missing values.")

        metric_mean = self.df[self.primary_metric].mean()
        metric_std = self.df[self.primary_metric].std()
        outlier_threshold = 3
        before_outlier_removal = len(self.df)
        self.df = self.df[
            np.abs(self.df[self.primary_metric] - metric_mean)
            <= (outlier_threshold * metric_std)
        ]
        if len(self.df) < before_outlier_removal:
            print(f"⚠ Removed {before_outlier_removal - len(self.df)} metric outliers.")

        print(f"✓ {len(self.df)} valid runs remaining for analysis.")
        self.df.sort_values(self.primary_metric, ascending=False, inplace=True)

    def analyze_performance(self) -> pd.DataFrame:
        """Analyzes and displays overall performance statistics."""
        print(f"\n{'='*70}\n4. ANALYZING PERFORMANCE\n{'='*70}")
        if self.df is None:
            raise ValueError("DataFrame not loaded.")
        stats = self.df[self.primary_metric].describe()
        print(
            f"Statistics for '{self.primary_metric}':\n"
            f"  Best:     {stats['max']:.6f}\n  Mean:     {stats['mean']:.6f}\n"
            f"  Std Dev:  {stats['std']:.6f}\n  Worst:    {stats['min']:.6f}"
        )

        top_n = max(5, int(len(self.df) * self.top_percent))
        top_df = self.df.head(top_n)

        print(f"\n--- Top {top_n} Configurations (Top {self.top_percent:.0%}) ---")
        display_cols = self.search_params + [self.primary_metric]
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(top_df[display_cols].head(10).to_string(index=False))
        return top_df

    def _analyze_single_parameter(
        self, param: str, top_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyzes a single parameter and suggests a refined search space."""
        print(f"\n--- Analyzing: {param.upper()} ---")

        is_discrete = False
        if param in self.original_params and "values" in self.original_params[param]:
            is_discrete = True
            print("  Detected as DISCRETE from original config.")
        elif self.df[
            param
        ].nunique() <= self.DISCRETE_INT_THRESHOLD and pd.api.types.is_integer_dtype(
            self.df[param].dropna()
        ):
            is_discrete = True
            print(
                "  Heuristically detected as DISCRETE (integer with few unique values)."
            )

        if is_discrete:
            value_counts = top_df[param].value_counts()
            best_overall_value = self.df.iloc[0][param]
            recommended_values = set(value_counts.nlargest(3).index)
            recommended_values.add(best_overall_value)
            final_values = sorted(
                [
                    v.item() if isinstance(v, np.generic) else v
                    for v in recommended_values
                ]
            )
            print(f"  Top-performing values: {list(value_counts.index)}")
            print(f"  Recommended Values: {final_values}")
            return {"values": final_values}
        else:
            print("  Analyzed as a CONTINUOUS parameter.")
            param_values = top_df[param]
            stats = param_values.describe().to_dict()
            print(f"  Top runs range: [{stats['min']:.6g}, {stats['max']:.6g}]")
            print(f"  Mean / Median:  {stats['mean']:.6g} / {stats['50%']:.6g}")
            distribution = self._infer_distribution(param, stats)
            refined_range = self._compute_refined_range(stats, distribution)
            print(f"  Recommended Distribution: {distribution}")
            print(
                f"  Refined Search Range:   [{refined_range['min']:.6g}, {refined_range['max']:.6g}]"
            )
            return {
                "distribution": distribution,
                "min": float(refined_range["min"]),
                "max": float(refined_range["max"]),
            }

    def analyze_correlation(self) -> None:
        """Calculates the correlation of parameters with the metric."""
        print(f"\n{'='*70}\n5. ANALYZING CORRELATION\n{'='*70}")
        if self.df is None:
            return
        numeric_params = [
            p for p in self.search_params if pd.api.types.is_numeric_dtype(self.df[p])
        ]
        if not numeric_params:
            print("⚠ No numeric hyperparameters to analyze for correlation.")
            return

        corr = (
            self.df[numeric_params + [self.primary_metric]]
            .corr(numeric_only=True)[self.primary_metric]
            .sort_values(ascending=False)
        )
        print(f"Correlation with '{self.primary_metric}':")
        for col, corr_val in corr.items():
            if col != self.primary_metric and col in self.search_params:
                strength = self._correlation_strength(abs(corr_val))
                print(f"  {col:.<40} {corr_val:>8.4f}  ({strength})")

    def rank_parameter_importance(self) -> None:
        """Ranks parameters by importance based on correlation."""
        print(f"\n{'='*70}\n6. RANKING PARAMETER IMPORTANCE\n{'='*70}")
        if self.df is None:
            return
        numeric_params = [
            p for p in self.search_params if pd.api.types.is_numeric_dtype(self.df[p])
        ]
        if not numeric_params:
            print("⚠ No numeric hyperparameters to rank.")
            return

        corr = (
            self.df[numeric_params + [self.primary_metric]]
            .corr(numeric_only=True)[self.primary_metric]
            .drop(self.primary_metric)
            .abs()
            .sort_values(ascending=False)
        )
        print("Importance ranked by absolute correlation (for numeric params):")
        for i, (param, corr_val) in enumerate(corr.items(), 1):
            bar = "█" * int(corr_val * 40)
            print(f"  {i:2d}. {param:.<40} {corr_val:.4f} {bar}")

    def generate_refined_config(self, param_configs: Dict) -> Dict[str, Any]:
        """Generates and saves the new sweep configuration YAML file."""
        print(f"\n{'='*70}\n7. GENERATING REFINED CONFIG\n{'='*70}")
        program, method, metric_goal = "train.py", "bayes", "maximize"
        if self.original_config_path and self.original_config_path.exists():
            with open(self.original_config_path, "r") as f:
                original = yaml.safe_load(f)
                program = original.get("program", program)
                method = original.get("method", method)
                if "metric" in original:
                    metric_goal = original["metric"].get("goal", metric_goal)

        for param, config in param_configs.items():
            if "values" in config and isinstance(config["values"], list):
                config["values"] = FlowList(config["values"])

        new_config = {
            "program": program,
            "method": method,
            "metric": {"name": self.primary_metric, "goal": metric_goal},
            "parameters": param_configs,
        }

        output_file = "sweep_config_refined.yaml"
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(
                new_config,
                f,
                Dumper=yaml.SafeDumper,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        print(f"✓ Refined sweep configuration saved to: {output_file}")
        return new_config

    # MODIFIED: This function now converts numpy types to native Python types before printing.
    def print_best_config(self) -> None:
        """Prints the best hyperparameter set in a clean, copy-pasteable format."""
        print(f"\n{'='*70}\n8. EXTRACTING BEST CONFIGURATION\n{'='*70}")
        if self.df is None or self.df.empty:
            return

        best_run = self.df.iloc[0]
        best_config = {param: best_run[param] for param in self.search_params}

        print("Copy-pasteable dictionary of the best performing hyperparameters:")
        print("\ndefault_config = {")
        for i, (key, value) in enumerate(best_config.items()):
            comma = "," if i < len(best_config) - 1 else ""

            # Convert numpy types (e.g., np.float64) to native Python types for clean output.
            py_value = value
            if isinstance(value, np.generic):
                py_value = value.item()

            print(f'    "{key}": {repr(py_value)}{comma}')
        print("}")
        print(
            f"\nAchieved a '{self.primary_metric}' of: {best_run[self.primary_metric]:.6f}"
        )

    def print_recommendations(self) -> None:
        """Prints recommended next steps."""
        if self.df is None or self.df.empty:
            return
        best_score = self.df[self.primary_metric].max()
        print(f"\n{'='*70}\n9. RECOMMENDED NEXT STEPS\n{'='*70}")
        print("1. Review the generated 'sweep_config_refined.yaml'.")
        print(
            "2. Launch the new, focused sweep:\n   wandb sweep sweep_config_refined.yaml"
        )
        print("3. Start a new agent:\n   wandb agent YOUR_SWEEP_ID")
        print(f"\nYour previous best score was {best_score:.6f}. Aim to beat this!")

    def _is_metric_or_metadata_column(self, col: str) -> bool:
        col_lower = col.lower()
        if col_lower == self.primary_metric:
            return True
        if col_lower.startswith(self.KNOWN_METADATA_PREFIXES):
            return True
        return any(keyword in col_lower for keyword in self.KNOWN_METRIC_KEYWORDS)

    def _is_numeric_column(self, col: str) -> bool:
        if self.df is None:
            return False
        numeric_series = pd.to_numeric(self.df[col], errors="coerce")
        notna_sum = self.df[col].notna().sum()
        return (
            False if notna_sum == 0 else numeric_series.notna().sum() / notna_sum > 0.5
        )

    def _infer_distribution(self, param: str, stats: Dict[str, float]) -> str:
        """Infers the best distribution (uniform vs. log_uniform_values)."""
        if param in self.original_params:
            if original_dist := self.original_params[param].get("distribution"):
                if "log" in original_dist:
                    return "log_uniform_values"
                if "q_" in original_dist:
                    return "uniform"
                return original_dist
        if stats["min"] > 1e-9 and (stats["max"] / stats["min"]) > 10:
            return "log_uniform_values"
        if stats["min"] > 1e-9:
            for pattern in self.LOG_SCALE_KEYWORDS:
                if re.match(pattern, param.lower()):
                    return "log_uniform_values"
        return "uniform"

    def _compute_refined_range(
        self, stats: Dict[str, float], dist: str
    ) -> Dict[str, float]:
        p_min, p_max, p_mean = stats["min"], stats["max"], stats["mean"]
        if dist == "log_uniform_values":
            log_min, log_max = np.log(p_min), np.log(p_max)
            log_mean = (log_min + log_max) / 2
            width_factor = 0.5
            new_log_min, new_log_max = (
                log_mean - (log_mean - log_min) * width_factor,
                log_mean + (log_max - log_mean) * width_factor,
            )
            refined_min, refined_max = np.exp(new_log_min), np.exp(new_log_max)
        else:
            p_std = stats.get("std", 0)
            std_factor = 1.5
            refined_min, refined_max = (
                p_mean - std_factor * p_std,
                p_mean + std_factor * p_std,
            )

        bound_factor = 1.5
        refined_min, refined_max = max(refined_min, p_min / bound_factor), min(
            refined_max, p_max * bound_factor
        )
        if refined_min > refined_max:
            refined_min = p_min
        if p_min >= 0:
            refined_min = max(0.0, refined_min)
        return {"min": refined_min, "max": refined_max}

    def _correlation_strength(self, abs_corr: float) -> str:
        if abs_corr >= 0.7:
            return "Strong"
        if abs_corr >= 0.4:
            return "Moderate"
        if abs_corr >= 0.2:
            return "Weak"
        return "Very Weak"


def main():
    parser = argparse.ArgumentParser(
        description="Analyzes W&B Sweep results to generate a refined configuration for both continuous and discrete parameters.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Basic usage (heuristic detection)
  python analyze_sweep_results.py wandb_export.csv

  # Recommended: Specify original config for accurate parameter detection
  python analyze_sweep_results.py wandb_export.csv -c sweep.yaml
  
  # Customize metric and top percentage to analyze
  python analyze_sweep_results.py wandb_export.csv -m val_loss -t 0.1
        """,
    )
    parser.add_argument("csv_file", help="Path to the W&B sweep results CSV file.")
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the original sweep YAML file (optional but recommended).",
    )
    parser.add_argument(
        "-m",
        "--metric",
        default="test/combined_score",
        help="The primary metric to optimize for.",
    )
    parser.add_argument(
        "-t",
        "--top",
        type=float,
        default=0.2,
        help="Percentage of top runs to analyze (0.0 to 1.0).",
    )

    args = parser.parse_args()
    try:
        analyzer = SweepAnalyzer(
            csv_file=args.csv_file,
            original_config=args.config,
            primary_metric=args.metric,
            top_percent=args.top,
        )
        analyzer.run_analysis()
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
