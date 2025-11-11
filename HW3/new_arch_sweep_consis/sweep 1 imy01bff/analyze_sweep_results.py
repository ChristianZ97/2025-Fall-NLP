# analyze_sweep_results.py

import pandas as pd
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SweepAnalyzer:
    """自動分析 W&B Sweep 結果並生成細化配置"""

    # 已知的 metric 和 metadata 列（不是超參數）
    METRIC_KEYWORDS = [
        "score",
        "accuracy",
        "loss",
        "pearson",
        "spearman",
        "f1",
        "precision",
        "recall",
        "auc",
        "metric",
    ]

    METADATA_COLS = [
        "epoch",
        "step",
        "runtime",
        "timestamp",
        "duration",
        "name",
        "state",
        "created",
        "tags",
        "notes",
        "id",
        "username",
        "version",
        "host",
        "sweep_id",
        "run_id",
        "batch_perplexity",
        "raw_grad_norm",
    ]

    # 通常使用對數尺度的參數關鍵字
    LOG_SCALE_KEYWORDS = [
        "lr",
        "learning_rate",
        "alpha",
        "beta",
        "eps",
        "epsilon",
        "weight_decay",
        "decay",
        "gamma",
        "momentum",
    ]

    def __init__(
        self,
        csv_file: str,
        original_config: Optional[str] = None,
        primary_metric: str = "test_combined_score",
        top_percent: float = 0.2,
    ):
        """
        初始化分析器

        Args:
            csv_file: W&B sweep 結果 CSV 檔案
            original_config: 原始 sweep 配置檔案（可選，用於保持一致性）
            primary_metric: 主要優化指標
            top_percent: 用於分析的 top 配置百分比 (default: 0.2 = 20%)
        """
        self.csv_file = csv_file
        self.original_config = original_config
        self.primary_metric = primary_metric
        self.top_percent = top_percent

        self.df = None
        self.search_params = []
        self.original_params = {}

    def load_data(self) -> None:
        """載入並預處理數據"""
        print(f"\n{'='*70}")
        print("載入數據")
        print("=" * 70)

        self.df = pd.read_csv(self.csv_file)
        print(f"載入 {len(self.df)} 筆記錄")
        print(f"可用欄位: {len(self.df.columns)} 個")

        # 載入原始配置（如果提供）
        if self.original_config and Path(self.original_config).exists():
            with open(self.original_config, "r") as f:
                config = yaml.safe_load(f)
                self.original_params = config.get("parameters", {})
            print(f"✓ 載入原始配置: {self.original_config}")
            print(f"  原始參數: {list(self.original_params.keys())}")

    def detect_parameters(self) -> List[str]:
        """自動檢測超參數列"""
        print(f"\n{'='*70}")
        print("自動檢測超參數")
        print("=" * 70)

        potential_params = []

        for col in self.df.columns:
            # 跳過已知的 metric 和 metadata 列
            if self._is_metadata_column(col):
                continue

            # 檢查是否為數值型
            if not self._is_numeric_column(col):
                continue

            # 檢查是否有變化（至少 2 個不同值）
            if self.df[col].nunique() <= 1:
                continue

            potential_params.append(col)

        self.search_params = potential_params

        print(f"✓ 檢測到 {len(self.search_params)} 個超參數:")
        for i, param in enumerate(self.search_params, 1):
            nunique = self.df[param].nunique()
            print(f"  {i:2d}. {param:.<40} ({nunique} 個不同值)")

        return self.search_params

    def _is_metadata_column(self, col: str) -> bool:
        """判斷是否為 metadata 列"""
        col_lower = col.lower()

        # 完全匹配 metadata 列名
        if col_lower in [m.lower() for m in self.METADATA_COLS]:
            return True

        # 包含 metric 關鍵字
        if any(keyword in col_lower for keyword in self.METRIC_KEYWORDS):
            return True

        # 以下劃線開頭（W&B 內部欄位）
        if col.startswith("_"):
            return True

        return False

    def _is_numeric_column(self, col: str) -> bool:
        """判斷是否為數值型列"""
        try:
            # 嘗試轉換為數值
            numeric_series = pd.to_numeric(self.df[col], errors="coerce")
            # 如果超過 50% 的值可以轉換，認為是數值型
            valid_ratio = numeric_series.notna().sum() / len(self.df)
            return valid_ratio > 0.5
        except:
            return False

    def preprocess_data(self) -> None:
        """數據預處理"""
        print(f"\n{'='*70}")
        print("數據預處理")
        print("=" * 70)

        # 檢查主要 metric 是否存在
        if self.primary_metric not in self.df.columns:
            raise ValueError(f"指標 '{self.primary_metric}' 不存在於數據中!")

        original_len = len(self.df)

        # 轉換超參數為數值型
        for param in self.search_params:
            self.df[param] = pd.to_numeric(self.df[param], errors="coerce")

        # 移除關鍵列有缺失值的行
        dropna_cols = [self.primary_metric] + self.search_params
        self.df = self.df.dropna(subset=dropna_cols)

        dropped = original_len - len(self.df)
        if dropped > 0:
            print(f"⚠ 移除 {dropped} 筆包含缺失值的記錄")

        # 移除異常值（超過 3 個標準差）
        metric_mean = self.df[self.primary_metric].mean()
        metric_std = self.df[self.primary_metric].std()
        threshold = 3

        before_outlier = len(self.df)
        self.df = self.df[
            (self.df[self.primary_metric] >= metric_mean - threshold * metric_std)
            & (self.df[self.primary_metric] <= metric_mean + threshold * metric_std)
        ]
        removed_outliers = before_outlier - len(self.df)

        if removed_outliers > 0:
            print(f"⚠ 移除 {removed_outliers} 筆異常值")

        print(f"✓ 最終保留 {len(self.df)} 筆有效記錄")

        # 排序
        self.df = self.df.sort_values(self.primary_metric, ascending=False)

    def analyze_performance(self) -> pd.DataFrame:
        """分析整體表現"""
        print(f"\n{'='*70}")
        print("整體表現分析")
        print("=" * 70)

        metric_stats = {
            "最佳值": self.df[self.primary_metric].max(),
            "平均值": self.df[self.primary_metric].mean(),
            "中位數": self.df[self.primary_metric].median(),
            "標準差": self.df[self.primary_metric].std(),
            "最差值": self.df[self.primary_metric].min(),
            "範圍": self.df[self.primary_metric].max()
            - self.df[self.primary_metric].min(),
        }

        print(f"\n{self.primary_metric} 統計:")
        for key, value in metric_stats.items():
            print(f"  {key:.<20} {value:.6f}")

        # Top 配置
        top_n = max(5, int(len(self.df) * self.top_percent))
        top_df = self.df.head(top_n)

        print(f"\n--- Top {top_n} 配置 ({self.top_percent*100:.0f}%) ---")
        display_cols = self.search_params + [self.primary_metric]
        display_cols = [col for col in display_cols if col in self.df.columns]

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", 25)
        print(top_df[display_cols].head(10).to_string(index=False))

        return top_df

    def analyze_parameter(self, param: str, top_df: pd.DataFrame) -> Dict:
        """分析單個參數並生成配置"""
        print(f"\n{'='*70}")
        print(f"{param.upper()} 分析")
        print("=" * 70)

        param_values = top_df[param]

        # 基本統計
        stats = {
            "min": param_values.min(),
            "max": param_values.max(),
            "mean": param_values.mean(),
            "median": param_values.median(),
            "std": param_values.std(),
        }

        print(f"Top 配置範圍: [{stats['min']:.8f}, {stats['max']:.8f}]")
        print(f"平均值: {stats['mean']:.8f}")
        print(f"中位數: {stats['median']:.8f}")
        print(f"標準差: {stats['std']:.8f}")

        # 判斷分布類型
        distribution = self._infer_distribution(param, stats)
        print(f"推薦分布: {distribution}")

        # 計算細化範圍
        refined_range = self._compute_refined_range(param, stats, distribution)
        print(f"細化範圍: [{refined_range['min']:.8f}, {refined_range['max']:.8f}]")

        # 與原始配置比較
        if param in self.original_params:
            original = self.original_params[param]
            print(f"\n原始配置:")
            print(
                f"  範圍: [{original.get('min', 'N/A')}, {original.get('max', 'N/A')}]"
            )
            print(f"  分布: {original.get('distribution', 'N/A')}")

        return {
            "distribution": distribution,
            "min": float(refined_range["min"]),
            "max": float(refined_range["max"]),
        }

    def _infer_distribution(self, param: str, stats: Dict) -> str:
        """推斷參數的分布類型"""
        param_lower = param.lower()

        # 1. 檢查是否在原始配置中已定義
        if param in self.original_params:
            original_dist = self.original_params[param].get("distribution")
            if original_dist:
                return original_dist

        # 2. 如果值都為正且跨越多個數量級，使用對數尺度
        if stats["min"] > 0:
            ratio = stats["max"] / stats["min"]
            if ratio > 10:  # 跨越一個數量級
                return "log_uniform_values"

        # 3. 根據參數名稱判斷
        for keyword in self.LOG_SCALE_KEYWORDS:
            if keyword in param_lower:
                if stats["min"] > 0:
                    return "log_uniform_values"
                break

        # 4. 默認使用 uniform
        return "uniform"

    def _compute_refined_range(
        self, param: str, stats: Dict, distribution: str
    ) -> Dict[str, float]:
        """計算細化範圍"""
        mean = stats["mean"]
        std = stats["std"]
        param_min = stats["min"]
        param_max = stats["max"]

        if distribution == "log_uniform_values":
            # 對數空間中計算
            if mean > 0 and std > 0:
                log_mean = np.log(mean)
                log_std = np.log(std + 1e-10)

                # ±0.5 in log space (約 ×/÷ 1.65)
                refined_min = np.exp(log_mean - 0.5)
                refined_max = np.exp(log_mean + 0.5)

                # 確保在合理範圍內
                refined_min = max(param_min * 0.5, refined_min)
                refined_max = min(param_max * 2.0, refined_max)
            else:
                refined_min = param_min
                refined_max = param_max
        else:
            # 線性空間：mean ± 1.5*std
            refined_min = mean - 1.5 * std
            refined_max = mean + 1.5 * std

            # 確保在原始範圍內或稍微擴展
            refined_min = max(param_min * 0.5, refined_min)
            refined_max = min(param_max * 2.0, refined_max)

            # 確保非負（如果原始值都非負）
            if param_min >= 0:
                refined_min = max(0.0, refined_min)

        return {"min": refined_min, "max": refined_max}

    def correlation_analysis(self) -> None:
        """相關性分析"""
        print(f"\n{'='*70}")
        print("相關性分析")
        print("=" * 70)

        numeric_cols = self.search_params + [self.primary_metric]
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]

        if len(numeric_cols) <= 1:
            print("⚠ 沒有足夠的數值列進行相關性分析")
            return

        corr = (
            self.df[numeric_cols]
            .corr()[self.primary_metric]
            .sort_values(ascending=False)
        )

        print(f"\n與 {self.primary_metric} 的相關係數:")
        for col, corr_val in corr.items():
            if col != self.primary_metric and col in self.search_params:
                strength = self._correlation_strength(abs(corr_val))
                print(f"  {col:.<45} {corr_val:>8.4f}  ({strength})")

    def _correlation_strength(self, abs_corr: float) -> str:
        """判斷相關性強度"""
        if abs_corr >= 0.7:
            return "強相關"
        elif abs_corr >= 0.4:
            return "中等相關"
        elif abs_corr >= 0.2:
            return "弱相關"
        else:
            return "極弱相關"

    def generate_config(self, param_configs: Dict) -> Dict:
        """生成細化的 sweep 配置"""
        print(f"\n{'='*70}")
        print("生成細化 Sweep 配置")
        print("=" * 70)

        # 讀取原始配置的其他設定
        program = "train.py"
        method = "bayes"
        metric_goal = "maximize"

        if self.original_config and Path(self.original_config).exists():
            with open(self.original_config, "r") as f:
                original = yaml.safe_load(f)
                program = original.get("program", program)
                method = original.get("method", method)
                if "metric" in original:
                    metric_goal = original["metric"].get("goal", metric_goal)

        config = {
            "program": program,
            "method": method,
            "metric": {"name": self.primary_metric, "goal": metric_goal},
            "parameters": param_configs,
        }

        # 保存配置
        output_file = "sweep_config_refined.yaml"
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(
                config, f, default_flow_style=False, sort_keys=False, allow_unicode=True
            )

        print(f"✓ 細化配置已保存至: {output_file}")

        print("\n新配置參數範圍:")
        for param, cfg in param_configs.items():
            print(f"  {param:.<35} [{cfg['min']:.8f}, {cfg['max']:.8f}]")
            print(f"  {'':.<35} ({cfg['distribution']})")

        return config

    def parameter_importance(self) -> None:
        """參數重要性排序"""
        print(f"\n{'='*70}")
        print("參數重要性排序 (基於相關性)")
        print("=" * 70)

        numeric_cols = self.search_params + [self.primary_metric]
        corr = (
            self.df[numeric_cols]
            .corr()[self.primary_metric]
            .drop(self.primary_metric)
            .abs()
            .sort_values(ascending=False)
        )

        print("\n按絕對相關係數排序:")
        for i, (param, corr_val) in enumerate(corr.items(), 1):
            if param in self.search_params:
                bar = "█" * int(corr_val * 30)
                print(f"  {i:2d}. {param:.<40} {corr_val:.4f} {bar}")

    def print_best_config(self) -> Dict:
        """輸出最佳配置為 Python dictionary 格式"""
        print(f"\n{'='*70}")
        print("最佳配置")
        print("=" * 70)

        # 取得最佳配置（score 最高的那一行）
        best_row = self.df.iloc[0]

        # 提取超參數值
        best_config = {}
        for param in self.search_params:
            value = best_row[param]
            best_config[param] = float(value)

        # 輸出為 Python dict 格式
        print("\ndefault_config = {")
        for i, (key, value) in enumerate(best_config.items()):
            # 格式化數值
            if abs(value) < 0.01 or abs(value) > 1000:
                # 科學記號表示的小數或大數用完整精度
                value_str = f"{value:.15f}".rstrip("0").rstrip(".")
            else:
                # 一般數值保留適當位數
                value_str = f"{value:.15f}".rstrip("0").rstrip(".")

            # 最後一項不加逗號
            comma = "," if i < len(best_config) - 1 else ""
            print(f'    "{key}": {value_str}{comma}')
        print("}")

        # 顯示對應的 metric 分數
        best_score = best_row[self.primary_metric]
        print(f"\n對應 {self.primary_metric}: {best_score:.6f}")

        return best_config

    def print_recommendations(self) -> None:
        """打印建議的下一步"""
        best_score = self.df[self.primary_metric].max()
        target_score = best_score * 1.01

        print(f"\n{'='*70}")
        print("建議的下一步")
        print("=" * 70)
        print("1. 檢查生成的配置: sweep_config_refined.yaml")
        print("2. 初始化新的 sweep:")
        print("   wandb sweep sweep_config_refined.yaml")
        print("3. 啟動 sweep agent:")
        print("   wandb agent YOUR_USERNAME/PROJECT/SWEEP_ID")
        print("4. 執行 30-50 次實驗（Bayes 方法）")
        print("5. 當改進 < 0.1% 時停止")
        print(f"\n當前最佳 {self.primary_metric}: {best_score:.6f}")
        print(f"目標超越: {target_score:.6f} (+1%)")

    def run_analysis(self) -> Dict:
        """執行完整分析流程"""
        # 1. 載入數據
        self.load_data()

        # 2. 檢測參數
        self.detect_parameters()

        if not self.search_params:
            raise ValueError("未檢測到任何超參數!")

        # 3. 預處理數據
        self.preprocess_data()

        # 4. 整體表現分析
        top_df = self.analyze_performance()

        # 5. 分析每個參數
        param_configs = {}
        for param in self.search_params:
            config = self.analyze_parameter(param, top_df)
            param_configs[param] = config

        # 6. 相關性分析
        self.correlation_analysis()

        # 7. 參數重要性
        self.parameter_importance()

        # 8. 生成配置
        final_config = self.generate_config(param_configs)

        best_config = self.print_best_config()

        # 9. 打印建議
        self.print_recommendations()

        return final_config


def main():
    parser = argparse.ArgumentParser(
        description="自動分析 W&B Sweep 結果並生成細化配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本用法
  python analyze_sweep_results.py sweep_results.csv
  
  # 指定原始配置文件
  python analyze_sweep_results.py sweep_results.csv -c sweep_config.yaml
  
  # 自定義 metric 和 top 百分比
  python analyze_sweep_results.py sweep_results.csv -m val_accuracy -t 0.15
        """,
    )

    parser.add_argument("csv_file", help="W&B sweep 結果 CSV 檔案")
    parser.add_argument("-c", "--config", help="原始 sweep 配置檔案（可選）")
    parser.add_argument(
        "-m",
        "--metric",
        default="test_combined_score",
        help="主要優化指標 (default: test_combined_score)",
    )
    parser.add_argument(
        "-t", "--top", type=float, default=0.2, help="分析 top 百分比 (default: 0.2)"
    )

    args = parser.parse_args()

    # 檢查文件是否存在
    if not Path(args.csv_file).exists():
        print(f"錯誤: 找不到檔案 '{args.csv_file}'")
        return

    # 創建分析器
    analyzer = SweepAnalyzer(
        csv_file=args.csv_file,
        original_config=args.config,
        primary_metric=args.metric,
        top_percent=args.top,
    )

    # 執行分析
    try:
        analyzer.run_analysis()
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
