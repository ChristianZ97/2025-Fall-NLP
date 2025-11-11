#!/bin/bash

# 設定參數
RUNS=10
OUTPUT_DIR="experiment_results"
SUMMARY_CSV="${OUTPUT_DIR}/summary_results.csv"

# 創建輸出目錄
mkdir -p "${OUTPUT_DIR}"

# Python 檔案列表
SCRIPTS=(
    "nlp_hw3_arch.py"
)

# 初始化 CSV header
echo "script,run,epoch1_pearson,epoch1_accuracy,epoch1_combine,epoch2_pearson,epoch2_accuracy,epoch2_combine,epoch3_pearson,epoch3_accuracy,epoch3_combine,test_pearson,test_accuracy,test_combine" > "${SUMMARY_CSV}"

# 對每個 script 運行
for script in "${SCRIPTS[@]}"; do
    echo "Running ${script}..."
    
    for run in $(seq 1 ${RUNS}); do
        echo "  Run ${run}/${RUNS}"
        
        # 運行 script 並保存輸出
        log_file="${OUTPUT_DIR}/${script%.py}_run${run}.log"
        python "${script}" > "${log_file}" 2>&1
        
        # 提取數據
        # 修正：正規表示式新增對負號 "-" 的支援
        epoch1=$(grep "Epoch 1:" "${log_file}" | grep -oP 'Pearson=\K[-0-9.]+|Accuracy=\K[-0-9.]+|Combine=\K[-0-9.]+' | tr '\n' ',' | sed 's/,$//')
        epoch2=$(grep "Epoch 2:" "${log_file}" | grep -oP 'Pearson=\K[-0-9.]+|Accuracy=\K[-0-9.]+|Combine=\K[-0-9.]+' | tr '\n' ',' | sed 's/,$//')
        epoch3=$(grep "Epoch 3:" "${log_file}" | grep -oP 'Pearson=\K[-0-9.]+|Accuracy=\K[-0-9.]+|Combine=\K[-0-9.]+' | tr '\n' ',' | sed 's/,$//')
        test=$(grep "^Test:" "${log_file}" | grep -oP 'Pearson=\K[-0-9.]+|Accuracy=\K[-0-9.]+|Combine=\K[-0-9.]+' | tr '\n' ',' | sed 's/,$//')

        
        
        # 寫入 CSV
        echo "${script},${run},${epoch1},${epoch2},${epoch3},${test}" >> "${SUMMARY_CSV}"
    done
done

echo "All runs completed. Results saved to ${SUMMARY_CSV}"

# 計算平均值的 Python script
python3 << 'EOF'
import pandas as pd
import numpy as np

df = pd.read_csv('experiment_results/summary_results.csv')

# 按 script 分組計算平均值和標準差
summary = df.groupby('script').agg(['mean', 'std']).round(4)

# 保存彙總統計
summary.to_csv('experiment_results/average_results.csv')

print("\n=== Average Results ===")
print(summary)

# 只顯示測試集的平均結果
test_cols = ['test_pearson', 'test_accuracy', 'test_combine']
test_summary = df.groupby('script')[test_cols].agg(['mean', 'std']).round(4)
print("\n=== Test Set Summary ===")
print(test_summary)
EOF

echo "Summary statistics saved to experiment_results/average_results.csv"
