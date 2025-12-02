#!/bin/bash

# usage ./run_all_tests.sh | tee execution_log.txt

# 初始化計數器
total=0
success=0
failed=0
failed_files=()

# 抓取開始時間
start_time=$(date +%s)

echo "========================================"
echo "🚀 Starting Batch Execution of Full Tests"
echo "   Date: $(date)"
echo "========================================"

# 迴圈遍歷所有開頭為 full_test_ 的 .py 檔案，並進行排序
for file in $(ls full_test_*.py | sort); do
    # 確保檔案存在
    [ -e "$file" ] || continue

    ((total++))
    echo ""
    echo "----------------------------------------"
    echo "📄 Running [$total]: $file"
    echo "----------------------------------------"

    # 執行 Python 檔案
    # 注意：這裡使用 'python'，它會使用您當前啟用的環境 (Py base)
    python "$file"
    
    # 抓取執行結果代碼 (0 代表成功，非 0 代表失敗)
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✅ $file completed successfully."
        ((success++))
    else
        echo "❌ $file FAILED with exit code $exit_code."
        ((failed++))
        failed_files+=("$file")
    fi
done

# 計算總耗時
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "========================================"
echo "📊 Execution Summary"
echo "========================================"
echo "Total Files Run: $total"
echo "Successful:      $success"
echo "Failed:          $failed"
echo "Total Time:      ${duration} seconds"

if [ ${#failed_files[@]} -ne 0 ]; then
    echo ""
    echo "⚠️  The following files failed:"
    for f in "${failed_files[@]}"; do
        echo " - $f"
    done
    # 如果有失敗，腳本以非 0 狀態退出
    exit 1
else
    echo ""
    echo "🎉 All tests passed!"
    exit 0
fi
