#!/bin/bash

# usage ./run_all_tests.sh | tee execution_log.txt

# Initialize counters
total=0
success=0
failed=0
failed_files=()

# Capture start time
start_time=$(date +%s)

echo "========================================"
echo " Starting Batch Execution of Full Tests"
echo " Date: $(date)"
echo "========================================"

# Loop through all .py files starting with full_test_
for file in $(ls full_test_*.py | sort); do
    # Ensure file exists
    [ -e "$file" ] || continue

    ((total++))
    echo ""
    echo "----------------------------------------"
    echo " Running [$total]: $file"
    echo "----------------------------------------"

    # Execute the Python file
    python "$file"
    
    # Capture exit code (0 is success, non-zero is failure)
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo " [SUCCESS] $file completed successfully."
        ((success++))
    else
        echo " [FAILED] $file FAILED with exit code $exit_code."
        ((failed++))
        failed_files+=("$file")
    fi

    # Clean up cache after each run to prevent dimension mismatch errors in the next script
    echo " Cleaning up embed_cache.db..."
    rm -f embed_cache*
done

# Calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "========================================"
echo " Execution Summary"
echo "========================================"
echo "Total Files Run: $total"
echo "Successful:      $success"
echo "Failed:          $failed"
echo "Total Time:      ${duration} seconds"

if [ ${#failed_files[@]} -ne 0 ]; then
    echo ""
    echo " The following files failed:"
    for f in "${failed_files[@]}"; do
        echo " - $f"
    done
    # Exit with error status if any failed
    exit 1
else
    echo ""
    echo " All tests passed!"
    exit 0
fi
