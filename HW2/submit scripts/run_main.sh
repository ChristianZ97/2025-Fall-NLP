#!/bin/bash

export LC_ALL=C

LOG_FILE="nlp_hw2_$(date +%Y%m%d_%H%M%S).log"

{
    echo "=== Start: $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "Running experiments with conda environment: nlp"
    echo "Log file: $LOG_FILE"
    echo ""
    
    echo "[1/1] Running Main Experiment..."
    python nlp_hw2_colab_adamw.py
    echo ""
    
    echo "=== End: $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "All experiments completed. Check $LOG_FILE for details."
} 2>&1 | tee -a "$LOG_FILE"
