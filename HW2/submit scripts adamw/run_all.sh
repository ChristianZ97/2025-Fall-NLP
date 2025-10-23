#!/bin/bash

export LC_ALL=C

LOG_FILE="nlp_hw2_$(date +%Y%m%d_%H%M%S).log"

{
    echo "=== Start: $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "Running experiments with conda environment: nlp"
    echo "Log file: $LOG_FILE"
    echo ""
    
    echo "[1/5] Running baseline LSTM..."
    python nlp_hw2_colab.py
    echo ""
    
    echo "[2/5] Running RNN variant..."
    python nlp_hw2_colab_rnn.py
    echo ""
    
    echo "[3/5] Running GRU variant..."
    python nlp_hw2_colab_gru.py
    echo ""
    
    echo "[4/5] Running 3-digit experiment..."
    python nlp_hw2_colab_3digit.py
    echo ""
    
    echo "[5/5] Running noise experiment..."
    python nlp_hw2_colab_noise.py
    echo ""
    
    echo "=== End: $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "All experiments completed. Check $LOG_FILE for details."
} 2>&1 | tee -a "$LOG_FILE"
