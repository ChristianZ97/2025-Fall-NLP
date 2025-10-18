#!/bin/bash

LOG_FILE="nlp_hw2_$(date +%Y%m%d_%H%M%S).log"

{
    echo "=== Start: $(date) ==="
    python nlp_hw2_colab.py
    python nlp_hw2_colab_rnn.py
    python nlp_hw2_colab_gru.py
    python nlp_hw2_colab_3digit.py
    python nlp_hw2_colab_noise.py
    echo "=== End:   $(date) ==="
} 2>&1 | tee -a "$LOG_FILE"
