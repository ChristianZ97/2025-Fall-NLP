# WattBot2025 Quick Start Guide

## Download the Kaggle dataset

```bash
mkdir -p data
kaggle competitions download -c WattBot2025 -p data
unzip WattBot2025
```

## Try Dummy Submit
```bash
python dummy_test.py
kaggle competitions submit -c WattBot2025 -f submission.csv -m "Message"
```

## Download all Metadata
```bash
python data_preprocess.py
```

## Run vLLM Server
```bash
vllm serve "meta-llama/Llama-3.2-1B" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.7 \
```

## Run RAG Pipeline
```bash
python rag_pipeline.py
```

## Submit Result
```bash
kaggle competitions submit -c WattBot2025 -f submission.csv -m "Message"
```
