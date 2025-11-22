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
CUDA_VISIBLE_DEVICES=4,5,6,7 \
vllm serve "openai/gpt-oss-20b" \
  --gpu-memory-utilization 0.5 \
  --tensor-parallel-size 4 \
  --enforce-eager


CUDA_VISIBLE_DEVICES=4,5,6,7 \
vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 16384 \
  --max-num-seqs 12 \
```

## Run RAG Pipeline
```bash
python rag_pipeline.py
```

## Run Full Test
```bash
CUDA_VISIBLE_DEVICES=1 \
python full_test.py
```

## Submit Result
```bash
kaggle competitions submit -c WattBot2025 -f submission.csv -m "Message"
```
