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
CUDA_VISIBLE_DEVICES=6,7
echo $CUDA_VISIBLE_DEVICES
vllm serve "openai/gpt-oss-20b" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 8192 \
  --enforce-eager
```

## Run RAG Pipeline
```bash
python rag_pipeline.py
```

## Run Full Test
```bash
CUDA_VISIBLE_DEVICES=1
echo $CUDA_VISIBLE_DEVICES
python full_test.py
```

## Submit Result
```bash
kaggle competitions submit -c WattBot2025 -f submission.csv -m "Message"
```
