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
export CUDA_VISIBLE_DEVICES=4,5,6,7
echo $CUDA_VISIBLE_DEVICES
vllm serve --host 0.0.0.0 --port 8090 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.7 \
  --trust-remote-code \
  openai/gpt-oss-20b
```

## Run RAG Pipeline
```bash
python rag_pipeline.py
```

## Submit Result
```bash
kaggle competitions submit -c WattBot2025 -f submission.csv -m "Message"
```
