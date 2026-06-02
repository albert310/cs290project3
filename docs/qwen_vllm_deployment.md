# Qwen3.6-27B vLLM Deployment

This project uses a local conda environment:

```bash
/2022533109/chenyuhan/miniconda3/envs/cs290-qwen-vllm
```

Download model weights through the Hugging Face mirror:

```bash
cd /2022533109/chenyuhan/cs290project3
bash scripts/download_qwen36_27b.sh
```

Start a resident OpenAI-compatible vLLM server with 7 GPUs and 50% GPU memory utilization:

```bash
cd /2022533109/chenyuhan/cs290project3
bash scripts/start_vllm_qwen36_27b.sh
```

Defaults:

```bash
GPU_IDS=0,1,2,3,4,5,6
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=7
GPU_MEMORY_UTILIZATION=0.50
PORT=8000
SERVED_MODEL_NAME=qwen3.6-27b
```

This model has `hidden_size=5120` and `num_attention_heads=24`, so tensor parallel size 7 is not valid. The 7-GPU default therefore uses pipeline parallelism.

The start script refuses to use busy GPUs by default. Override only when intentional:

```bash
GPU_IDS=0,1,3,4,5,6,7 ALLOW_BUSY_GPUS=1 bash scripts/start_vllm_qwen36_27b.sh
```

Check the server:

```bash
bash scripts/check_vllm_qwen36_27b.sh
```

Stop it:

```bash
bash scripts/stop_vllm_qwen36_27b.sh
```

Model weights, Hugging Face cache, logs, and pid files are intentionally ignored by Git.
