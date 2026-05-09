# Maximum Activations in Open Large Language Models

This repository contains the experimental code and results for **“Measuring Maximum Activations in Open Large Language Models”**. The project measures the maximum activation magnitude in modern open LLMs, defined as `M = max |a|`, and analyzes how it varies across model families, architectures, training stages, and INT-8 quantization behavior.

## Key Findings

- Across 8 open model families and 24 main-analysis checkpoints, maximum activation magnitudes span multiple orders of magnitude.
- `M=max|a|` is a more deployment-relevant risk indicator than a binary massive-activation criterion.
- Global maximum activations mainly appear in the residual stream; the paper reports that 22 of 24 main checkpoints peak in hidden states.
- MoE routing, vision-language adaptation, instruction tuning, training stage, and model family all affect peak magnitude; parameter count alone does not explain the behavior.
- Bootstrap subset experiments show that the observed peaks are not artifacts of a few outlier samples.
- INT-8 sanity checks show that extreme activations can substantially degrade reconstruction quality.

## Repository Structure

```text
./
├── README.md
├── requirements.txt
├── analyze_model.py
├── convert_dataset.py
├── make_bootstrap_datasets.py
├── quant_sanity_check.py
├── scan_top5.py
├── run.sh
├── run_supplementary_experiments.sh
└── results/
    ├── */json/*_activation_stats.json
    ├── */json/*_top5_stats.json
    ├── quant/
    ├── bootstrap/
    └── supplementary_experiments/
```

## Setup

A Linux machine with NVIDIA GPUs is recommended. The code assumes local/offline Hugging Face models and tokenizers by default.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ./requirements.txt

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Before rerunning experiments, prepare:

```text
./models/
./datasets/
```

## Data Preparation

The shared evaluation dataset used in this experiment has been uploaded to Hugging Face:

```text
https://huggingface.co/datasets/Clxxxx/eval_diverse_dataset
```

Convert the shared evaluation set to the tokenizer encoding of each model family:

```bash
python ./convert_dataset.py --series Qwen3
python ./convert_dataset.py --all
```

Supported series: `Qwen2.5`, `Qwen3`, `gemma2`, `gpt_oss`, `ling`, `Qwen3.5`, `gemma3`, `Qwen2.5-vl`.

## Activation Analysis

Run a single model:

```bash
python ./analyze_model.py \
  --model_path ./models/Qwen3/Qwen3-8B \
  --data_path ./datasets/eval_diverse_5k_qwen3.jsonl \
  --output_dir ./results/Qwen3 \
  --max_samples 5000 \
  --max_seq_len 32768 \
  --batch_size 16 \
  --gpu_id 0
```

Outputs:

```text
./results/<series>/json/<model>_activation_stats.json
./results/<series>/json/<model>_top5_stats.json
```

Run the main experiment pipeline:

```bash
bash ./run.sh --convert
bash ./run.sh --analyze
bash ./run.sh stop
```

## Scripts

- `analyze_model.py`: Core activation-statistics script. It uses PyTorch hooks to collect activations from embeddings, hidden states, attention outputs, MLP/MoE outputs, gates, and final normalization layers.
- `scan_top5.py`: Scans generated Top-K activation results and reports peak concentration, sign consistency, and hot-layer information.
- `make_bootstrap_datasets.py`: Builds category-proportional 1k/2k bootstrap subsets.
- `quant_sanity_check.py`: Runs INT-8 activation quantization sanity checks.
- `run.sh`: Main entry point for dataset conversion and activation analysis.
- `run_supplementary_experiments.sh`: Unified supplementary-experiment runner for bootstrap stability and the extended quantization sanity check.

## Supplementary Experiments

Run all supplementary experiments:

```bash
bash ./run_supplementary_experiments.sh
```

Run only the quantization sanity check:

```bash
bash ./run_supplementary_experiments.sh --quant
```

Run only bootstrap stability:

```bash
bash ./run_supplementary_experiments.sh --bootstrap
```

Supplementary results are stored in:

```text
./results/quant/
./results/bootstrap/
./results/supplementary_experiments/
```

## Existing Results

The current `results/` directory contains JSON outputs for Qwen2.5, Qwen2.5-Instruct, Qwen2.5-VL, Qwen3, Qwen3.5, Gemma2, Gemma3, Ling, GPT-OSS, and the supplementary experiments.

Main experiment outputs usually include:

```text

<model>_activation_stats.json

<model>_top5_stats.json
```

## Notes

- Model weights and datasets should be placed under `./models/` and `./datasets/` before running experiments.
- The scripts use `nvidia-smi` to schedule jobs on available GPUs and are intended for multi-GPU NVIDIA servers.
- Large models require substantial GPU memory; the code selects single-GPU loading or `device_map="auto"` based on model size.
- The paper is currently anonymous; author information and a license file should be added before public release.
