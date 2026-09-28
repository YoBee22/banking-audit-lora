# Banking Audit Extraction via LoRA Fine-Tuning

Automated compliance extraction pipeline fine-tuning **Llama-3.2-3B-Instruct** using low-rank adaptation (LoRA) on Apple Silicon via Apple MLX.

## Overview
This project fine-tunes a local 4-bit Llama-3.2-3B model to convert unstructured, noisy banking compliance notes into strictly compliant, typed JSON records (flag types, severity, resolution status, and USD amounts).

## Architecture & Workflow
1. **Synthetic Data Generation:** Generated 270+ domain-specific audit records using `gemini-3.5-flash-lite` with structured schema enforcement.
2. **LoRA Fine-Tuning:** Trained low-rank adapters (`mlx_lm.lora`) on 4-bit quantized Llama-3.2-3B weights.
3. **Weight Fusion:** Merged LoRA adapters into base weights (`mlx_lm.fuse`) for zero-latency local inference.
4. **Evaluation:** Evaluated strict JSON parsing accuracy and schema adherence across validation samples.

## Evaluation Benchmark

```text
==========================================
Model: ./fused_model
Validation Samples: 20
Valid JSON Extraction Accuracy: 95.0%
==========================================
