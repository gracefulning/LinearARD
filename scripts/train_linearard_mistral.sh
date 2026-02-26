#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python train.py \
  --model-id "mistralai/mistral-7b-v0.1" \
  --tokenizer-id "mistralai/mistral-7b-v0.1" \
  --student-rope-scaling '{"rope_type":"linear","factor":4.0}' \
  --num-gpus 2 \
  --ctx-len 4096 \
  --batch-size 1 \
  --grad-accum 2 \
  --steps 144 \
  --base-lr 2e-5 \
  --writer-dir "checkpoints/tflogs" \
  --tune-student-qkv-only true \
  --tune-student-qkv-lora true \
  --lora-rank 1024 \
  --lora-alpha 2048 \
  --lora-dropout 0.05 \
  --use-gradient-checkpointing true \
  --warmup-steps 4 \
  --min-lr-ratio 0.9 \
  --save-every-steps 64 \
  --ddp-init-method "tcp://127.0.0.1:29501" \
  --save-step-ckpt-prefix "checkpoints/attn_align_step_" \
  --save-final-ckpt-dir "checkpoints/attn_align_final" \
  --temperature 1.0 \
  --lam-l 0.0 \
  --lam-a 1.0 \
  --logits-keep-k 512 \
  --max-grad-norm 5.0
