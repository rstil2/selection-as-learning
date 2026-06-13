#!/bin/bash
# Launch ViT experiment with the conda Python that has PyTorch/MPS.
cd "$(dirname "$0")"
PYTHON="/Users/stillwell/miniconda3/bin/python3"
nohup "$PYTHON" -u ml_vit_experiment.py > ml_vit_log.txt 2>&1 &
echo $! > ml_vit_pid.txt
echo "Started ViT experiment PID $(cat ml_vit_pid.txt)"
