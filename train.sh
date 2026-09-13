#!/bin/bash

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT=13245

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=$SLURM_NNODES --nproc-per-node=4 --rdzv_id $SLURM_JOB_ID --rdzv_backend c10d --rdzv_endpoint "${MASTER_ADDR}":"${MASTER_PORT}" \
    -m lcm.train launcher=standalone \
    +pretrain=two_tower \
    ++trainer.output_dir="checkpoints/two_tower_lcm_FINEWEB_MULTILINGUAL" \
    +trainer.use_submitit=false 