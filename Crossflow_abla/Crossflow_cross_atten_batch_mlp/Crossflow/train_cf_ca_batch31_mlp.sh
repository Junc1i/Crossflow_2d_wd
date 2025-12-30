#!/bin/bash
set -x

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SEED_MODELS_LOGGING_LEVEL=WARN
export TOKENIZERS_PARALLELISM=false
export VESCALE_SINGLE_DEVICE_RAND=0
export TF_CPP_MIN_LOG_LEVEL=2

NNODES=${NNODES:=$ARNOLD_WORKER_NUM}
NPROC_PER_NODE=${NPROC_PER_NODE:=$ARNOLD_WORKER_GPU}
NPROC_PER_NODE=${NPROC_PER_NODE:=$ARNOLD_WORKER_GPU_PER_NODE}
NODE_RANK=${NODE_RANK:=$ARNOLD_ID}
MASTER_ADDR=${MASTER_ADDR:=$ARNOLD_WORKER_0_HOST}
MASTER_ADDR=${MASTER_ADDR:=$ARNOLD_EXECUTOR_0_HOST}
MASTER_PORT=${MASTER_PORT:=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)}
MASTER_PORT=${MASTER_PORT:=$(echo "$ARNOLD_EXECUTOR_0_PORT" | cut -d "," -f 1)}
NPROC=$((NNODES * NPROC_PER_NODE))

# add wandb api key
WANDB_API_KEY=""

# 登录 wandb
echo "Logging in to wandb..."
export WANDB_API_KEY=$WANDB_API_KEY
wandb login --relogin $WANDB_API_KEY
if [ $? -ne 0 ]; then
    echo "Error: Failed to login to wandb. Exiting..."
    exit 1
fi
echo "Successfully logged in to wandb"

echo $NODE_RANK
echo $NPROC
echo $MASTER_ADDR
echo $MASTER_PORT

cd /mnt/bn/pistis/weixian/workplace/flowone/Crossflow_2d_wd/Crossflow_abla/Crossflow_cross_atten_batch_mlp/Crossflow

accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --multi_gpu \
    --num_processes $NPROC \
    --num_machines $NNODES \
    --mixed_precision bf16 \
    Crossflow/train_t2i.py \
    --config=Crossflow/configs/t2i_training_demo.py \
    --workdir_base="/mnt/bn/pistis/weixian/exp/flow-one-pt-combined-crossatten-batch3-1-mlp-1b" \
    --vae_pretrained_path="/mnt/bn/pistis/weixian/ckpt/CrossFlow/assets/stable-diffusion/autoencoder_kl.pth" \
    --model_pretrained_path="/mnt/bn/pistis/weixian/ckpt/CrossFlow/pretrained_models/t2i_256px_clip_dimr.pth" \
    --fid_stat_path="/mnt/bn/pistis/weixian/ckpt/CrossFlow/assets/fid_stats/fid_stats_mscoco256_val.npz" \
    --inception_ckpt_path="/mnt/bn/pistis/weixian/ckpt/pt_inception-2015-12-05-6726825d.pth" \
    --sample_path="/mnt/bn/pistis/weixian/exp/flow-one-pt-combined-crossatten-batch3-1-mlp-1b/save_test_samples" \
    --train_tar_pattern="/mnt/bn/zilongdata-us/weixian/data/visual_instruction_dataset_wds/pairs-{000000..001497}.tar,/mnt/hdfs/pistis/weixian/data/flowone/JourneyDB_wds/pairs-{000000..007356}.tar" \
    --test_tar_pattern="/mnt/bn/zilongdata-us/weixian/data/visual_instruction_dataset_wds/pairs-{001498..001523}.tar,/mnt/hdfs/pistis/weixian/data/flowone/JourneyDB_wds/pairs-{007357..007382}.tar" \
    --vis_image_root="/mnt/bn/pistis/weixian/data/flowone/test_vis" \
    --n_steps=1000000 \
    --batch_size=256 \
    --log_interval=10 \
    --eval_interval=1000 \
    --save_interval=100000 \
    --n_samples_eval=11 \
    --dataset_name=online_features \
    --task=visual_instruction \
    --resolution=256 \
    --shuffle_buffer=500 \
    --resampled=True \
    --split_data_by_node=True \
    --estimated_samples_per_shard=600 \
    --sampling_weights=0.75,0.25 \
    --sample_steps=50 \
    --n_samples=30000 \
    --mini_batch_size=16 \
    --scale=7 \
    --optimizer_name=adamw \
    --lr=0.00001 \
    --weight_decay=0.03 \
    --betas=0.9,0.9 \
    --adamw_impl=AdamW \
    --use_cross_attention=true \
    --wandb_project=crossflow_2d_mixed_ca_batch3-1-mlp-1b-new-wds \
    --wandb_mode=online \
    --num_workers=16