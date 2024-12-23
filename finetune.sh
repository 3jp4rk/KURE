# # scripts/finetune.sh

# EPOCH=2
# LR=1e-5
# BATCH_SIZE=32
# DATE=

# export WANDB_PROJECT="KoE5"
# export WANDB_NAME="KoE5-large-v1.2-InfoNCE-bs=${BATCH_SIZE}-ep=${EPOCH}-lr=${LR}-${DATE}"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py \
#     --model_name_or_path intfloat/multilingual-e5-large \
#     --output_dir /data/KoE5/MODELS/${WANDB_NAME} \
#     --data_dir /data/KoE5/DATA/datasets/ \
#     --cache_dir projects/KoE5/cache \
#     --num_train_epochs $EPOCH \
#     --learning_rate $LR \
#     --per_device_train_batch_size $BATCH_SIZE \
#     --per_device_eval_batch_size $BATCH_SIZE \
#     --warmup_steps 100 \
#     --logging_steps 2 \
#     --save_steps 100 \
#     --cl_temperature 0.02 \
#     --test False

# scripts/finetune.sh

EPOCH=5
LR=1e-5
BATCH_SIZE=8
DATE=

# export WANDB_PROJECT="KoE5"
# export WANDB_NAME="KoE5-large-v1.2-InfoNCE-bs=${BATCH_SIZE}-ep=${EPOCH}-lr=${LR}-${DATE}-12000"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py \
#     --model_name_or_path intfloat/multilingual-e5-large \
#     --output_dir /data/ejpark/KoE5/MODELS/${WANDB_NAME} \
#     --num_train_epochs $EPOCH \
#     --learning_rate $LR \
#     --per_device_train_batch_size $BATCH_SIZE \
#     --per_device_eval_batch_size $BATCH_SIZE \
#     --warmup_steps 100 \
#     --logging_steps 1 \
#     --save_steps 100 \
#     --cl_temperature 0.02 \
#     --test False \
#     --remove_unused_columns false


export WANDB_PROJECT="tunib-electra-kotriplet"
export WANDB_NAME="tunib-electra-base-v1.2-InfoNCE-bs=${BATCH_SIZE}-ep=${EPOCH}-lr=${LR}-${DATE}-12000"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py \
    --model_name_or_path tunib/electra-ko-en-base \
    --output_dir /data/ejpark/KoE5/MODELS/${WANDB_NAME} \
    --num_train_epochs $EPOCH \
    --learning_rate $LR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --warmup_steps 100 \
    --logging_steps 1 \
    --save_steps 100 \
    --cl_temperature 0.02 \
    --test False \
    --remove_unused_columns false
