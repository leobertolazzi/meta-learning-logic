#!/bin/bash

source activate syllogistic_llms

start_time=$(date +%s)

# Global configuration variables
CACHE_DIR=""
SAVE_MODEL_DIR=""
MODEL= # "qwen-1.5b", "qwen-3b", "qwen-7b"
FT_TYPE= # "lora" for LoRA fine-tuning, "full" for full fine-tuning
UNSEEN_LENGTHS=5
SUBSAMPLE_TRAIN= # 1000 or 100
SEED= # 1048, 1596, 512

# Training parameters
PRECISION= # "bfloat16" for "base", "int4" for "meta" 
LR=5e-5
WEIGHT_DECAY=0.0
WARMUP_STEPS=0
WARMUP_RATIO=0.0
EPOCHS= # 1 for "meta", 4 for "base"
VAL_PER_EPOCH=10
SEQ_LEN=2048
TRAIN_BATCH_SIZE=4
VAL_BATCH_SIZE=16

# Test parameters
TEST_BATCH_SIZE=32
MAX_NEW_TOKENS=500
TEST_TYPE="normal" # ood_constants, ood_words, ood_support
DEVICE="cuda"

# Experiment parameters
experiments=(core long-to-short short-to-long)
dataset= # "meta" for MIND meta-learning, "base" for baseline
test_model_type= # "meta" or "base" - type of model to load at test time

# Run training and testing scripts for each type of experiment
for exp in "${experiments[@]}"; do 

    echo "Running experiment: $exp, dataset: $dataset"

    accelerate launch --config_file config_single.yaml src/train.py \
        --cache_dir "${CACHE_DIR}" \
        --save_model_dir "${SAVE_MODEL_DIR}" \
        --model "${MODEL}" \
        --precision "${PRECISION}" \
        --ft_type "${FT_TYPE}" \
        --dataset "$dataset" \
        --experiment "$exp" \
        --unseen_lengths ${UNSEEN_LENGTHS} \
        --subsample_train ${SUBSAMPLE_TRAIN} \
        --lr ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_steps ${WARMUP_STEPS} \
        --warmup_ratio ${WARMUP_RATIO} \
        --epochs ${EPOCHS} \
        --val_per_epoch ${VAL_PER_EPOCH} \
        --seq_len ${SEQ_LEN} \
        --batch_size ${TRAIN_BATCH_SIZE} \
        --val_batch_size ${VAL_BATCH_SIZE} \
        --seed ${SEED} \

    python src/test.py \
        --cache_dir "${CACHE_DIR}" \
        --save_model_dir "${SAVE_MODEL_DIR}" \
        --model "${MODEL}" \
        --ft_type "${FT_TYPE}" \
        --test_model_type "$test_model_type" \
        --dataset "$dataset" \
        --experiment "$exp" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --unseen_lengths ${UNSEEN_LENGTHS} \
        --subsample_train ${SUBSAMPLE_TRAIN} \
        --test_type ${TEST_TYPE} \
        --batch_size ${TEST_BATCH_SIZE} \
        --device ${DEVICE} \
        --seed ${SEED} \

done

end_time=$(date +%s)

execution_time_seconds=$((end_time - start_time))
execution_time_hours=$(printf "%.2f" "$(bc -l <<< "scale=2; $execution_time_seconds / 3600")")

echo "Execution time: $execution_time_hours hours"