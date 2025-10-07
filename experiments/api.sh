#!/bin/bash

source activate syllogistic_llms

start_time=$(date +%s)

# Configuration variables
EXPERIMENTS= # "core", "long-to-short", "short-to-long"
DATASET= # "base" for baseline, "meta" for MIND meta-learning
AZURE_ENDPOINT= # Set your Azure endpoint here, e.g., "https://your-endpoint.openai.azure.com/"
API_VERSION="2025-03-01-preview"
MODEL= # "o3-mini" "gpt-4o"
DEPLOYMENT= # Set your deployment name here, e.g., "gpt-4o-deployment"
TEST_TYPE="normal" # "ood_constants", "ood_words", "ood_support"
UNSEEN_LENGTHS=5
MAX_RETRIES=5
RETRY_DELAY=3

python src/api.py \
    --azure_endpoint "${AZURE_ENDPOINT}" \
    --api_version "${API_VERSION}" \
    --model "${MODEL}" \
    --deployment "${DEPLOYMENT}" \
    --dataset "${DATASET}" \
    --experiment "${EXPERIMENTS}" \
    --test_type "${TEST_TYPE}" \
    --unseen_lengths "${UNSEEN_LENGTHS}" \
    --max_retries "${MAX_RETRIES}" \
    --retry_delay "${RETRY_DELAY}"

end_time=$(date +%s)

execution_time_seconds=$((end_time - start_time))
execution_time_hours=$(printf "%.2f" "$(bc -l <<< "scale=2; $execution_time_seconds / 3600")")

echo "Execution time: $execution_time_hours hours"
