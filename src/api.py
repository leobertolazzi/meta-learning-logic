import os
import argparse
import logging
import json
from collections import defaultdict
from typing import List, Dict
from datetime import datetime
import time

import numpy as np
from tqdm import tqdm
from datasets import Dataset
from openai import AzureOpenAI

from dataset import get_dataset
from utils import (
    evaluate_model_output,
    set_seed,
)


def setup_args() -> argparse.Namespace:
    """Setup and return command line arguments.

    Returns:
        Namespace object containing all runtime arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--azure_endpoint",
        type=str,
        required=True,
        help="Azure OpenAI API endpoint."
    )
    parser.add_argument(
        "--api_version",
        type=str,
        default="2025-03-01-preview",
        help="Azure OpenAI API version."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="o3-mini",
        help="Name of the OpenAI or Deepseek model to use."
    )
    parser.add_argument(
        "--deployment",
        type=str,
        default="o3-mini-b",
        help="Name of the AzureOpenAI deployment model to use."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1048,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="meta",
        choices=["base", "meta"],
        help="Dataset to use for evaluation (base or meta-learning)."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="core",
        choices=["core", "short-to-long", "long-to-short"],
        help="Type of experiment to run (core, short-to-long, or long-to-short)."
    )
    parser.add_argument(
        "--test_type",
        type=str,
        default="normal",
        choices=["normal", "ood_words", "ood_support", "ood_constants"],
        help="Type of test evaluation to perform."
    )
    parser.add_argument(
        "--unseen_lengths",
        type=int,
        default=5,
        help="Number of unseen lengths to use for non-core experiment."
    )
    parser.add_argument(
        "--simple_corpus",
        action=argparse.BooleanOptionalAction,
        help="Whether to use a simplified version of the corpus with only type 2 inferences."
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum number of retries for API calls"
    )
    parser.add_argument(
        "--retry_delay",
        type=int,
        default=3,
        help="Delay between retries in seconds"
    )

    args = parser.parse_args()

    # Add derived attributes to args namespace
    args.ft_type = "api"
    args.model_name = f"{args.model}_{args.ft_type}_{args.dataset}_{args.experiment}"
    if args.simple_corpus:
        args.model_name += "_simple"

    # System prompt
    if args.dataset == "meta":
        args.system = (
            "You are tasked with logical premise selection. Given:\n"
            "1. A knowledge base consisting of premises.\n"
            "2. Example hypotheses along with their correct minimal premise sets, preceded by the token <STUDY>.\n"
            "3. A query hypothesis to solve, preceded by the token <QUERY>.\n\n"
            "Your task is to identify the unique minimal set of premises from the knowledge base that logically proves the query hypothesis. "
            "Since the knowledge base is non-redundant, every valid hypothesis has exactly one minimal set of premises that proves it.\n\n"
            "Examine the provided examples carefully to understand how to select the correct minimal set of premises. "
            "The examples demonstrate correct premise selections for various hypotheses.\n\n"
            "Provide your answer in exactly this format:\n"
            "### Answer: premise1, premise2, ..., premiseN"
        )
    elif args.dataset == "base":
        args.system = (
            "You are tasked with logical premise selection. Given:\n"
            "1. A knowledge base consisting of premises.\n"
            "2. A query hypothesis to solve, preceded by the token <QUERY>.\n\n"
            "Your task is to identify the unique minimal set of premises from the knowledge base that logically proves the query hypothesis. "
            "Since the knowledge base is non-redundant, every valid hypothesis has exactly one minimal set of premises that proves it.\n\n"
            "Provide your answer in exactly this format:\n"
            "### Answer: premise1, premise2, ..., premiseN"
        )

    return args


def save_results(
    folder: str,
    save_file: str,
    accuracy_dict: Dict[str, float],
    args: argparse.Namespace
) -> None:
    """Save evaluation results to a CSV file.

    Args:
        folder: Directory to save results
        save_file: Name of the CSV file
        accuracy_dict: Dictionary containing accuracy metrics
        args: Runtime arguments
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(os.path.join(folder, save_file)):
        with open(os.path.join(folder, save_file), "w") as f:
            labels = ",".join([str(key) for key in accuracy_dict.keys()])
            f.write(f"model,seed,type_x_len_samples,test_type,ft_type,setting,{labels}\n")
    with open(os.path.join(folder, save_file), "a") as f:
        values = ",".join([str(val) for val in accuracy_dict.values()])
        f.write(f"{args.model},-,-,{args.test_type},{args.ft_type},{args.dataset},{values}\n")


def extract_predictions(
    text: str,
) -> str:
    """Extract clean prediction from model outputs by removing formatting.

    Args:
        text: Raw model output text

    Returns:
        Cleaned prediction string
    """
    if "### answer: " in text.lower():
        pred = text.lower().split("### answer: ")[-1].strip()
    else:
        logging.warning(f"Anwer format not found in model output: '{text}'")
        pred = text.lower()

    return pred


def process_ground_truth(
    y_true: str
) -> str:
    """Process and clean ground truth outputs for evaluation.

    Args:
        outputs: Raw ground truth output string

    Returns:
        Cleaned ground truth strings
    """
    y_true = y_true.split("premises:")[-1].strip() if "premises:" in y_true else y_true
    y_true = y_true.replace(" <STOP>", "")
    return y_true


def calculate_type_length_accuracies(
    correct: bool,
    inpt: Dict[str, str],
    accuracy_dict: Dict[str, List]
) -> Dict[str, List]:
    """Calculate accuracy metrics broken down by inference type and length.

    Args:
        correct: Boolean accuracy values for prediction
        inpt: Dictionary containing inpt data including inference types and lengths
        accuracy_dict: Dictionary to accumulate accuracy metrics

    Returns:
        Updated accuracy dictionary with new metrics added
    """
    accuracy_dict["accuracy"].append(correct)
    typ = inpt["inf_type"]
    length = inpt["inf_length"]
    accuracy_dict[f"type_{typ}_len_{length}"].append(correct)
    return accuracy_dict


def initialize_error_log(args: argparse.Namespace) -> None:
    """Initialize empty JSON file for logging errors at script startup.
    
    Args:
        args: Runtime arguments containing model and API configuration
    """
    save_folder = "results/full_logic" if not args.simple_corpus else "results/simple_logic"
    log_dir = os.path.join(save_folder, "errors")
    json_file = os.path.join(log_dir, f"{args.model_name}_{args.test_type}.json")
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create empty results structure
    results = {}
    
    # Save initial empty file
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)


def log_predictions_to_json(
    pred: str,
    inpt: Dict[str, str], 
    y_true: str,
    args: argparse.Namespace
) -> None:
    """Log predictions to a JSON file with nested structure by type and length.
    
    The JSON structure will look like:
    {
        "1": { # Inference type
            "2": { # Inference length
                "All A are C": { # Hypothesis
                        "prediction": "All A are B, All B are C", 
                        "answer": "All A are B, All B are C",
                        "kb_id": "te_L_ds_3"
                    },
                ...
            },
        }
    }
    """
    save_folder = "results/full_logic" if not args.simple_corpus else "results/simple_logic" 
    log_dir = os.path.join(save_folder, "errors")
    json_file = os.path.join(log_dir, f"{args.model_name}_{args.test_type}.json")

    # Load current results
    with open(json_file, 'r') as f:
        results = json.load(f)

    # Convert inference type and length to strings for JSON keys
    inf_type_str = str(inpt['inf_type'])
    inf_len_str = str(inpt['inf_length'])
    
    # Initialize nested structures if they don't exist
    if inf_type_str not in results:
        results[inf_type_str] = {}
    if inf_len_str not in results[inf_type_str]:
        results[inf_type_str][inf_len_str] = {}
        
    # Create new entry
    hyp = inpt['query_hyp'].replace("hypothesis: ", "")
    results[inf_type_str][inf_len_str][hyp] = {
        "prediction": pred,
        "answer": y_true,
        "kb_id": inpt['kb_id']
    }
    
    # Save updated results
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)


def prepare_batch_data(test_data: Dataset, args: argparse.Namespace) -> str:
    """Convert dataset to JSONL format required for batch processing.
    
    Args:
        test_data: Dataset containing test examples to process
        args: Runtime arguments containing model configuration
        
    Returns:
        str: Path to the created batch file
    """
    batch_file = f"data/batch_{args.test_type}.jsonl"
    
    with open(batch_file, 'w') as f:
        for idx, item in enumerate(test_data):
            batch_item = {
                "custom_id": f"task-{idx}",
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": args.deployment,
                    "messages": [
                        {"role": "system", "content": args.system},
                        {"role": "user", "content": item["input"]}
                    ]
                }
            }
            f.write(json.dumps(batch_item) + "\n")
    
    return batch_file


def monitor_batch_status(client: AzureOpenAI, batch_id: str) -> str:
    """Monitor the status of a batch job until completion.
    
    Args:
        client: Azure OpenAI client instance
        batch_id: ID of the batch job to monitor
        
    Returns:
        str: Final status of the batch job
        
    Raises:
        Exception: If batch processing fails
    """
    status = "none"
    last_status = status
    
    while status not in ("completed", "failed", "canceled"):
        time.sleep(30)
        batch_response = client.batches.retrieve(batch_id)
        status = batch_response.status
        
        if status != last_status:
            print(f"{datetime.now()} Batch Id: {batch_id}, Status: {status}")
            last_status = status
    
    if status == "failed":
        for error in batch_response.errors.data:
            logging.error(f"Error code {error.code} Message {error.message}")
        raise Exception("Batch processing failed")
        
    return status


def process_batch_results(
    results_file: str,
    test_data: Dataset,
    args: argparse.Namespace
) -> Dict[str, float]:
    """Process batch results from LLM responses and compute accuracy metrics.
    This function processes a batch of results from LLM responses, extracts predictions,
    and computes various accuracy metrics based on inference types and lengths.
    
    Args:
        results_file (str): Path to the file containing raw LLM responses
        test_data (Dataset): Dataset containing test examples and ground truth
        args (argparse.Namespace): Command line arguments
    
    Returns:
        Dict[str, float]: Dictionary containing accuracy metrics by type/length,
            where keys are metric names and values are accuracy percentages
    """
    with open(results_file, 'r') as f:
        raw_responses = f.read().strip().split('\n')
    
    # Extract predictions
    predictions = ["None"] * len(test_data)  
    for raw_response in raw_responses:
        json_response = json.loads(raw_response)
        id = int(json_response['custom_id'].split('-')[1])
        try:
            text_response = json_response['response']['body']['choices'][0]['message']['content']
        except KeyError:
            logging.warning(f"No answer found in response for id {id}")
            text_response = "None"
        pred = extract_predictions(text_response)
        predictions[id] = pred 
    
    # Initialize accuracy tracking
    accuracy_dict = defaultdict(list)
    
    # Calculate metrics
    for idx, (pred, item) in enumerate(zip(predictions, test_data)):
        y_true = process_ground_truth(item["output"])
        accuracy = evaluate_model_output([pred], [y_true])[0]
        accuracy_dict = calculate_type_length_accuracies(accuracy, item, accuracy_dict)
        if not accuracy:
            log_predictions_to_json(pred, item, y_true, args)
    
    # Calculate final accuracies
    for key, value in accuracy_dict.items():
        accuracy_dict[key] = round(np.mean(value)*100, 2)
    
    return accuracy_dict


def batch_inference(client: AzureOpenAI, test_data: Dataset, args: argparse.Namespace) -> Dict[str, float]:
    """Execute complete batch inference workflow.
    
    Args:
        client: Azure OpenAI client instance
        test_data: Dataset containing test examples
        args: Runtime arguments containing configuration
        
    Returns:
        Dict[str, float]: Dictionary containing accuracy metrics
        
    Raises:
        Exception: If batch processing fails or output is missing
    """
    # Prepare batch data
    batch_file = prepare_batch_data(test_data, args)

    # Upload file
    file = client.files.create(
        file=open(batch_file, "rb"),
        purpose="batch",
        extra_body={"expires_after": {"seconds": 1209600, "anchor": "created_at"}}
    )

    # Create batch job
    batch_response = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    # Monitor progress
    status = monitor_batch_status(client, batch_response.id)
    batch_response = client.batches.retrieve(batch_response.id)
    
    if status == "completed":
        # Retrieve and save results
        output_file_id = batch_response.output_file_id
        if output_file_id:
            file_response = client.files.content(output_file_id)
            output_dir = "results/full_logic/api"
            os.makedirs(output_dir, exist_ok=True)
            results_file = os.path.join(output_dir, f"outputs_{args.model_name}_{args.test_type}.jsonl")
            with open(results_file, 'w') as f:
                f.write(file_response.text)
            
            return process_batch_results(results_file, test_data, args)
    
    raise Exception("No output file found")


def main(args: argparse.Namespace) -> None:
    """Main execution function.

    Args:
        args: Runtime arguments containing model and API configuration
    """
    # Set seed
    set_seed(args.seed)

    # Initialize error logging
    initialize_error_log(args)

    # Load dataset
    train, dev, test = get_dataset(
        args.dataset,
        args.experiment,
        unseen_lengths=args.unseen_lengths,
        subsample_train=None,
        test_type=args.test_type,
        simple_corpus=args.simple_corpus,
        print_info=False
    )
    
    # Initialize client
    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    # Run batch inference
    accuracy_dict = batch_inference(client, test, args)

    # Save results
    acc = accuracy_dict["accuracy"]
    tqdm.write(f"Accuracy = {acc}")
    save_folder = "results/full_logic" if not args.simple_corpus else "results/simple_logic"
    save_results(save_folder, f"results_{args.experiment}.csv", accuracy_dict, args)


if __name__ == "__main__":
    args = setup_args()
    main(args)