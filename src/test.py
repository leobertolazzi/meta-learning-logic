import os
import argparse
import logging
from collections import defaultdict
import re
import json
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer

from dataset import get_dataset
from utils import (
  load_best_trained_model,
  get_model_id,
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
        "--cache_dir",
        type=str,
        help="Directory to store cached models."
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="syllogistic-llms",
        help="Directory to save trained models."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1048,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-1.5b",
        help="Name of the model to use."
    )
    parser.add_argument(
        "--ft_type",
        type=str,
        default="full",
        choices=["full", "lora"],
        help="Type of fine-tuning performed (full or lora)."
    )
    parser.add_argument(
        "--test_model_type",
        type=str,
        default="base",
        choices=["base", "meta"],
        help="Dataset type the model was trained on (for loading the correct saved model)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="base",
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
        "--subsample_train",
        type=int,
        default=None,
        help="Number of training examples to use for each inf_type x inf_length (if None, use all)."
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
        default=3,
        help="Number of unseen lengths to use for non-core experiment."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--simple_corpus",
        action=argparse.BooleanOptionalAction,
        help="Whether to use a simplified version of the corpus with only type 2 inferences."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on (cuda/cpu)."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1000,
        help="Maximum number of new tokens to generate."
    )
    
    args = parser.parse_args()
    
    # Add derived attributes to args namespace
    args.model_id = get_model_id(args.model)
    args.model_name = f"{args.model}_{args.ft_type}_{args.test_model_type}_{args.experiment}_seed_{args.seed}"
    if args.simple_corpus:
        args.model_name += "_simple"
    if args.subsample_train:
        args.model_name += f"_{args.subsample_train}"
    
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
            f.write(f"model,seed,type_x_len_samples,test_type,ft_type,model_type,dataset,{labels}\n")
    with open(os.path.join(folder, save_file), "a") as f:
        values = ",".join([str(val) for val in accuracy_dict.values()])
        f.write(f"{args.model},{args.seed},{args.subsample_train},{args.test_type},{args.ft_type},{args.test_model_type},{args.dataset},{values}\n")


def batches(lst: List[Any], n: int) -> List[Any]:
    """Yield successive n-sized batches from lst.
    
    Args:
        lst: Input list to be batched
        n: Batch size
    
    Yields:
        Batches of size n from the input list
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@torch.inference_mode()
def get_model_predictions(
    model: Any,
    tokenizer: Any,
    batch: Dict[str, List],
    max_new_tokens: int,
) -> List[str]:
    """Generate predictions from the model for a batch of inputs.
    
    Args:
        model: The pretrained model to use for inference
        tokenizer: Tokenizer for processing input text
        batch: Dictionary containing input data for the batch
        max_new_tokens: Maximum number of new tokens to generate
    
    Returns:
        List of decoded model outputs as strings
    """
    inputs =tokenizer(
        batch["input"],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)
    
    # Generate model outputs with greedy decoding
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    # Extract and decode model outputs
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_texts


def extract_predictions(
    texts: List[str],
    queries: List[str],
    args: argparse.Namespace,
) -> List[str]:
    """Extract clean predictions from model outputs by removing formatting.
    
    Args:
        texts: Raw model output texts
        queries: Query hypotheses to find in the texts
        args: Runtime arguments containing model and dataset configuration
    
    Returns:
        List of cleaned prediction strings
    """
    preds = []
    for text, query in zip(texts, queries):
        pred = text
        if query in text:
            pred = text.split(query+", premises: ")[-1]
            pred = pred.split(" <STOP>")[0].strip()
            if args.model_type == "base" and args.dataset == "meta":
                pred = pred.split(";")[0].strip()
        else:
            logging.warning(f"Query '{query}' not found in model output: '{text}'")
        preds.append(pred)
    return preds

def process_ground_truth(
    outputs: List[str],
) -> List[str]:
    """Process and clean ground truth outputs for evaluation.
    
    Args:
        outputs: Raw ground truth output strings
    
    Returns:
        List of cleaned ground truth strings
    """
    y_trues = [y.split("premises:")[-1].strip() if "premises:" in y else y for y in outputs]
    y_trues = [y.split(" <STOP>")[0].strip() for y in y_trues]
    return y_trues


def clean_prediction(pred: str) -> str:
    """Clean and validate logical statements in a prediction string.
    
    Args:
        pred: Input string containing comma-separated logical statements
        
    Returns:
        Cleaned string with validated, unique statements joined by commas
        
    Example:
        >>> clean_prediction("All A are B, Some B are C, All A are B")
        "All A are B, Some B are C"
    """
    # Split by comma and strip whitespace
    statements = [s.strip() for s in pred.split(',')]
    # Remove duplicates
    statements = list(set(statements))
    # Regex pattern for valid statements
    pattern = r'^(all|some|no) [A-Za-z0-9]* (are|are not) [A-Za-z0-9]*$'
    valid_statements = []
    for stmt in statements:
        if not re.match(pattern, stmt):
            logging.warning(f"Invalid statement format: {stmt}")
        else:
            valid_statements.append(stmt)
    # Join valid statements back into comma-separated string
    return ", ".join(valid_statements)


def calculate_type_length_accuracies(
    accuracies: List[bool],
    batch: Dict[str, List],
    test_ranges: Dict,
    accuracy_dict: Dict[str, List]
) -> Dict[str, List]:
    """Calculate accuracy metrics broken down by inference type and length.
    
    Args:
        accuracies: List of boolean accuracy values for each prediction
        batch: Dictionary containing batch data including inference types and lengths
        test_ranges: Dictionary defining min/max lengths for each inference type
        accuracy_dict: Dictionary to accumulate accuracy metrics
    
    Returns:
        Updated accuracy dictionary with new metrics added
    """
    accuracy_dict["accuracy"] += accuracies
    for typ in test_ranges:
        for length in range(test_ranges[typ]["min"], test_ranges[typ]["max"]+1):
            accuracy_dict[f"type_{typ}_len_{length}"] += [
                correct for correct, l, t in zip(accuracies, batch["inf_length"], batch["inf_type"])
                if l == length and t == typ
            ]
    return accuracy_dict


def initialize_error_log(args: argparse.Namespace) -> None:
    """Initialize empty JSON file for logging errors at script startup.
    
    Args:
        args: Runtime arguments containing model and API configuration
    """
    save_folder = "results/full_logic" if not args.simple_corpus else "results/simple_logic"
    log_dir = os.path.join(save_folder, "errors")
    json_file = os.path.join(log_dir, f"{args.model_name}_{args.dataset}_{args.test_type}.json")
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create empty results structure
    results = {}
    
    # Save initial empty file
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)


def log_predictions_to_json(
    preds: List[str],
    batch: Dict[str, List],
    y_trues: List[str],
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
    json_file = os.path.join(log_dir, f"{args.model_name}_{args.dataset}_{args.test_type}.json")
    
    os.makedirs(log_dir, exist_ok=True)

    # Load existing data if file exists
    results = {}
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            results = json.load(f)

    # Build nested dictionary structure
    for pred, hyp, inf_len, inf_type, y_true, kb_id in zip(
        preds,
        batch["query_hyp"],
        batch["inf_length"],
        batch["inf_type"],
        y_trues,
        batch.get("kb_id", [""] * len(preds))
    ):

        # Clean and validate predictions
        cleaned_pred = clean_prediction(pred)
        cleaned_y_true = clean_prediction(y_true)
        
        # Convert inference type and length to strings for JSON keys
        inf_type_str = str(inf_type)
        inf_len_str = str(inf_len)
        
        # Initialize nested structures if they don't exist
        if inf_type_str not in results:
            results[inf_type_str] = {}
        if inf_len_str not in results[inf_type_str]:
            results[inf_type_str][inf_len_str] = {}
            
        # Create new entry
        hyp = hyp.replace("hypothesis: ", "")
        results[inf_type_str][inf_len_str][hyp] = {
            "prediction": cleaned_pred,
            "answer": cleaned_y_true,
            "kb_id": kb_id
        }
    
    # Save updated results
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)


def test_loop(
    model: Any,
    tokenizer: Any,
    test_data: Dataset,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Run evaluation loop on test data.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for processing text
        test_data: Dataset containing test examples
        args: Runtime arguments and configuration
    
    Returns:
        Dictionary containing accuracy metrics (core and by type/length)
    """
    accuracy_dict = defaultdict(list)
    test_ranges = {
        t: test_data.to_pandas()[test_data.to_pandas()['inf_type'] == t]['inf_length'].agg(['min', 'max']).to_dict()
        for t in set(test_data['inf_type'])
    }

    for batch in tqdm(batches(test_data, args.batch_size), desc="Test", leave=False, total=len(test_data)//args.batch_size):
        # Get model predictions
        texts = get_model_predictions(model, tokenizer, batch, args.max_new_tokens)
        preds = extract_predictions(texts, batch["query_hyp"], args)
        
        # Process ground truth and calculate accuracies
        y_trues = process_ground_truth(batch["output"])
        accuracies = evaluate_model_output(preds, y_trues)
        accuracy_dict = calculate_type_length_accuracies(accuracies, batch, test_ranges, accuracy_dict)
        
        # Log wrong predictions
        wrong_indices = [i for i, acc in enumerate(accuracies) if not acc]
        if wrong_indices:
            wrong_preds = [preds[i] for i in wrong_indices]
            wrong_batch = {k: [v[i] for i in wrong_indices] for k, v in batch.items()}
            wrong_y_trues = [y_trues[i] for i in wrong_indices]
            log_predictions_to_json(wrong_preds, wrong_batch, wrong_y_trues, args)

    # Calculate final accuracies
    for key, value in accuracy_dict.items():
        accuracy_dict[key] = round(np.mean(value)*100, 2)
    
    return accuracy_dict


def main(args: argparse.Namespace) -> None:
    """Main execution function.
    
    Args:
        args: Runtime arguments containing model and training configuration
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
        subsample_train=args.subsample_train,
        test_type=args.test_type,
        simple_corpus=args.simple_corpus,
        print_info=False
    )
        
    # Load tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = load_best_trained_model(model_id=args.model_id, model_name=args.model_name, cache_dir=args.cache_dir, save_model_dir=args.save_model_dir, device=args.device, ft_type=args.ft_type)
    # Test step
    accuracy_dict = test_loop(model, tokenizer, test, args)
    # Save results
    acc = accuracy_dict["accuracy"]
    tqdm.write(f"Accuracy = {acc}")
    save_folder = "results/full_logic" if not args.simple_corpus else "results/simple_logic"
    save_results(save_folder, f"results_{args.experiment}.csv", accuracy_dict, args)


if __name__ == "__main__":
    args = setup_args()
    main(args)
