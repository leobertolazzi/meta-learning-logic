import os
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar

from dataset import get_dataset
from utils import (
    set_seed,
    load_model,
    get_model_id,
    evaluate_model_output,
    mask_labels_for_completion
)
from test import get_model_predictions, extract_predictions, process_ground_truth


def setup_accelerator() -> Accelerator:
    """Initialize and return the Accelerator object for distributed training.

    Returns:
        Accelerator: Configured accelerator instance
    """
    accelerator = Accelerator()
    return accelerator


def setup_args(accelerator: Accelerator) -> argparse.Namespace:
    """Set up and parse command line arguments.
    
    Args:
        accelerator: The Accelerator instance for getting device information

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    # Saving dirs
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
    # Model and Dataset
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
        help="Type of fine-tuning to perform (full or lora)."
    )
    parser.add_argument(   
        "--precision",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "int8", "int4"],
        help="Precision for model training (bfloat16, int8, or int4)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="base",
        choices=["base", "meta"],
        help="Dataset to use for training (base or meta-learning)."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="core",
        choices=["core", "short-to-long", "long-to-short"],
        help="Type of experiment to run (core, short-to-long, or long-to-short)."
    )
    parser.add_argument(
        "--unseen_lengths",
        type=int,
        default=3,
        help="Number of unseen lengths to use for non-core experiment."
    )
    parser.add_argument(
        "--subsample_train",
        type=int,
        default=None,
        help="Number of training examples to use for each inf_type x inf_length (if None, use all)."
    )
    parser.add_argument(
        "--simple_corpus",
        action=argparse.BooleanOptionalAction,
        help="Whether to use a simplified version of the corpus with only type 2 inferences."
    )
    # Optimizer args
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for the optimizer."
    )
    # Add scheduler arguments
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for the scheduler."
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Ratio of total training steps to use as warmup."
    )
    # Training args
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--val_per_epoch",
        type=int,
        default=10,
        help="Number of validation steps per epoch."
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length for the model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training."
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=64,
        help="Batch size for validation."
    )
    # Log validation
    parser.add_argument(
        "--log",
        action=argparse.BooleanOptionalAction,
        help="Whether to log validation results."
    )
    args = parser.parse_args()
    
    # Set derived arguments
    args.device = accelerator.device
    args.model_id = get_model_id(args.model)
    args.model_name = f"{args.model}_{args.ft_type}_{args.dataset}_{args.experiment}_seed_{args.seed}"
    if args.simple_corpus:
        args.model_name += "_simple"
    if args.subsample_train:
        args.model_name += f"_{args.subsample_train}"

    return args


def setup_tokenizer(args: argparse.Namespace) -> Any:
    """Initialize and configure the tokenizer for the specified model.
    
    Args:
        args: Parsed command line arguments

    Returns:
        Any: Configured tokenizer for the model
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.seq_len
    return tokenizer


def preprocess_function_lm(
    examples: Dict[str, List[str]],
    return_tensors: Optional[str] = None
) -> Dict[str, List[int]]:
    """Preprocess examples for language modeling.
    
    Args:
        examples: Dictionary containing input and output text pairs

    Returns:
        Dict[str, List[int]]: Tokenized and processed inputs
    """
    strings = [i+o for i,o in zip(examples["input"], examples["output"])]
    return tokenizer(strings, padding=True, truncation=True, max_length=args.seq_len, return_tensors=return_tensors)


def batches(lst: List[Any], n: int) -> List[Any]:
    """Split a list into batches of size n.
    
    Args:
        lst: List to be batched
        n: Batch size

    Yields:
        List[Any]: Batch of elements from the input list
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def log_predictions(
    preds: List[str],
    query_hyp: List[str],
    y_trues: List[str],
    texts: List[str],
) -> None:
    """Log detailed information about predictions for debugging.
    
    Args:
        preds: List of model predictions
        query_hyp: List of query hypotheses
        y_trues: List of ground truth outputs
        texts: List of raw model output texts
    """
    for p, h, y, t in zip(preds, query_hyp, y_trues, texts):
        tqdm.write("-"*50)
        tqdm.write("TEXT")
        tqdm.write(t)
        tqdm.write("HYP")
        tqdm.write(h)
        tqdm.write("PREDICTED:")
        tqdm.write(p)
        tqdm.write("TARGET:")
        tqdm.write(y)


def log_training_details(
    file_path: str,
    epoch: int,
    step: int, 
    val_loss: float,
    val_accuracy: float,
) -> None:
    """Log training details to a file.
        
    Args:
        file_path: Path to the log file
        epoch: Current epoch number
        step: Current iteration number
        val_loss: Validation loss
        val_accuracy: Validation accuracy
    """
    # logging messages
    tqdm.write("epoch = {}\t|\titer = {}\t|\tval_loss = {}\t|\teval/acc = {}".format(
        epoch, step, val_loss, val_accuracy))
    # log to file
    with open(file_path, "a") as f:
        f.write(f"{epoch},{step},{val_loss},{val_accuracy}\n")


def compute_validation_loss(
    model: Any,
    tokenizer: Any,
    batch: Dict[str, Any],
) -> float:
    """Compute validation loss for a batch of data.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        batch: Dictionary containing input and output text pairs

    Returns:
        float: Validation loss for the batch
    """
    # Only decoder logic remains
    model_inputs = preprocess_function_lm(batch, return_tensors="pt")
    labels = model_inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    model_inputs = mask_labels_for_completion(model_inputs, " <QUERY>", tokenizer)
    
    # Convert inputs to tensors on the proper device
    for k, v in model_inputs.items():
        if not isinstance(v, torch.Tensor):
            model_inputs[k] = torch.tensor(v).to(args.device)
        else:
            model_inputs[k] = v.to(args.device)
    
    # Compute loss
    outputs = model(**model_inputs)
    return outputs.loss.item()


@torch.inference_mode()
def validation_loop(
    model: Any,
    tokenizer: Any,
    val_data: Dataset,
) -> (float, float):
    """Run validation loop to evaluate model performance. 
    Returns both validation accuracy and loss.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        val_data: Validation dataset

    Returns:
        float: Validation accuracy
        float: Validation loss
    """
    accuracy = []
    loss_total = 0.0
    count = 0

    with accelerator.autocast():
        
        dev_bar = tqdm(total=len(val_data)//args.val_batch_size, desc="Validation", leave=False)
        for batch in batches(val_data, args.val_batch_size):
            # Calculate max_new_tokens based on the longest output in the batch
            max_new_tokens = max(len(tokenizer.encode(output)) for output in batch["output"])

            # Collect predictions to track accuracy
            texts = get_model_predictions(model, tokenizer, batch, max_new_tokens)
            preds = extract_predictions(texts, batch["query_hyp"])
            y_trues = process_ground_truth(batch["output"])
            accuracy += evaluate_model_output(preds, y_trues)
            
            # Compute validation loss
            loss = compute_validation_loss(model, tokenizer, batch)
            loss_total += loss
            count += 1

            if args.log:
                log_predictions(preds, batch["query_hyp"], y_trues, texts)

            dev_bar.update(1)

    avg_accuracy = round(np.mean(accuracy)*100, 2)
    avg_loss = loss_total / count if count > 0 else 0.0

    return avg_accuracy, avg_loss


def calculate_warmup_steps(total_steps: int, warmup_steps: int, warmup_ratio: float) -> int:
    """Calculate the number of warmup steps based on either explicit steps or ratio.
    
    Args:
        total_steps: Total number of training steps
        warmup_steps: Explicit number of warmup steps (takes precedence if > 0)
        warmup_ratio: Ratio of total steps to use for warmup
        
    Returns:
        int: Number of warmup steps to use
    """
    if warmup_steps > 0:
        return warmup_steps
    
    return int(total_steps * warmup_ratio)


def main() -> None:
    """Main training function that handles the complete training pipeline."""
    # Set seed
    set_seed(args.seed)

    # Save paths
    save_dir = os.path.join(args.save_model_dir, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Train log dir and file
    log_dir = "results/full_logic" if not args.simple_corpus else "results/simple_logic"
    log_dir = os.path.join(log_dir, "train_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, f"{args.model_name}.csv")
    with open(log_file_path, "w") as f:
        f.write("epoch,iter,val_loss,eval/acc\n")

    # Load dataset
    if not accelerator.is_main_process:
        disable_progress_bar()
    print_info = True if accelerator.is_main_process else False
    train, dev, test = get_dataset(
        args.dataset,
        args.experiment, 
        unseen_lengths=args.unseen_lengths,
        subsample_train=args.subsample_train,
        test_type="normal",
        print_info=print_info,
        simple_corpus=args.simple_corpus
    )
    
    # Tokenize the dataset (only decoder logic)
    train_tokenized = train.map(
        preprocess_function_lm,
        batched=True,
        remove_columns=["input", "output", "query_hyp", "query_inf", "inf_length", "inf_type", "kb_id"]
    )
    train_tokenized.set_format("torch")

    # Model
    model = load_model(args.model_id, args.cache_dir, device=args.device, ft_type=args.ft_type, precision=args.precision)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Create dataloader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        train_tokenized, collate_fn=data_collator, batch_size=args.batch_size
    )   

    # Training parameters
    epoch_len = len(train_dataloader)
    total_steps = epoch_len * args.epochs
    save_every_iter = epoch_len // args.val_per_epoch
    
    # Calculate warmup steps
    num_warmup_steps = calculate_warmup_steps(
        epoch_len,
        args.warmup_steps,
        args.warmup_ratio
    )
    
    # Create scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    # Move optimizer, model, scheduler to accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
        
    # Training loop
    model.train()

    best_accuracy = 0
    best_epoch = 0
    best_iter = 0

    if accelerator.is_main_process:
        epoch_bar = tqdm(total=args.epochs, desc="Epoch", leave=False)

    for epoch in range(args.epochs):

        if accelerator.is_main_process:
            step_bar = tqdm(total=epoch_len, desc=f"Training epoch {epoch}", leave=False)

        for step, batch in enumerate(train_dataloader):

            batch = mask_labels_for_completion(batch, " <QUERY>", tokenizer)
            
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()  # Step the scheduler
            optimizer.zero_grad()

            # Validate after n iterations and save best model
            if step % save_every_iter == 0:

                #validation loop
                if accelerator.is_main_process:

                    unwrapped_model = accelerator.unwrap_model(model)

                    val_accuracy, val_loss = validation_loop(unwrapped_model, tokenizer, dev)
                    global_step = step + epoch*epoch_len

                    log_training_details(
                        log_file_path,
                        epoch,
                        global_step,
                        val_loss,
                        val_accuracy
                    )
                    
                    if val_accuracy > best_accuracy:

                        best_accuracy = val_accuracy
                        best_iter = global_step
                        best_epoch = epoch
            
                        save_path = os.path.join(save_dir, f"{args.model_name}_best")
                        unwrapped_model.save_pretrained(
                            save_path,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save
                        )

                accelerator.wait_for_everyone()

            # Re-set model on training
            model.train()

            if accelerator.is_main_process:
                step_bar.update(1)

        if accelerator.is_main_process:
            epoch_bar.update(1)
        
    # Best model log 
    if accelerator.is_main_process:
        tqdm.write("-"*50)
        tqdm.write("BEST MODEL:")
        tqdm.write("epoch = {}\t|\titer = {}".format(best_epoch, best_iter))


if __name__ == "__main__":

    accelerator = setup_accelerator()
    args = setup_args(accelerator)
    tokenizer = setup_tokenizer(args)
    main()

