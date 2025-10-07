import logging
from typing import Tuple, Dict, Optional

import pandas as pd
from datasets import Dataset, load_dataset 


def get_dataset(
    dataset_type: str,
    experiment: str,
    unseen_lengths: int = 3,
    subsample_train: Optional[int] = None,
    test_type: str = "normal",
    print_info: bool = True,
    simple_corpus: bool = False,
    ) -> Tuple[Dataset, Dataset, Dataset]:
    """Get the dataset for the baseline and meta-learning experiments.

    Args:
        dataset_type (str): Type of dataset ('base' or 'meta').
        experiment (str): Type of experiment ('core', 'long-to-short', or 'short-to-long').
        unseen_lengths (int): Number of lengths to keep unseen during training. Defaults to 3.
        subsample_train (Optional[int]): Number of instances to sample for each train inf_type x inf_length.
            If "None" the full data is used. Defaults to None.
        test_type (str): Type of test split ('normal', 'ood_words', 'ood_support', 'ood_constants'). Defaults to "normal".
        print_info (bool): If True, print dataset statistics. Defaults to True.
        simple_corpus (bool): If True, filter dataset to only include inf_type=2. Defaults to False.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Tuple containing train, dev, and test datasets.
    """

    # Load and filter data
    data_files = {
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
        "test_ood_constants": "test_ood_constants.csv",
        "test_ood_support": "test_ood_support.csv",
        "test_ood_words": "test_ood_words.csv",
    }
    full_data = load_dataset("leobertolazzi/syllogistic-logic", data_files=data_files)

    train_df = full_data["train"].to_pandas()
    dev_df = full_data["validation"].to_pandas()
    if test_type != "normal":
        test_df = full_data[f"test_{test_type}"].to_pandas()
    else:
        test_df = full_data["test"].to_pandas()

    if simple_corpus:
        train_df = train_df[train_df["inf_type"] == 2]
        dev_df = dev_df[dev_df["inf_type"] == 2]
        test_df = test_df[test_df["inf_type"] == 2]

    df_ranges = {
        split : {t: df[df['inf_type'] == t]['inf_length'].agg(['min', 'max']).to_dict() for t in set(df['inf_type'])} 
        for split, df in zip(["train", "dev", "test"], [train_df, dev_df, test_df])
    }

    # Set study examples based on experiment
    experiment_to_study_examples = {
        "core": "study_examples_all",
        "long-to-short": "study_examples_comp",
        "short-to-long": "study_examples_rec",
    }
    study_examples_to_filter = experiment_to_study_examples[experiment]

    # Split data
    train_df, dev_df, test_df = process_length(
        train_df, dev_df, test_df, df_ranges,
        experiment, unseen_lengths, subsample_train
    )

    # Format input/output
    for split_df in [train_df, dev_df, test_df]:
        format_input_output(split_df, dataset_type, study_examples_to_filter)

    if print_info:
        print_dataset_info(train_df, dev_df, test_df)

    # Create datasets
    columns = ['input', 'output', 'query_hyp', 'query_inf', 'inf_length', 'inf_type', 'kb_id']
    train = Dataset.from_pandas(train_df[columns], preserve_index=False).shuffle()
    dev = Dataset.from_pandas(dev_df[columns], preserve_index=False)
    test = Dataset.from_pandas(test_df[columns], preserve_index=False)

    return train, dev, test


def print_dataset_info(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print general dataset information and split statistics.

    Args:
        train_df: Training split DataFrame.
        dev_df: Development split DataFrame.
        test_df: Test split DataFrame.
    """
    print("-"*50)
    print("Dataset splits:")
    for split_name, split_df in [("Train", train_df), ("Dev", dev_df), ("Test", test_df)]:
        print(f"{split_name} size: {len(split_df)}")
        print(f"\n{split_name} split distribution:")
        print(f"Number of KBs: {split_df['kb_id'].nunique()}")
        print(f"Number of premises: min={split_df['#prem_kb'].min()}, max={split_df['#prem_kb'].max()}")
        matrix = pd.crosstab(split_df['inf_type'], split_df['inf_length'])
        print(matrix.to_string())
        print()
    print("-"*50)


def process_length(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame, 
    df_ranges: Dict[str, Dict[int, Dict[str, int]]],
    experiment: str,
    unseen_lengths: int,
    subsample_train: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare train, dev, and test splits based on experiment type.

    Args:
        train_df: DataFrame containing training data.
        dev_df: DataFrame containing validation data.
        test_df: DataFrame containing test data.
        df_ranges: Dictionary of min/max ranges for each inference type and split.
        experiment: Type of experiment ('core', 'long-to-short', or 'short-to-long').
        unseen_lengths: Number of lengths to keep unseen during training.
        subsample_train: Number of instances to sample for each inf_type x inf_length.
            If "None" the full data is used. Defaults to None.

    Returns:
        Tuple containing train, dev, and test DataFrames.
    """
    if experiment == "core":
        # Subsample validation split
        dev_df = constant_sample(dev_df, k=5)
        if subsample_train:
            train_df = constant_sample(train_df, k=subsample_train)
        return train_df, dev_df, test_df
    
    # Apply length filtering
    highest = experiment == "short-to-long"
    train_df = train_df.groupby('inf_type').apply(lambda x: filter_extreme_n_lengths(x, df_ranges["train"], unseen_lengths, highest=highest, complement=False)).reset_index(drop=True)
    dev_df = dev_df.groupby('inf_type').apply(lambda x: filter_extreme_n_lengths(x, df_ranges["dev"], unseen_lengths, highest=highest, complement=False)).reset_index(drop=True)
    test_df = test_df.groupby('inf_type').apply(lambda x: filter_extreme_n_lengths(x, df_ranges["test"], unseen_lengths, highest=highest, complement=True)).reset_index(drop=True)
    
    # Subsample
    if subsample_train:
        train_df = constant_sample(train_df, k=subsample_train)

    dev_df = constant_sample(dev_df, k=5)

    return train_df, dev_df, test_df


def filter_extreme_n_lengths(
    group: pd.DataFrame,
    df_ranges: Dict[int, Dict[str, int]],
    n: int,
    highest: bool,
    complement: bool = False
    ) -> pd.DataFrame:
    """Filter DataFrame based on extreme n lengths.

    Args:
        group: DataFrame group to filter.
        df_ranges: Dictionary of min/max ranges for each inference type.
        n: Number of lengths to filter.
        highest: If True, filter highest n lengths, else filter lowest n lengths.
        complement: If True, return complement of filtered set.

    Returns:
        Filtered DataFrame.
    """
    max_len = df_ranges[group['inf_type'].iloc[0]]['max']+1
    min_len = df_ranges[group['inf_type'].iloc[0]]['min']
    if highest:
        threshold = range(min_len, max_len)[-n]
        filtered_df = group[group['inf_length'] < threshold]
        complement_df = group[group['inf_length'] >= threshold]
    else:
        threshold = range(min_len, max_len)[n-1]
        filtered_df = group[group['inf_length'] > threshold]
        complement_df = group[group['inf_length'] <= threshold]
    return filtered_df if complement == False else complement_df


def format_input_output(
    df: pd.DataFrame,
    dataset_type: str,
    study_examples_to_filter: str = None
    ) -> pd.DataFrame:
    """Format input and output strings for the dataset.

    Args:
        df: Input DataFrame to format.
        dataset_type: Type of dataset ('base' or 'meta').
        study_examples_to_filter: Column name containing study examples to filter.

    Returns:
        DataFrame with formatted input and output strings.
    """
    if dataset_type == "base":
        df['input'] = df['knowledge_base'].astype(str) + "; <QUERY> " + df['query_hyp'].astype(str)
        df['output'] = ", " + df['query_inf'].astype(str) + " <STOP>"
    else:  # meta
        df['input'] = df['knowledge_base'].astype(str) + "; <STUDY> " + df[study_examples_to_filter].astype(str) + "; <QUERY> " + df['query_hyp'].astype(str)
        df['output'] = ", " + df['query_inf'].astype(str) + " <STOP>"
    return df


def constant_sample(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Sample a fixed number of datapoints for each combination of inf_type and inf_length.
    
    Args:
        df: DataFrame containing the data
        k: Number of datapoints to sample for each combination
        
    Returns:
        DataFrame with sampled data
    """
    result = []
    for inf_type in df['inf_type'].unique():
        for inf_length in df['inf_length'].unique():
            # Filter data for the current combination
            subset = df[(df['inf_type'] == inf_type) & (df['inf_length'] == inf_length)]
            
            if len(subset) != 0:
                if len(subset) <= k:
                    # If we have fewer samples than k, take all of them
                    logging.warning(f"Only {len(subset)} samples available for inf_type={inf_type}, "
                                f"inf_length={inf_length}, which is less than the requested {k}. "
                                f"Taking all samples.")
                    result.append(subset)
                else:
                    # Sample k datapoints from the current combination
                    result.append(subset.sample(k))
    
    # Concatenate all sampled data
    return pd.concat(result, ignore_index=True)


def prepare_baseline_ablation_dataset(df: pd.DataFrame, study_examples_to_filter: str) -> pd.DataFrame:
    """Prepare baseline dataset by expanding study examples into individual rows.

    Args:
        df: Input DataFrame.
        study_examples_to_filter: Column name containing study examples to expand.

    Returns:
        DataFrame with expanded study examples.
    """
    new_rows = []
    for _, row in df.iterrows():
        examples = row[study_examples_to_filter].split(';')
        for example in examples:
            query_hyp, query_inf = example.split(',')[0], example.split(',')[1]
            
            # Create new row by copying original row data and adding new query_hyp and query_inf
            new_row = row.copy()
            new_row['query_hyp'] = query_hyp.strip()
            new_row['query_inf'] = query_inf.strip()
            new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    final_df = pd.concat([df, new_df], axis=0, ignore_index=True)
    return final_df
