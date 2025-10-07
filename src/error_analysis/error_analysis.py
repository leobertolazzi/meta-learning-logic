import os
import json

from typing import Dict, List, Set, Any

import numpy as np
import pandas as pd


class Errors:
    def __init__(self, errors_file: str, kb_subs_file: str):
        """
        Initialize the Errors class for error analysis.

        Args:
            errors_file (str): Path to the JSON file containing incorrect predictions.
            kb_subs_file (str): Path to the JSON file containing KB substitutions.
        """
        self.eval = json.load(open(errors_file, 'r'))   
        self.name = self.get_name(errors_file)         
        self.ds = json.load(open(kb_subs_file, 'r'))    
        self.errors = self.compute_errors()            
        self.types = self.eval.keys()                 

    def get_name(self, file_path: str) -> str:
        """
        Extract the base filename (without extension) from a file path.

        Args:
            file_path (str): The file path.

        Returns:
            str: The base filename without extension.
        """
        file_name = os.path.basename(file_path)
        return os.path.splitext(file_name)[0]

    # FUNCTIONS TO CAPTURE DIFFERENT TYPES OF ERRORS
    def __well_formed_formulas(self, formulas: Set[str]) -> bool:
        """
        Check if all formulas in the set are well-formed.

        Args:
            formulas (Set[str]): A set of formulas.

        Returns:
            bool: True if all are well-formed formulas, False otherwise.
        """
        checked = [False]*len(formulas)
        for i, f in enumerate(formulas):
            split = f.split()
            if (len(split) == 4 and split[0] in ['all', 'some', 'no'] and split[2] == 'are') or \
               (len(split) == 5 and split[0] == 'some' and split[2] == 'are' and split[3] == 'not'): # well-formed formula
               checked[i] = True
        if all(checked):
            return True
        return False

    def compute_errors(self) -> Dict[str, Dict[str, List[Any]]]:
        """
        Iterates the "incorrect predictions" file and for each type and length
        computes and stores various error metrics.

        Returns:
            Dict[str, Dict[str, List[Any]]]: Nested dictionary with error statistics.
        """
        d ={t:{} for t in self.eval.keys()}
        for t,T in self.eval.items():
            d[t] = {l:[0,0,[],[],0,0] for l in T.keys()} 
            for l,L in T.items():
                d[t][l][0] = len(L) # <- [0]
                for h,v in L.items():
                    prediction = {f.strip() for f in v['prediction'].split(',')}
                    if self.__well_formed_formulas(prediction):
                        d[t][l][1] += 1 # <- [1]
                        answer = {f.strip() for f in v['answer'].split(',')}

                        # non-minimal set of premises
                        if answer.issubset(prediction):
                            d[t][l][2].append(len(prediction) - len(answer)) # <- [2]

                        # incomplete A-chains
                        correct_a_chain = {f for f in prediction if f.split()[0] == 'all'}.intersection({f for f in answer if f.split()[0] == 'all'})
                        missing = int(l) - len(correct_a_chain) # distance from ground truth
                        if missing > 0:
                            d[t][l][3].append(missing) # <- [3]

                        # hallucination rate
                        KB = {f for f in self.ds[v['kb_id']]['KB']}
                        vocabulary = {t for t in self.ds[v['kb_id']]['substitutions'].values()}
                        
                        predicted_terms = {t for f in prediction for t in [f.split()[1],f.split()[-1]]}
                        if not prediction.issubset(KB):
                            d[t][l][4] += 1 # <- [4]
                        if not predicted_terms.issubset(vocabulary):
                            d[t][l][5] += 1 # <- [5]

        return d

    # FUNCTIONS TO BUILD AND SAVE A PANDAS DATAFRAME
    def f(self, value: float, scale: int = 100, precision: int = 2) -> float:
        """
        Format a float value by scaling and rounding.

        Args:
            value (float): The value to format.
            scale (int, optional): Scaling factor. Defaults to 100.
            precision (int, optional): Number of decimal places. Defaults to 2.

        Returns:
            float: The formatted value.
        """
        return round(value*scale, precision)

    def op(self, value: Any, avg_list: bool = False) -> Any:
        """
        Compute either the mean of a list or a ratio (percentage).

        Args:
            value (Any): List or tuple of values.
            avg_list (bool, optional): Whether to average a list. Defaults to False.

        Returns:
            Any: The computed value or '-' if not applicable.
        """
        if avg_list: # mean of a list
            if value: 
                return self.f(np.mean(value), scale=1)
            else:
                return '-'
        else: # ratio (percentage)
            return self.f(value[0]/value[1])

    def create_row(self, label: str, val: List[Any]) -> List[Any]:
        """
        Create a row for the summary table.

        Args:
            label (str): Row label.
            val (List[Any]): List of error statistics.

        Returns:
            List[Any]: Row for the DataFrame.
        """
        return [label, 
                self.op([val[1], val[0]]),  
                self.op([len(val[2]), val[1]]), 
                self.op(val[2], avg_list=True),
                self.op([len(val[3]), val[1]]),
                self.op(val[3], avg_list=True),
                self.op([val[4], val[1]]),
                self.op([val[5], val[1]])]

    def main_table(self, show: bool = True, save: bool = False) -> pd.DataFrame:
        """
        Build and optionally display or save the main error analysis table.

        Args:
            show (bool, optional): Whether to print the table. Defaults to True.
            save (bool, optional): Whether to save the table as CSV. Defaults to False.

        Returns:
            pd.DataFrame: The summary table as a DataFrame.
        """
        table = []
        total = [0,0,[],[],0,0]
        for t in self.types:
            by_types = [0,0,[],[],0,0]
            for i in range(6):
                if type(by_types[i]) == int:
                    by_types[i] = sum([L[i] for L in self.errors[t].values()])
                else:
                    by_types[i] = [l for L in self.errors[t].values() for l in L[i]]

                # update total
                total[i] += by_types[i]

            # update table (by types)
            table.append(self.create_row(t, by_types))

        # update table (last row)
        table.append(self.create_row('Total', total))

        # create dataframe
        columns = ['Type', 'WFF', 'NVM', 'Avg NVM', 'MAP', 'Avg MAP', 'HP', 'HT']
        df = pd.DataFrame(table, columns=columns)
        
        # show table
        if show: 
            print(df)
        
        # save as 'csv' file
        if save:
            filename = f'{self.name}.csv'
            df.to_csv(filename, index=False)
            print(f'File successfully saved to {filename}')

        return df


def process_all_error_files(source: str = 'results/full_logic/filtered_errors') -> Dict[str, pd.DataFrame]:
    """
    Process all error files in the specified directory and return their analysis tables.

    Args:
        source (str, optional): Directory containing error files. Defaults to 'results/full_logic/filtered_errors'.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping setting names to DataFrames.
    """
    error_analysis_dict = {}
    for filename in os.listdir(source):
        if not filename.startswith('.'):
            file_path = os.path.join(source, filename)
            sub_file = 'src/error_analysis/error_analysis_test.json'
            if 'constants' in filename:
                sub_file = 'src/error_analysis/error_analysis_test_constants.json'
            err = Errors(errors_file=file_path, kb_subs_file=sub_file)
            setting = err.get_name(file_path)
            df = err.main_table(show=False, save=False)
            error_analysis_dict[setting] = df
    return error_analysis_dict


def error_analysis_table(
    tables: Dict[str, pd.DataFrame],
    setting_filter: str = "core",
    type_x_len_samples: int = 1000
) -> None:
    """
    Process the files and print results filtered by the specified setting.

    Args:
        tables (Dict[str, pd.DataFrame]): A dictionary where keys are filenames and values are DataFrames.
        setting_filter (str): The specific setting to filter by. If None, process all settings.
        type_x_len_samples (int): The required type_x_len value to filter.
    """
    # Initialize an empty list to store the filtered rows
    filtered_data = []

    # Iterate through all files in the directory
    for filename in tables.keys():

        # Extract the model name and setting from the filename
        model_name = filename.split('_')[0]
        setting = filename.split('_')[3]
        test_type = filename.split('_')[-1]
        # Check if after 100 or 1000 in the filename there is 'meta' or 'base'
        parts = filename.split('_')
        dataset = None
        for i in range(len(parts) - 1):
            if parts[i] in ['100', '1000'] and parts[i + 1] in ['meta', 'base']:
                dataset = parts[i + 1]
                break
        if dataset is None:
            dataset = filename.split('_')[2]

        if 'qwen' in model_name:
            model_type = "ML" if filename.split('_')[2] == 'meta' else "Baseline"
            if filename.split('_')[2] == dataset:
                type_x_len = filename.split('_')[-2] if test_type == 'normal' else filename.split('_')[-3]
                if test_type == 'normal' and model_type == "ML" and setting != 'core':
                    type_x_len = filename.split('_')[-2]
                    model_type += "(disaligned)"
                elif test_type == 'support' and model_type == "ML":
                    type_x_len = filename.split('_')[-3]
                    model_type += "(aligned)"
                else:
                    type_x_len = filename.split('_')[-2]
            else:
                type_x_len = filename.split('_')[-3] if test_type == 'normal' else filename.split('_')[-4]
                if test_type == 'normal' and model_type == "ML" and setting != 'core':
                    type_x_len = filename.split('_')[-3]
                    model_type += "(disaligned)"
                elif test_type == 'support' and model_type == "ML":
                    type_x_len = filename.split('_')[-4]
                    model_type += "(aligned)"
                else:
                    type_x_len = filename.split('_')[-3]
        else: # gpt
            model_type = "Few-shot" if filename.split('_')[2] == 'meta' else "Zero-shot"
            type_x_len = None

        if test_type != 'normal' and setting == 'core':
            continue

        if type_x_len:
            if int(type_x_len) != type_x_len_samples:
                continue

        # Apply the setting filter during data generation
        if setting_filter and setting != setting_filter:
            continue

        # Read the CSV file
        df = tables[filename]

        # Remove the HT column and extract the "Total" 
        df.drop(columns=['HT'], inplace=True)
        total_row = df[df['Type'] == 'Total'].copy()  # Make a copy to avoid SettingWithCopyWarning
        
        if not total_row.empty:
            # Add the model name as a new column using .loc
            total_row.loc[:, 'Model'] = model_name
            total_row.loc[:, 'Method'] = model_type
            total_row.loc[:, 'Dataset'] = dataset

            # Append the row to the filtered data list
            filtered_data.append(total_row)

    # Combine all the filtered rows into a single DataFrame
    result_df = pd.concat(filtered_data, ignore_index=True)
    result_df.drop(columns=['Type'], inplace=True)  # Remove the 'Type' column

    # Reorder columns to have Model as the first column
    columns = ['Model', 'Method', 'Dataset'] + [col for col in result_df.columns if col not in ['Model', 'Method', 'Dataset']]
    result_df = result_df[columns]

    # Sort the DataFrame by the 'Model' column
    result_df.sort_values(by='Model', inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    
    # Add separator line after header
    lines = result_df.to_string(index=False, justify='center', col_space=10).split('\n')
    header_line = '-' * len(lines[0])
    print(header_line)  # Separator
    print(lines[0])  # Header
    print(header_line)  # Separator
    for line in lines[1:]:  # Rows
        print(line)
    print(header_line)  # Separator


if __name__ == '__main__':

    # Run the error analysis and save the results
    error_analysis_dict = process_all_error_files()

    # Print the error analysis for length generalization
    print("\n\nError analysis on Length Generalization")
    print("\nLong->Short")
    error_analysis_table(error_analysis_dict, setting_filter="long-to-short", type_x_len_samples=1000)
    print("\nShort->Long")
    error_analysis_table(error_analysis_dict, setting_filter="short-to-long", type_x_len_samples=1000)

    # Print the error analysis for core generalization
    print("\nError analysis on Core Generalization")
    error_analysis_table(error_analysis_dict, setting_filter="core", type_x_len_samples=1000)
