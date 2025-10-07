import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def learning_curves(
    log_dir: str = "results/full_logic/train_logs",
    result_dir: str = "results/full_logic",
    type_x_len_samples: int = 1000
) -> None:
    """Generate learning curve plots showing validation accuracy over training iterations.

    Args:
        result_dir: Directory containing the results CSV files.
            Defaults to "results/full_logic".
        log_dir: Directory containing the training log CSV files.
            Defaults to "results/full_logic/train_logs".
        type_x_len_samples: Number of samples for type x length.
            Defaults to 1000.
    
    The function generates plots for different experimental settings (core, short-to-long,
    long-to-short) and models (qwen-1.5b, qwen-3b, qwen-7b), comparing meta-learning vs
    baseline approaches using the best performing seed for each model.
    """
    # First find the best seed for each model from core results
    for exp in ['core', 'short-to-long', 'long-to-short']:

        exp_results = pd.read_csv(f"{result_dir}/results_{exp}.csv")
        exp_results['type_x_len_samples'] = exp_results['type_x_len_samples'].astype(str)
        exp_results['seed'] = exp_results['seed'].astype(str)

        for model in ['qwen-1.5b', 'qwen-3b', 'qwen-7b']:
            plt.figure(figsize=(10, 6))
            colors = ['tab:red', 'tab:blue']
            
            for idx, dataset in enumerate(['meta', 'base']):
                # Get all files for this model and setting
                seed_files = [f for f in os.listdir(log_dir) if f.startswith(f"{model}_lora_{dataset}_{exp}") and f.endswith(f"{type_x_len_samples}.csv")]
                
                # Get accuracies for this model and setting from core results
                model_accs = exp_results[
                    (exp_results['model'] == model) & 
                    (exp_results['dataset'] == dataset) &
                    (exp_results['type_x_len_samples'] == str(type_x_len_samples))
                ]['accuracy']
                
                # Find seed with highest accuracy
                best_seed = exp_results.iloc[model_accs.idxmax()].seed

                # Find corresponding log file
                best_file = next((f for f in seed_files if best_seed in f), None)

                if best_file:
                    df = pd.read_csv(f'{log_dir}/{best_file}')
                    plt.plot(df['iter'], df['eval/acc'], marker='o', linestyle='-', 
                            label=f'{dataset.capitalize()}', color=colors[idx])
            
            plt.xlabel('iterations')
            plt.ylabel('eval/accuracy')
            plt.title(f'Validation accuracy over iterations - {exp.capitalize()} ({model.capitalize()})')
            plt.legend()
            data_regime = "low" if type_x_len_samples == 100 else "high"
            plt.savefig(f'{result_dir}/plots/{model}_eval_acc_plot_{exp}_{data_regime}.png')
            plt.close()


def accuracy_heatmaps(
    result_dir: str = "results/full_logic",
    type_x_len_samples: int = 1000,
    test_type: str = "normal"
) -> None:
    """Generate heatmaps showing accuracy differences between meta-learning and baseline approaches.

    Args:
        result_dir: Directory containing the results CSV files.
            Defaults to "results/full_logic".
        type_x_len_samples: Number of samples for type x length.
            Defaults to 1000.
        test_type: Type of generalization considered.
    
    The function creates heatmaps for different experimental settings and models,
    visualizing the delta in accuracy between model_type=meta and model_type=base
    for the same dataset across different types and lengths.
    """
    for exp in ['core', 'short-to-long', 'long-to-short']:
        for model in ['qwen-1.5b', 'qwen-3b', 'qwen-7b']:
            
            df = pd.read_csv(f'{result_dir}/results_{exp}.csv')
            df['type_x_len_samples'] = df['type_x_len_samples'].astype(str)
            df['seed'] = df['seed'].astype(str)

            # Filter to desired cases
            df = df[
                (df['dataset'] == df['model_type']) |
                ((df['model_type'] == 'base') & (df['dataset'] == 'meta'))
            ]
            
            df = df[
                (df['model'] == model) & 
                (df['type_x_len_samples'] == str(type_x_len_samples)) &
                (df['test_type'] == test_type)
            ]
            
            df = df.drop(columns=['accuracy', 'ft_type', 'test_type'])
            
            df_melted = df.melt(id_vars=['model', 'model_type', 'dataset', 'seed', 'type_x_len_samples'], 
                                var_name='type_len', 
                                value_name='value')

            df_melted[['type', 'len']] = df_melted['type_len'].str.extract(r'type_(\d+)_len_(\d+)')
            df_melted['type'] = df_melted['type'].astype(int)
            df_melted['len'] = df_melted['len'].astype(int)
                        
            # Compute mean over seeds for each model_type and dataset
            df_melted_base = df_melted[df_melted['model_type'] == 'base'].groupby(['dataset', 'type', 'len'])['value'].mean().reset_index()
            df_melted_meta = df_melted[df_melted['model_type'] == 'meta'].groupby(['dataset', 'type', 'len'])['value'].mean().reset_index()

            # Merge the averaged results
            df_merged = pd.merge(df_melted_meta, df_melted_base, on=['dataset', 'type', 'len'], suffixes=('_meta', '_base'))
            df_merged['delta'] = df_merged['value_meta'] - df_merged['value_base']

            # Create crosstab for delta, averaging over datasets
            ct_delta = pd.crosstab(df_merged['type'], 
                                   df_merged['len'],
                                   values=df_merged['delta'],
                                   aggfunc='mean')

            # Create heatmap visualization for delta
            # Calculate figure size based on data dimensions to ensure square cells
            height, width = ct_delta.shape
            cell_size = 1.2  # Size of each cell in inches
            plt.figure(figsize=(width * cell_size + 3, height * cell_size + 2))  # Additional space for labels and colorbar
            
            cmap = sns.diverging_palette(240, 10, as_cmap=True)
            sns.heatmap(ct_delta, 
                annot=True, 
                fmt='.1f', 
                cmap=cmap, 
                center=0, 
                vmin=-ct_delta.abs().max().max(), 
                vmax=ct_delta.abs().max().max(),
                cbar_kws={'label': 'Δ Accuracy (%)', 'shrink': 0.7},
                square=True
            )
            plt.title(f'{exp.capitalize()} accuracy Δ - {model.capitalize()}')
            plt.xlabel('Length')
            plt.ylabel('Type')
            
            # Save plot
            data_regime = "low" if type_x_len_samples == 100 else "high"
            plt.savefig(f'{result_dir}/plots/{model}_heatmap_delta_{exp}_{data_regime}.png', bbox_inches='tight', dpi=300)
            plt.close()


def sota_comparison_heatmaps(
    result_dir: str = "results/full_logic",
    sota_model: str = "o3-mini",
    type_x_len_samples: int = 1000,
    test_type: str = "normal"
) -> None:
    """Generate heatmaps showing accuracy differences between meta-learning and SOTA model.

    Args:
        result_dir: Directory containing the results CSV files.
            Defaults to "results/full_logic".
        sota_model: Name of the SOTA model to compare against.
            Defaults to "o3-mini".
        type_x_len_samples: Number of samples for type x length.
            Defaults to 1000.
        test_type: Type of generalization considered.
    
    The function creates heatmaps for different experimental settings and models,
    visualizing the delta in accuracy between meta-learning and SOTA model
    across different types and lengths.
    """
    for exp in ['core']:
        for model in ['qwen-1.5b', 'qwen-3b', 'qwen-7b']:
            
            # Read the full dataset to get both the current model and SOTA
            df = pd.read_csv(f'{result_dir}/results_{exp}.csv')
            df['type_x_len_samples'] = df['type_x_len_samples'].astype(str)
            df['seed'] = df['seed'].astype(str)
            
            # Filter to desired cases
            df = df[
                (df['dataset'] == df['model_type']) |
                ((df['model_type'] == 'base') & (df['dataset'] == 'meta'))
            ]
            
            # Get meta-learning data for the current model and average over seeds
            df_meta = df[
                (df['model'] == model) & 
                (df['model_type'] == 'meta') &
                (df['dataset'] == 'meta') &
                (df['type_x_len_samples'] == str(type_x_len_samples)) &
                (df['test_type'] == test_type)
            ]
            df_sota = df[df['model'] == sota_model]
            
            # Skip if either is empty
            if df_meta.empty or df_sota.empty:
                print(f"Skipping {model} for {exp} - missing data")
                continue
            
            # Drop unnecessary columns
            df_meta = df_meta.drop(columns=['accuracy', 'ft_type', 'model', 'model_type', 'dataset', 'type_x_len_samples', 'test_type'])
            df_sota = df_sota.drop(columns=['accuracy', 'ft_type', 'model', 'model_type', 'dataset', 'type_x_len_samples', 'test_type'])
            
            # Melt and compute means
            df_meta_melted = df_meta.melt(id_vars=['seed'],
                                          var_name='type_len',
                                          value_name='value_meta')
            df_sota_melted = df_sota.melt(id_vars=['seed'],
                                          var_name='type_len',
                                          value_name='value_sota')

            # Extract type and len from the type_len column
            df_meta_melted[['type', 'len']] = df_meta_melted['type_len'].str.extract(r'type_(\d+)_len_(\d+)')
            df_sota_melted[['type', 'len']] = df_sota_melted['type_len'].str.extract(r'type_(\d+)_len_(\d+)')
            
            df_meta_melted['type'] = df_meta_melted['type'].astype(int)
            df_meta_melted['len'] = df_meta_melted['len'].astype(int)
            df_sota_melted['type'] = df_sota_melted['type'].astype(int)
            df_sota_melted['len'] = df_sota_melted['len'].astype(int)
            
            # Average over seeds
            df_meta_melted = df_meta_melted.groupby(['type', 'len'])['value_meta'].mean().reset_index()
            df_sota_melted = df_sota_melted.groupby(['type', 'len'])['value_sota'].mean().reset_index()
            
            # Merge meta and SOTA dataframes on type and len
            df_merged = pd.merge(df_meta_melted, df_sota_melted, on=['type', 'len'])

            # Compute the delta (meta - sota)
            df_merged['delta'] = df_merged['value_meta'] - df_merged['value_sota']

            # Create crosstab for delta
            ct_delta = pd.crosstab(df_merged['type'], 
                                   df_merged['len'],
                                   values=df_merged['delta'],
                                   aggfunc='mean')

            # Create heatmap visualization for delta
            # Calculate figure size based on data dimensions to ensure square cells
            height, width = ct_delta.shape
            cell_size = 1.2  # Size of each cell in inches
            plt.figure(figsize=(width * cell_size + 3, height * cell_size + 2))  # Additional space for labels and colorbar
            
            sns.heatmap(ct_delta, 
                annot=True, 
                fmt='.1f', 
                cmap="Spectral", 
                center=0, 
                vmin=-ct_delta.abs().max().max(), 
                vmax=ct_delta.abs().max().max(),
                cbar_kws={'label': 'Δ Accuracy (%)', 'shrink': 0.7},
                square=True
            )
            plt.title(f'{model.capitalize()} Meta-Learning vs {sota_model} - {exp.capitalize()} accuracy Δ')
            plt.xlabel('Length')
            plt.ylabel('Type')
            
            # Save plot
            data_regime = "low" if type_x_len_samples == 100 else "high"
            plt.savefig(f'{result_dir}/plots/{model}_vs_{sota_model}_heatmap_delta_{exp}_{data_regime}.png', bbox_inches='tight', dpi=300)
            plt.close()


def individual_accuracy_heatmaps(
    result_dir: str = "results/full_logic",
    type_x_len_samples: int = 1000,
    test_type: str = "normal"
) -> None:
    """Generate individual heatmaps showing accuracies for different combinations of model_type and dataset.

    Args:
        result_dir: Directory containing the results CSV files.
            Defaults to "results/full_logic".
        type_x_len_samples: Number of samples for type x length.
            Defaults to 1000.
        test_type: Type of generalization considered.
    
    The function creates separate heatmaps for different combinations of model_type and dataset
    for different experimental settings and models, visualizing the accuracy
    across different types and lengths.
    """
    # Define which experiments to run for each model
    model_experiments = {
        'qwen-1.5b': ['short-to-long', 'core', 'long-to-short'],
        'qwen-3b': ['short-to-long', 'core', 'long-to-short'],
        'qwen-7b': ['short-to-long', 'core', 'long-to-short'],
        'o3-mini': ['core'],
        'gpt-4o': ['core']
    }

    for model in model_experiments:
        for exp in model_experiments[model]:
            df = pd.read_csv(f'{result_dir}/results_{exp}.csv')
            df['type_x_len_samples'] = df['type_x_len_samples'].astype(str)
            df['seed'] = df['seed'].astype(str)

            # Filter to desired cases
            df = df[
                (df['dataset'] == df['model_type']) |
                ((df['model_type'] == 'base') & (df['dataset'] == 'meta'))
            ]

            df = df[
                (df['model'] == model) & 
                ((df['type_x_len_samples'] == str(type_x_len_samples)) | (df['type_x_len_samples'] == '-')) &
                (df['test_type'] == test_type)
            ]
            
            if df.empty:
                continue

            df = df.drop(columns=['accuracy', 'ft_type', 'test_type'])
            
            df_melted = df.melt(id_vars=['model', 'model_type', 'dataset', 'seed', 'type_x_len_samples'], 
                                var_name='type_len', 
                                value_name='value')

            df_melted[['type', 'len']] = df_melted['type_len'].str.extract(r'type_(\d+)_len_(\d+)')
            df_melted['type'] = df_melted['type'].astype(int)
            df_melted['len'] = df_melted['len'].astype(int)
            
            combinations = [('base', 'base'), ('meta', 'meta'), ('base', 'meta')]
            
            for model_type, dataset in combinations:
                # Average over seeds for this combination
                df_setting = df_melted[(df_melted['model_type'] == model_type) & (df_melted['dataset'] == dataset)].groupby(['type', 'len'])['value'].mean().reset_index()
                
                if df_setting.empty:
                    continue
                
                # Create crosstab
                ct = pd.crosstab(df_setting['type'], 
                                df_setting['len'],
                                values=df_setting['value'],
                                aggfunc='mean')
            
                # Exp name and model name for titles
                if exp == "short-to-long":
                    exp_name = "Short-to-Long"
                    cmap = "Reds"
                elif exp == "core":
                    exp_name = "Core Generalization"
                    cmap = "Blues"
                elif exp == "long-to-short":
                    exp_name = "Long-to-Short"
                    cmap = "Greens"
                else:
                    ValueError(f"Unknown experiment type: {exp}")

                if model.startswith('qwen'):
                    size = model.split("-")[1].upper()
                    model_name = f"Qwen-2.5 {size}"
                    setting_name = f"{'ML' if model_type == 'meta' else 'Baseline'} on {dataset.capitalize()}"
                else:
                    if model == 'gpt-4o':
                        model_name = 'GPT-4o'
                    else:
                        model_name = model
                    setting_name = f"{'Few-shot' if model_type == 'meta' else 'Zero-shot'} on {dataset.capitalize()}"

                # Create heatmap visualization
                height, width = ct.shape
                cell_size = 1.2
                plt.figure(figsize=(width * cell_size + 3, height * cell_size + 2))
                
                sns.heatmap(ct, 
                    annot=True, 
                    fmt='.1f', 
                    cmap=cmap, 
                    vmin=0, 
                    vmax=100,
                    cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.7},
                    square=True
                )
                
                plt.title(f'{setting_name} - {exp_name} - {model_name}', fontsize=23)
                plt.xlabel('Length', fontsize=14)
                plt.ylabel('Type', fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                
                # Save plot
                data_regime = "low" if type_x_len_samples == 100 else "high"
                plt.savefig(f'{result_dir}/plots/{model}_heatmap_{model_type}_{dataset}_{exp}_{data_regime}.png', bbox_inches='tight', dpi=300)
                plt.close()


def core_generalization_table(
    result_dir: str = "results/full_logic",
    type_x_len_samples: int = 1000
) -> None:
    """Create and print a summary table of core generalization results for different models and combinations.

    This function reads results from a CSV file and generates a formatted table showing model performance
    on all lengths, on top 5 shortest lengths only and on top 5 longest lengths only for different
    combinations of model_type and dataset. Results are based on 1000 samples per type/length combination.

    Args:
        result_dir (str, optional): Directory path containing the results CSV files.
            Defaults to "results/full_logic".
        type_x_len_samples (int, optional): Number of samples for type x length. Defaults to 1000.
    """
    def aggregate_accuracies(
        csv_path: str, 
        type_x_len: int = 1000
    ) -> pd.DataFrame:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_path)

        # Filter to desired cases
        df = df[
            (df['dataset'] == df['model_type']) |
            ((df['model_type'] == 'base') & (df['dataset'] == 'meta'))
        ]

        # Get all column names
        cols = df.columns.tolist()

        # Create a dictionary to store type numbers and their cases
        type_cases = {}

        # Find columns matching pattern 'type_X_len_Y' and organize them
        for col in cols:
            if col.startswith('type_') and '_len_' in col:
                type_num = col.split('_')[1]
                if type_num.isdigit():
                    if type_num not in type_cases:
                        type_cases[type_num] = []
                    case_parts = col.split('_len_')[1]
                    type_cases[type_num].append(case_parts)

        # For each type, create new columns for short and long cases
        for type_num in type_cases:
            # Sort cases by length
            sorted_cases = sorted(type_cases[type_num], key=lambda x: int(x))
            short_cases = sorted_cases[:5]  # 5 shortest
            long_cases = sorted_cases[-5:]  # 5 longest

            # Create columns for short cases average
            short_cols = [f'type_{type_num}_len_{case}' for case in short_cases]
            df[f'type_{type_num}_short_mean'] = df[short_cols].mean(axis=1)

            # Create columns for long cases average
            long_cols = [f'type_{type_num}_len_{case}' for case in long_cases]
            df[f'type_{type_num}_long_mean'] = df[long_cols].mean(axis=1)

        # Average across all types
        short_cols = [col for col in df.columns if 'short_mean' in col]
        long_cols = [col for col in df.columns if 'long_mean' in col]
        
        df['all_types_short_mean'] = df[short_cols].mean(axis=1)
        df['all_types_long_mean'] = df[long_cols].mean(axis=1)

        # Define grouping columns
        group_cols = ['model', 'type_x_len_samples', 'test_type', 'ft_type', 'model_type', 'dataset']
        
        # Use only the collapsed columns across all types
        metric_cols = ['all_types_short_mean', 'all_types_long_mean']
        
        # Group by columns and calculate mean and std for collapsed metric columns
        aggregated = df.groupby(group_cols)[metric_cols].agg(['mean', 'std']).reset_index()
        
        # Flatten column names from MultiIndex
        aggregated.columns = [f"{col[0]}_{col[1]}" if col[1] in ['mean', 'std'] else col[0] 
                                for col in aggregated.columns]
        
        aggregated = aggregated[aggregated['test_type'] == 'normal']
        aggregated = aggregated[(aggregated['type_x_len_samples'] == str(type_x_len)) | (aggregated['type_x_len_samples'] == "-")]
        aggregated = aggregated.drop(columns=['type_x_len_samples', 'test_type', 'ft_type'])

        return aggregated

    # Read the core results
    core_df = pd.read_csv(f'{result_dir}/results_core.csv')
    core_df['type_x_len_samples'] = core_df['type_x_len_samples'].astype(str)
    core_df['seed'] = core_df['seed'].astype(str)

    # Filter to desired cases
    core_df = core_df[
        (core_df['dataset'] == core_df['model_type']) |
        ((core_df['model_type'] == 'base') & (core_df['dataset'] == 'meta'))
    ]

    # Get aggregated accuracies for short and long
    aggregated_df = aggregate_accuracies(f'{result_dir}/results_core.csv', type_x_len_samples)

    print(f"\nCore Generalization Summary Table ({type_x_len_samples} type_x_len samples)")
    print("-" * 95)
    print(f"{'Model':<12} {'Type':<20} {'All':<20} {'Short':<20} {'Long':<20}")
    print("-" * 95)

    models = sorted(core_df['model'].unique())
    combinations = [('base', 'base'), ('meta', 'meta'), ('base', 'meta')]
    for model in models:
        for model_type, dataset in combinations:
            if model in ['o3-mini', 'gpt-4o']:
                type_name = f"{'Few-shot' if model_type == 'meta' else 'Zero-shot'} on {dataset.capitalize()}"
            else:
                type_name = f"{'ML' if model_type == 'meta' else 'Baseline'} on {dataset.capitalize()}"
            row = [model, type_name]
            
            # All lengths (original accuracy)
            df = core_df[
                (core_df['model'] == model) &
                (core_df['model_type'] == model_type) &
                (core_df['dataset'] == dataset) &
                ((core_df['type_x_len_samples'] == str(type_x_len_samples)) | 
                    (core_df['type_x_len_samples'] == "-")) &
                (core_df['test_type'] == 'normal')
            ]
            if df.empty:
                all_stat = "N/A"
            else:
                mean = df['accuracy'].mean()
                std = df['accuracy'].std()
                all_stat = f"{mean:.2f}" if pd.isna(std) else f"{mean:.2f} ± {std:.2f}"
            row.append(all_stat)
            
            # Short and Long from aggregated data
            agg_df = aggregated_df[
                (aggregated_df['model'] == model) &
                (aggregated_df['model_type'] == model_type) &
                (aggregated_df['dataset'] == dataset)
            ]
            
            if agg_df.empty:
                short_stat = "N/A"
                long_stat = "N/A"
            else:
                # Short
                short_mean = agg_df['all_types_short_mean_mean'].iloc[0]
                short_std = agg_df['all_types_short_mean_std'].iloc[0]
                short_stat = f"{short_mean:.2f}" if pd.isna(short_std) else f"{short_mean:.2f} ± {short_std:.2f}"
                
                # Long
                long_mean = agg_df['all_types_long_mean_mean'].iloc[0]
                long_std = agg_df['all_types_long_mean_std'].iloc[0]
                long_stat = f"{long_mean:.2f}" if pd.isna(long_std) else f"{long_mean:.2f} ± {long_std:.2f}"
            
            row.extend([short_stat, long_stat])
            print(f"{row[0]:<12} {row[1]:<20} {row[2]:<20} {row[3]:<20} {row[4]:<20}")
    print("-" * 95)


def lexical_generalization_table(result_dir: str = "results/full_logic") -> None:
    """Create and print a summary table of lexical generalization results for different models and combinations.

    This function reads results from a CSV file and generates a formatted table showing model performance
    across different test conditions (seen pseudowords, OOD pseudowords, and OOD constants) for different
    combinations of model_type and dataset. Results are based on 1000 samples per type/length combination.

    Args:
        result_dir (str, optional): Directory path containing the results CSV files.
            Defaults to "results/full_logic".
    """
    type_x_len_samples = 1000
    core_df = pd.read_csv(f'{result_dir}/results_core.csv')
    core_df['type_x_len_samples'] = core_df['type_x_len_samples'].astype(str)
    core_df['seed'] = core_df['seed'].astype(str)

    # Filter to desired cases
    core_df = core_df[
        (core_df['dataset'] == core_df['model_type']) |
        ((core_df['model_type'] == 'base') & (core_df['dataset'] == 'meta'))
    ]

    print("\nLexical Generalization Summary Table (1000 type_x_len samples)")
    print("-" * 95)
    print(f"{'Model':<12} {'Type':<20} {'Core':<15} {'Unseen Pseudowords':<20} {'Unseen Constants':<15}")
    print("-" * 95)

    models = sorted(core_df['model'].unique())
    combinations = [('base', 'base'), ('meta', 'meta'), ('base', 'meta')]
    for model in models:
        for model_type, dataset in combinations:
            if model in ['o3-mini', 'gpt-4o']:
                type_name = f"{'Few-shot' if model_type == 'meta' else 'Zero-shot'} on {dataset.capitalize()}"
            else:
                type_name = f"{'ML' if model_type == 'meta' else 'Baseline'} on {dataset.capitalize()}"
            row = [model, type_name]
            for test_type in ['normal', 'ood_words', 'ood_constants']:
                df = core_df[
                    (core_df['model'] == model) &
                    (core_df['model_type'] == model_type) &
                    (core_df['dataset'] == dataset) &
                    ((core_df['type_x_len_samples'] == str(type_x_len_samples)) | 
                     (core_df['type_x_len_samples'] == "-")) &
                    (core_df['test_type'] == test_type)
                ]
                if df.empty:
                    stat = "N/A"
                else:
                    mean = df['accuracy'].mean()
                    std = df['accuracy'].std()
                    stat = f"{mean:.2f}" if pd.isna(std) else f"{mean:.2f} ± {std:.2f}"
                row.append(stat)
            print(f"{row[0]:<12} {row[1]:<20} {row[2]:<15} {row[3]:<20} {row[4]:<15}")
    print("-" * 95)


def generalization_summary_table(
    result_dir: str = "results/full_logic",
    type_x_len_samples: int = 1000
) -> None:
    """Creates and prints a summary table of model performance on generalization tasks.

    Creates a formatted table showing accuracy results for different models on short-to-long and
    long-to-short tasks, comparing performance between different combinations of model_type and dataset
    for both in-distribution (ID) and out-of-distribution (OOD) scenarios.

    Args:
        result_dir (str, optional): Directory containing the results CSV files. 
            Defaults to "results/full_logic".
        type_x_len_samples (int, optional): Number of samples for type x length.
            Defaults to 1000.
    """
    rec_df = pd.read_csv(f'{result_dir}/results_short-to-long.csv')
    comp_df = pd.read_csv(f'{result_dir}/results_long-to-short.csv')
    for df in [rec_df, comp_df]:
        df['type_x_len_samples'] = df['type_x_len_samples'].astype(str)
        df['seed'] = df['seed'].astype(str)

        # Filter to desired cases
        df = df[
            (df['dataset'] == df['model_type']) |
            ((df['model_type'] == 'base') & (df['dataset'] == 'meta'))
        ]

    print(f"\nGeneralization Summary Table ({type_x_len_samples} type_x_len samples)")
    print("-" * 105)
    print(f"{'Model':<12} {'Type':<20} {'          Short->Long':<35} {'          Long->Short':<35}")
    print(f"{'':12} {'':20} {'Disaligned':<20} {'Aligned':<15} {'Disaligned':<20} {'Aligned':<15}")
    print("-" * 105)

    models = sorted(set(rec_df['model'].unique()) | set(comp_df['model'].unique()))
    combinations = [('base', 'base'), ('meta', 'meta'), ('base', 'meta')]
    for model in models:
        for model_type, dataset in combinations:
            if model in ['o3-mini', 'gpt-4o']:
                if model_type == 'base':
                    continue
                else:
                    type_name = f"{'Few-shot' if model_type == 'meta' else 'Zero-shot'} on {dataset.capitalize()}"
            else:
                type_name = f"{'ML' if model_type == 'meta' else 'Baseline'} on {dataset.capitalize()}"
            
            row = [model, type_name]
            # short-to-long
            for test_type in ['normal', 'ood_support']:
                df = rec_df[
                    (rec_df['model'] == model) &
                    (rec_df['model_type'] == model_type) &
                    (rec_df['dataset'] == dataset) &
                    ((rec_df['type_x_len_samples'] == str(type_x_len_samples)) | 
                     (rec_df['type_x_len_samples'] == "-")) &
                    (rec_df['test_type'] == test_type)
                ]
                if df.empty:
                    stat = "N/A"
                else:
                    mean = df['accuracy'].mean()
                    std = df['accuracy'].std()
                    stat = f"{mean:.2f}" if pd.isna(std) else f"{mean:.2f} ± {std:.2f}"
                row.append(stat)
            # long-to-short
            for test_type in ['normal', 'ood_support']:
                df = comp_df[
                    (comp_df['model'] == model) &
                    (comp_df['model_type'] == model_type) &
                    (comp_df['dataset'] == dataset) &
                    ((comp_df['type_x_len_samples'] == str(type_x_len_samples)) | 
                     (comp_df['type_x_len_samples'] == "-")) &
                    (comp_df['test_type'] == test_type)
                ]
                if df.empty:
                    stat = "N/A"
                else:
                    mean = df['accuracy'].mean()
                    std = df['accuracy'].std()
                    stat = f"{mean:.2f}" if pd.isna(std) else f"{mean:.2f} ± {std:.2f}"
                row.append(stat)
            print(f"{row[0]:<12} {row[1]:<20} {row[2]:<15} {row[3]:<20} {row[4]:<15} {row[5]:<20}")
    print("-" * 105)


def low_data_summary_tables(
    result_dir: str = "results/full_logic",
    type_x_len_samples: int = 100
) -> None:
    """Summarize model results for low data regime and create a formatted summary table.

    This function reads CSV files containing model performance results and generates a summary
    table comparing different models across three evaluation aspects: core performance,
    long-to-short, and short-to-long. For each model, it shows results for different combinations
    of model_type and dataset where applicable.

    Args:
        result_dir (str, optional): Directory path containing the results CSV files.
            Defaults to "results/full_logic".
        type_x_len_samples (int, optional): Number of samples for type x length.
            Defaults to 100.
    """
    core_df = pd.read_csv(f'{result_dir}/results_core.csv')
    comp_df = pd.read_csv(f'{result_dir}/results_long-to-short.csv')
    rec_df = pd.read_csv(f'{result_dir}/results_short-to-long.csv')
    for df in [core_df, comp_df, rec_df]:
        df['type_x_len_samples'] = df['type_x_len_samples'].astype(str)
        df['seed'] = df['seed'].astype(str)

        # Filter to desired cases
        df = df[
            (df['dataset'] == df['model_type']) |
            ((df['model_type'] == 'base') & (df['dataset'] == 'meta'))
        ]

    print(f"\nLow Data Regime Summary Table ({type_x_len_samples} type_x_len samples)")
    print("-" * 85)
    print(f"{'Model':<12} {'Type':<20} {'Core':<15} {'Long->Short':<15} {'Short->Long':<15}")
    print("-" * 85)

    models = sorted(set(core_df['model'].unique()) | set(comp_df['model'].unique()) | set(rec_df['model'].unique()))
    combinations = [('base', 'base'), ('meta', 'meta'), ('base', 'meta')]
    for model in models:
        for model_type, dataset in combinations:
            if model in ['o3-mini', 'gpt-4o']:
                continue
            else:
                type_name = f"{'ML' if model_type == 'meta' else 'Baseline'} on {dataset.capitalize()}"
            row = [model, type_name]
            for df in [core_df, comp_df, rec_df]:
                d = df[
                    (df['model'] == model) &
                    (df['model_type'] == model_type) &
                    (df['dataset'] == dataset) &
                    ((df['type_x_len_samples'] == str(type_x_len_samples)) | 
                     (df['type_x_len_samples'] == "-")) &
                    (df['test_type'] == 'normal')
                ]
                if d.empty:
                    stat = "N/A"
                else:
                    mean = d['accuracy'].mean()
                    std = d['accuracy'].std()
                    stat = f"{mean:.2f}" if pd.isna(std) else f"{mean:.2f} ± {std:.2f}"
                row.append(stat)
            print(f"{row[0]:<12} {row[1]:<20} {row[2]:<15} {row[3]:<15} {row[4]:<15}")
    print("-" * 85)


if __name__ == "__main__":

    os.makedirs("results/full_logic/plots", exist_ok=True)
    learning_curves(type_x_len_samples=1000)
    accuracy_heatmaps(type_x_len_samples=1000)
    individual_accuracy_heatmaps(type_x_len_samples=1000)
    sota_comparison_heatmaps(type_x_len_samples=1000)
    core_generalization_table(type_x_len_samples=1000)
    generalization_summary_table()
    low_data_summary_tables()
    lexical_generalization_table()
