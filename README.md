<h1 align="center">Teaching Small Language Models to Learn Logic through Meta-Learning</h1>

This repository contains the code for the paper: Leonardo Bertolazzi, Manuel Vargas Guzmán, Raffaella Bernardi, Maciej Malicki, Jakub Szymanik (2025). [*Teaching Small Language Models to Learn Logic through Meta-Learning*](https://arxiv.org/pdf/2505.14313).

> **Abstract:** Large language models (LLMs) are increasingly evaluated on reasoning tasks, yet their logical abilities remain contested. To address this, we study LLMs’ reasoning in a well-defined fragment of logic: syllogistic reasoning. We cast the problem as premise selection and construct controlled datasets to isolate logical competence. Beyond evaluation, an open challenge is enabling LLMs to acquire abstract inference patterns that generalize to novel structures. We propose to apply few-shot meta-learning to this domain, thereby encouraging models to extract rules across tasks rather than memorize patterns within tasks. Although meta-learning has been little explored in the context of logic learnability, our experiments show that it is effective: small models (1.5B–7B) fine-tuned with meta-learning demonstrate strong gains in generalization, with especially pronounced benefits in low-data regimes. These meta-learned models outperform GPT-4o and o3-mini on our syllogistic reasoning task.

## Table of Contents
- [Setup](#setup)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Main Experiments](#main-experiments)
  - [Fine-tuning and Test](#fine-tuning-and-test)
  - [API Models Evaluation](#api-models-evaluation)
- [Analysis of Results](#analysis-of-results)
- [License](#license)
- [Citation](#citation)

## Setup

Create and activate the environment used for the project with Conda:

```bash
conda env create -f environment.yaml
conda activate syllogistic_llms
```

## Dataset

The datasets used in our experiments are publicly available on Hugging Face:

- [leobertolazzi/syllogistic-logic](https://huggingface.co/datasets/leobertolazzi/syllogistic-logic)

You can download and explore the dataset directly from the link above.

## Project Structure

- **Source code:** Located in the `src/` directory.
- **Experiment scripts:** Bash scripts for running experiments are in the `experiments/` folder.
- **Results:** Full tables and plots from the paper are saved in the `results/` directory.

## Main Experiments

### Fine-tuning and Test

We provide a bash script, `train_test.sh`, to facilitate the fine-tuning and evaluation of local models (Qwen) on the logical premise selection task. This script automates both training and testing phases for the three experimental settings: core generalization, long-to-short, and short-to-long generalization. You can configure the model, dataset, and training parameters directly in the script. Notably, it is possible to train a model on a dataset (e.g. `"meta"`) and test it on another (e.g. `"base"`)

To use:
```bash
./experiments/train_test.sh
```
**Important variables to set in `train_test.sh`:**
- `MODEL`: The model to use (`"qwen-1.5b"`, `"qwen-3b"`, `"qwen-7b"`).
- `dataset`: The dataset type (`"meta"` for MIND meta-learning, `"base"` for baseline).
- `EPOCHS`: Number of training epochs (e.g., `1` for "meta", `4` for "base").
- `SEED`: Random seed for reproducibility (we used the three seeds `1048`, `1596`, `512`).
- `PRECISION`: Precision type (`"bfloat16"` for "base", `"int4"` for "meta").
- Other parameters (batch size, learning rate, etc.) can also be adjusted as needed.

**For `src/test.py` only:**
- `test_model_type`: The dataset type (`"meta"` for meta-learning, `"base"` for baseline) used at training time. This allows loading a specific saved model (trained on `"meta"` or `"base"` data) and testing it on a different `dataset` (`"meta"` or `"base"`).

Edit these variables at the top of the script to match your desired configuration (e.g., model size, dataset type, number of epochs, etc.).

### API Models Evaluation

For evaluating API-based models (such as GPT-4o or o3-mini), we provide the `api.sh` script. This script handles querying the API endpoints and evaluating model performance on the logical premise selection task. You can set the Azure endpoint, model, and other parameters within the script.

To use:
```bash
./experiments/api.sh
```
**Important variables to set in `api.sh`:**
- `AZURE_ENDPOINT`: Your Azure OpenAI endpoint URL (e.g., `"https://your-endpoint.openai.azure.com/"`). This must be set for the script to work.
- `MODEL`: The API model to use (e.g., `"o3-mini"`, `"gpt-4o"`).
- `DEPLOYMENT`: The deployment name for your Azure model (e.g., `"o3-mini"`, `"gpt-4o"`).
- Other parameters (dataset, experiment type, etc.) can be adjusted as needed.

Make sure to set your Azure endpoint and any required API credentials in the script before running.

## Analysis of Results

To generate all tables, plots, and error analyses—including accuracy heatmaps, result tables, and error statistics—use the provided `results.sh` script:

```bash
./experiments/results.sh
```

All outputs will be printed in the terminal or saved in the `results/` directory.

## License
[![MIT license](https://img.shields.io/badge/License-Creative%20Commons%20Attribution--ShareAlike%204.0%20International%20Public%20License-green.svg)](https://creativecommons.org/licenses/by-sa/4.0)

This work is licensed under a [CC BY-SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).

## Citation

If you find our work helpful, you can cite this paper as:
```
@misc{bertolazzi-et-al-2025-mind,
      title={A MIND for Reasoning: Meta-learning for In-context Deduction}, 
      author={Leonardo Bertolazzi and Manuel Vargas Guzmán and Raffaella Bernardi and Maciej Malicki and Jakub Szymanik},
      year={2025},
      eprint={2505.14313},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.14313}, 
}
```
