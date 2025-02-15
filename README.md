# Language Model Evaluation Harness

![](https://github.com/EleutherAI/lm-evaluation-harness/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/EleutherAI/lm-evaluation-harness/branch/master/graph/badge.svg?token=JSG3O2427J)](https://codecov.io/gh/EleutherAI/lm-evaluation-harness)

## Overview

This project provides a unified framework to test autoregressive language models (GPT-2, GPT-3, GPTNeo, etc) on a large number of different evaluation tasks.

Features:

- 200+ tasks implemented. See the [task-table](./docs/task_table.md) for a complete list.
- Support for GPT-2, GPT-3, GPT-Neo, GPT-NeoX, and GPT-J, with flexible tokenization-agnostic interface.
- Task versioning to ensure reproducibility.

## Install

```bash
pip install lm-eval
```

To install additional multlingual tokenization and text segmenation packages, you must install the package with the `multilingual` extra:

```bash
pip install "lm-eval[multilingual]"
```

## Basic Usage

> **Note**: When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility. This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible. See the [Task Versioning](#task-versioning) section for more info.

To evaluate a model (e.g. GPT-2) on NLP tasks such as SuperGLUE WiC, you can run the following command:


```bash
python main.py \
    --model gpt2 \
    --tasks lambada_openai,hellaswag \
    --device 0
```

This example uses gpt2-117M by default as per HF defaults.

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most importantly, the `gpt2` model can be used to load an arbitrary HuggingFace CausalLM. For example, to run GPTNeo use the following:

```bash
python main.py \
    --model gpt2 \
    --model_args pretrained=EleutherAI/gpt-neo-2.7B \
    --tasks lambada_openai,hellaswag \
    --device 0
```

If you have access to the OpenAI API, you can also evaluate GPT-3:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag
```

And if you want to verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

To evaluate mesh-transformer-jax models that are not available on HF, please invoke eval harness
through [this script](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

### Weights & Biases Reports

To enable [Weights & Biases](https://wandb.ai/site) reporting for your evaluations install the wandb extras by
running `pip install lm-eval[wandb]`.
Add the `--wandb_project=<YOUR_PROJECT_NAME>` to the main command. You can also add `--wandb_entity=<YOUR_ENTITY_NAME>`
if you are using a team account.

```bash
python main.py \
	--model gpt2 \
	--model_args pretrained=EleutherAI/gpt-neo-2.7B \
	--device 0 \
	--tasks lambada,hellaswag
	--wandb_project=<YOUR_PROJECT_NAME>
	--wandb_entity=<YOUR_ENTITY_NAME>
	--wandb_group=<YOUR_RUN_GROUP_NAME>
```

This auto generates a [Weights & Biases report](https://wandb.ai/site/reports) that can be easily shared. Here is
an [example report](https://wandb.ai/parambharat/lm_eval/reports/-2023-01-03-08-36-21-Model-distilgpt2-Evaluation-report--VmlldzozMjU0MTgy):

To compare multiple models across tasks, run the above evaluation with the same `--wandb_group` argument and then run
the `wandb_reporte.py` as follows.

```bash
python wandb_reporter.py \
    --project <YOUR_PROJECT_NAME> \
    --entity <YOUR_ENTITY_NAME> \
    --group <YOUR_RUN_GROUP_NAME>
```

This logs a run with charts that compare the models and generates a report with the logged metrics table.
Here is
an [example report](https://wandb.ai/parambharat/lm_eval/reports/-2023-01-03-09-06-02-Model-comparison-report--VmlldzozMjU0Mzg5)
and the corresponding [run](https://wandb.ai/parambharat/lm_eval/runs/1awvscu4)

*Note: This only works for models that have been evaluated on the same tasks and models that are a subclass of `HFLM`
or `GPT2LM` models.*

💡 **Tip**: You can inspect what the LM inputs look like by running the following command:

```bash
python write_out.py \
    --tasks all_tasks \
    --num_fewshot 5 \
    --num_examples 10 \
    --output_base_path /path/to/output/folder
```

This will write out one text file for each task.

## Implementing new tasks

To implement a new task in the eval harness, see [this guide](./docs/task_guide.md).

## Task Versioning

To help improve reproducibility, all tasks have a `VERSION` field. When run from the command line, this is reported in a column in the table, or in the "version" field in the evaluator return dict. The purpose of the version is so that if the task definition changes (i.e to fix a bug), then we can know exactly which metrics were computed using the old buggy implementation to avoid unfair comparisons. To enforce this, there are unit tests that make sure the behavior of all tests remains the same as when they were first implemented. Task versions start at 0, and each time a breaking change is made, the version is incremented by one.

When reporting eval harness results, please also report the version of each task. This can be done either with a separate column in the table, or by reporting the task name with the version appended as such: taskname-v0.

## Test Set Decontamination

For details on text decontamination, see the [decontamination guide](./docs/decontamination.md).

Note that the directory provided to the `--decontamination_ngrams_path` argument should contain the ngram files and info.json. See the above guide for ngram generation for the pile, this could be adapted for other training sets.

```bash
python main.py \
    --model gpt2 \
    --tasks sciq \
    --decontamination_ngrams_path path/containing/training/set/ngrams \
    --device 0
```

## Cite as

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
