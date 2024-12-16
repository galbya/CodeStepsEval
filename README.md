# Beyond Code: Evaluate Thought Steps for Complex Code Generation

# Overview

In the paper, we present **CodeStepsEval**  for **Competition-level** Code Generation, a manual curated benchmark to evaluate large pre-trained language models for both thoutht generation and code generation. CodeStepsEval contains 10,479 competition-level problems (9476 for training, 1000 for testing) and multiple santilized solution in cpp programming  language and fine-grained thought steps for each problem. We take 277 problems from the CodeStepsEval test set and organize them in HumanEval format for more convenient evaluation.

## Project

This project contains the basic components, here is an overview:

```shell
|-- code generation.py # generate code with vLLM inference library 
|-- src
    |-- data.py # read problems and data process utils
    |-- execution.py # perform execution with the given code solutions and test cases in CodeStepsEval
    |-- execution_with_test_cases.py # perform execution in more fined-grained way with given test cases (eg. we can evaluate the correctness of extra provided test cases with canonical solution, or get the fine-grained error message for code execution)
    |-- evaluation.py # calculate pass@k results
|-- data # contains the CodeStepsEval
|-- generated_data # contains the output code by code LLMs
```

# Quickstart

## Prepare Environment
First, you should set up a python environment.  we strongly recommend you to use `conda` to manage the python environment. You could use following commands to create an environment `venv` and activate it.

```bash
$ conda create -n venv python=3.11
$ conda activate venv
$ pip install -r requirements.txt
```

## How to use

```bash
python evaluation.py --sample_file test_data/CodeStepsEval277_CPP_CodeQwen-7b-hf_temp0.8_topp0.95_num100_max800_test_samples.jsonl --problem_file data/CodeStepsEval277-CPP.jsonl --result_path resuts --with_test_cases
```

# Citation

If our work is useful for you, please consider citing our paper:

```bibtex
@inproceedings{cao2024beyond,
  title={Beyond Code: Evaluate Thought Steps for Complex Code Generation},
  author={Cao, Liuwen and Cai, Yi and Wang, Jiexin and He, Hongkui and Huang, Hailin},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages={2296--2306},
  year={2024}
}
```