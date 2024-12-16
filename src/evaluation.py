# Original Copyright 2021 OpenAI under MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

import itertools
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Union

import numpy as np
import tqdm
import json
from data import read_problems, stream_jsonl, write_jsonl

# Amazon modification
# import check correctness for all languages
from execution import (
    check_correctness,
    check_correctness_cpp,
    check_correctness_csharp,
    check_correctness_go,
    check_correctness_java,
    check_correctness_javascript,
    check_correctness_kotlin,
    check_correctness_perl,
    check_correctness_php,
    check_correctness_ruby,
    check_correctness_scala,
    check_correctness_swift,
    check_correctness_typescript,
)
from execution_with_test_cases import (
    check_cpp_correctness_with_test_cases,
    check_python_correctness_with_test_cases,
)


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = os.cpu_count() - 1,
    timeout: float = 10.0,
    problem_file: str = "data/CodeStepsEval277-CPP.jsonl",
    result_path: str = "results",
    with_test_cases: bool = False
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl"
    """

    check_correctness_map = {
            "python": check_correctness,
            "java": check_correctness_java,
            "javascript": check_correctness_javascript,
            "typescript": check_correctness_typescript,
            "kotlin": check_correctness_kotlin,
            "ruby": check_correctness_ruby,
            "php": check_correctness_php,
            "cpp": check_correctness_cpp,
            "csharp": check_correctness_csharp,
            "go": check_correctness_go,
            "perl": check_correctness_perl,
            "scala": check_correctness_scala,
            "swift": check_correctness_swift,
        }

    check_correctness_with_test_cases_map = {
        "cpp": check_cpp_correctness_with_test_cases,
        "python": check_python_correctness_with_test_cases,
    }

    problems = read_problems(problem_file)

    # see execution.py for details
    # Check the generated samples against test suites.

    check_correctness_map_to_use = check_correctness_with_test_cases_map if with_test_cases else check_correctness_map
    
    seed = int(time.time() * 1000000) % 1000000
    np.random.seed(seed=seed)  # microsecond

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            language = sample["language"]
            if with_test_cases:
                args = {
                    "problem": problems[task_id], 
                    "completion": completion, 
                    "timeout": timeout, 
                    "completion_id": completion_id[task_id], 
                    "language": language, 
                    "extension": ".cpp", 
                    "compile_command_lambda": lambda x: ["g++", "-std=c++14", f"{os.path.basename(x)}.cpp", "-o", f"{os.path.basename(x)}_cpp"],
                    "compile_timeout": 100,
                    "subprocess_command_lambda": lambda x: [f"./{os.path.basename(x)}_cpp"],
                    "extra_cleanup": lambda x: f"{x}_cpp",
                    "cwd": True
                }
            else:
                args = {
                    "problem": problems[task_id], 
                    "completion": completion, 
                    "timeout": timeout, 
                    "completion_id": completion_id[task_id]
                }
            
            check_correctness_function = check_correctness_map_to_use[language]
            future = executor.submit(check_correctness_function, **args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()  # this is the execution stage
            results[result["task_id"]].append((result["completion_id"], result))

    # common code for all languages
    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            sample["completion_id"] = result[1]["completion_id"]
            sample["time_elapsed"] = result[1]["time_elapsed"]
            sample["compiled"] = result[1]["compiled"]
            if "error_message" in result[1]:
                sample["error_message"] = result[1]["error_message"]
            if "status" in result[1]:
                sample["status"] = result[1]["status"]
            yield sample

    file_name = sample_file.split("/")[-1].split('.jsonl')[0]
    out_file = os.path.join(result_path, file_name + "_results.jsonl")
    out_file_passatk = os.path.join(result_path, file_name + "_passatk.json")

    os.makedirs(result_path, exist_ok=True)
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    with open(out_file_passatk, "w") as f:
        f.write(json.dumps(pass_at_k))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_file", type=str, default="test_data/CodeStepsEval277_CPP_CodeLlama_test_samples.jsonl")
    parser.add_argument("--problem_file", type=str, default="data/CodeStepsEval277-CPP.jsonl")
    parser.add_argument("--k", type=str, default="1,3,5,10,100")
    parser.add_argument("--n_workers", type=int, default=os.cpu_count() - 1)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--result_path", type=str, default="results")
    parser.add_argument("--with_test_cases", action="store_true")
    args = parser.parse_args()

    k_values = [int(k) for k in args.k.split(',')]

    evaluate_functional_correctness(
        sample_file=args.sample_file,
        problem_file=args.problem_file,
        k=k_values,
        n_workers=args.n_workers,
        timeout=args.timeout,
        result_path=args.result_path,
        with_test_cases=args.with_test_cases
    )


if __name__ == "__main__":
    main()