import json
import os
from typing import *
from utils import *
import platform
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

resources_dir = 'resources'
res_dir = 'result'

pids: List[str] = []
# [ [{"input": ,"output": },{"input": ,"output": }],[{},{},..], ...]
all_codes: List[List[str]] = []
all_difficulity: List[str] = []
all_test_cases: List[List[Dict[str, str]]] = []
all_nl: List[str] = []
all_input_format: List[str] = []
all_output_format: List[str] = []
all_step: List[str] = []

overlen_problems: List[str] = []

CUDA = True

def init(input_file: str):
    with open(f"{resources_dir}/{input_file}", 'r', encoding='utf-8') as f:
        data = json.load(f)
    for problem in data:
        pid = problem['pid']
        pids.append(pid)

        test_cases = problem['test_case']
        all_test_cases.append(test_cases)

        all_difficulity.append(problem["difficulty"])
        all_nl.append(problem["nl"])
        all_input_format.append(problem["input_format"])
        all_output_format.append(problem["output_format"])
        all_step.append("\n".join(problem["step"]))


def compile(code: str):
    cur_name = threading.current_thread().name
    with open(rf'temp/{cur_name}.cpp', 'w', encoding='utf-8') as f:
        f.write(code)
    if platform.uname().system == "Windows":
        res = os.system(
            r"g++ -w temp/one_code.cpp -o temp/one_code.exe 2>temp/compile_err.txt")
    else:  # Linux
        res = os.system(
            rf"g++ -w ./temp/{cur_name}.cpp -o ./temp/{cur_name} 2>temp/{cur_name}_compile_err.txt")
    return res


def run_code():
    cur_name = threading.current_thread().name
    if platform.uname().system == "Windows":
        cmd = r"temp\one_code.exe < temp\input.txt > temp\output.txt"
    else:
        cmd = rf"./temp/{cur_name} < ./temp/{cur_name}_in.txt > ./temp/{cur_name}_out.txt"
    res = run_cmd(cmd)
    return res


def check_test_cases(test_cases: List[Dict[str, str]]) -> Tuple[List, List]:
    cur_name = threading.current_thread().name
    cases_msg, cases_res = [], []
    in_file = rf"./temp/{cur_name}_in.txt"
    if not os.path.exists(in_file):
        with open(in_file, 'w', encoding='utf-8') as f:
            pass

    with open(in_file, 'r+', encoding='utf-8') as f:
        for case in test_cases:
            f.seek(0)
            f.truncate()
            f.write(case['input'])
            f.flush()
            run_res = run_code()
            if run_res == RUN_ERROR or run_res == UNKNOWN_ERR:
                cases_msg.append("run error or unknown error")
                cases_res.append(False)
            elif run_res == TIME_OUT:
                cases_msg.append("timeout")
                cases_res.append(False)
            else:
                with open(rf"./temp/{cur_name}_out.txt", 'r', encoding='utf-8', errors='ignore') as outf:
                    out = outf.read()
                    if out.rstrip() == case['output'].rstrip():
                        cases_msg.append("pass")
                        cases_res.append(True)
                    else:
                        cases_msg.append("fail")
                        cases_res.append(False)
    return cases_msg, cases_res


def generate_codes(index: int, prompt: str, n: int, max_to_generate: int = 512):
    codes = []
    already_overlen = False
    for i in range(n):
        print(f"problem{index}-{pids[index]}-generate code {i}\n", end='')
        res_final, overlen = generate_one_code(
            model, tokenizer, prompt=prompt, max_to_generate=max_to_generate, top_p=0.95, temperature=0.2)
        if not already_overlen and overlen:
            overlen_problems.append(pids[index])
            already_overlen = True

        codes.append(res_final)
    return codes


def check_one_problem(index: int, n, k: List[int]) -> Tuple[List, Dict, Dict]:
    pid = pids[index]
    difficulity = all_difficulity[index]
    test_cases: List = all_test_cases[index]
    nl = all_nl[index]
    input_format = all_input_format[index]
    output_format = all_output_format[index]
    steps = all_step[index]
    # with step
    prompt = f'Problem description:\n{nl}\nSteps:\n{steps}\nInput format:\n{input_format}\n' \
        f'Output format:\n{output_format}\nExamples:\n' \
        f'Input>>\n{test_cases[0]["input"]}\nOutput>>\n{test_cases[0]["output"]}\nAnswer:\n'
    codes = generate_codes(index=index, prompt=prompt, n=n)

    one_pro_res: List[Dict] = []
    num_hard_correct, num_soft_correct = 0, 0
    for i, code in enumerate(codes):
        print(f"problem{index}-{pid}-code{i}\n", end='')
        compile_res = compile(code)
        compile_err_msg = ""
        if compile_res != 0:
            cases_msg = ["compile error"]*len(test_cases)
            cases_res = [False]*len(test_cases)
            with open(rf"temp/{threading.current_thread().name}_compile_err.txt", 'r', encoding='utf-8') as f:
                compile_err_msg = f.read()
        else:
            cases_msg, cases_res = check_test_cases(test_cases)
        case_pass_rate = cases_res.count(True)/len(cases_res)
        one_pro_res.append(
            {"pid": pid, "difficulity": difficulity, "code": code, "result": cases_msg,
             "passed": cases_res, "pass_rate": case_pass_rate, "compile_err_msg": compile_err_msg})
        num_hard_correct += (1 if all(cases_res) else 0)
        num_soft_correct += (1 if any(cases_res) else 0)

    hard_pass_values = pass_at_k(n, num_hard_correct, k)
    soft_pass_values = pass_at_k(n, num_soft_correct, k)
    hard_pass_at_k: Dict = {f"hard_pass@{k}": v for k,
                            v in zip(k, hard_pass_values)}
    soft_pass_at_k: Dict = {f"soft_pass@{k}": v for k,
                            v in zip(k, soft_pass_values)}
    return one_pro_res, hard_pass_at_k, soft_pass_at_k, index


def get_input_files() -> List:
    input_file_list = [path for path in os.listdir(rf'your test data dir/')]
    return input_file_list


def get_difficulity_ids(difficulity: str):
    res = []
    for i, difficul in enumerate(all_difficulity):
        if difficul.strip() == difficulity.strip():
            res.append(i)
    return res


def evaluate_functional_correctness(num_samples: int, k: List[int], n_workers: int):
    input_file_list = get_input_files()
    print(input_file_list)
    print(len(input_file_list))
    for input_file in input_file_list:
        print(f"current file: {input_file}")
        init(input_file)
    print(f"test set has {len(pids)} samples")
    all_res, all_hard_pass_at_k, all_soft_pass_at_k = [], [], []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completed_index = []
        for index in range(len(pids)):
            args = (index, num_samples, k)
            future = executor.submit(check_one_problem, *args)
            futures.append(future)
            completed_index.append(index)

        assert len(completed_index) == len(
            pids), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            one_pro_res, hard_pass_at_k, soft_pass_at_k, p_index = future.result()
            all_res.extend(one_pro_res)
            all_hard_pass_at_k.append({"pid": pids[p_index], **hard_pass_at_k})
            all_soft_pass_at_k.append({"pid": pids[p_index], **soft_pass_at_k})
            with open(f'{res_dir}/problem_result.json', 'w', encoding='utf-8') as f:
                json.dump(all_res, f, ensure_ascii=False, indent=2)
            with open(f'{res_dir}/hard_pass_result.json', 'w', encoding='utf-8') as f:
                json.dump(all_hard_pass_at_k, f, ensure_ascii=False, indent=2)
            with open(f'{res_dir}/soft_pass_result.json', 'w', encoding='utf-8') as f:
                json.dump(all_soft_pass_at_k, f, ensure_ascii=False, indent=2)
    return all_res, all_hard_pass_at_k, all_soft_pass_at_k


if __name__ == '__main__':
    all_res, all_hard_pass_at_k, all_soft_pass_at_k = evaluate_functional_correctness(
        num_samples=20, k=[1, 5, 10], n_workers=3)
    avg_hard_pass_at_k = avg_pass_at_k(all_hard_pass_at_k)
    avg_soft_pass_at_k = avg_pass_at_k(all_soft_pass_at_k)
    avg_pass_rate = avg_test_case_pass_rate(all_res)
    with open(f'{res_dir}/problem_result.json', 'w', encoding='utf-8') as f:
        json.dump(all_res, f, ensure_ascii=False, indent=2)
    with open(f'{res_dir}/hard_pass_result.json', 'w', encoding='utf-8') as f:
        json.dump(all_hard_pass_at_k, f, ensure_ascii=False, indent=2)
    with open(f'{res_dir}/soft_pass_result.json', 'w', encoding='utf-8') as f:
        json.dump(all_soft_pass_at_k, f, ensure_ascii=False, indent=2)
    print(avg_hard_pass_at_k)
    print(avg_soft_pass_at_k)
    print(f"avg test case pass rate: {avg_pass_rate}")
    print(
        f"max_length >= 2048 number: {len(overlen_problems)}. The pids are: \n{overlen_problems}")
