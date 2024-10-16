import os
import signal
import subprocess
import platform
from typing import *
import numpy as np
import re

SUCCESS = 0
RUN_ERROR = 1
TIME_OUT = 2
UNKNOWN_ERR = 3


def run_cmd(cmd_string, timeout=5):
    p = subprocess.Popen(cmd_string, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True, close_fds=True,
                         start_new_session=True)

    format = 'utf-8'
    if platform.system() == "Windows":
        format = 'gbk'

    try:
        (msg, errs) = p.communicate(timeout=timeout)
        ret_code = p.poll()
        if ret_code:
            code = RUN_ERROR
        else:
            code = SUCCESS
    except subprocess.TimeoutExpired:
        p.kill()
        p.terminate()
        os.killpg(p.pid, signal.SIGTERM)
        code = TIME_OUT
    except Exception as e:
        code = UNKNOWN_ERR

    return code


def pass_at_k(num_samples: int, num_correct: int, k: List[int]):

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return [estimator(num_samples, num_correct, i) for i in k]


def avg_pass_at_k(all_pass_at_k: List[Dict]) -> Dict:
    final_dict = {f"avg_{key}": [d[key]for d in all_pass_at_k]
                  for key in all_pass_at_k[0] if key != "pid"}
    # print(final_dict)
    res = {key: np.mean(value_list) for key, value_list in final_dict.items()}
    return res


def avg_test_case_pass_rate(all_res: List[Dict]) -> float:
    pass_rate_list = [d["pass_rate"] for d in all_res]
    return np.mean(pass_rate_list)
