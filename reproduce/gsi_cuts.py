import os
import math
import matplotlib.pyplot as plt

from settings import *
from utils import *


PATTERNS = [
"0111100111110011110111110",
"0111010111110101110111110",
"0110110111110111110011110",
"0111010111110111100101110",
"0011110110110011110111110",
"0111010111010111110101110",
"0101110111110011110011110",
"0111110111100101110110110",
"0111110111100110100111110",
"0001110110110110110111110",
"0111110100100111110111010",
"001101100111110101111011111101111010",
"010111100111110110111011111100111100",
"011011101100110111111011101101011110",
"011111001001010111111011111001011110",
"010110001101110111111011111001110110",
"010011001111100101111011111101101110",
"010111101111100111111001011100111010",
"011111100101110011111011101101011100",
"011111101011100000111011111001111100",
"010011101111100111111010110101100110",
"000111101011110111001010111101111010",
"0101110100111101010111110111011101111011011110010",
"0110101101111011011111110011111101111111000110000",
"0011111000111010011011110111101101101111010011110",
"0101110100101111001110110011011101111111001110110",
"0110011101001111011111110110011101101111010010110",
"0110111100111111011110000101100101111101001111110",
"0111011101111111001111010110011101011100010011110",
"0101111101111100000111000111011101101111000111110",
"0111011100111111011111000111101100101001011110100",
"0110010100111101011111000011111101101110011111010",
"0111111101111011000111010111011100010100010111110"
]

GRAPHS = ["Enron", "gowalla", "roadNetCa", "roadNetPa", "roadNetTx", "wikiTalk"]


def pattern_matching_gsi_cuts(
    is_gpu: bool = False,
    bin_path: str = "../build/bin",
    log_path: str = "../reproduce_log/pattern_matching_gsi_cuts",
    result_path: str = "../reproduce_result",
):
    log_path = log_path + ("_gpu" if is_gpu else "_cpu")

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    result_path = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = (
        f"{result_path}/pattern_matching_gsi_cuts" + ("_gpu" if is_gpu else "_cpu") + ".csv"
    )

    data = []
    data.append(["Pattern"] + GRAPHS)

    execute_name = "gpu_graph" if is_gpu else "pm_test"

    for p in range(len(PATTERNS)):
        tmp = [f"p{p+1}"]
        for graph in GRAPHS:

            log_name = f"{log_path}/{graph}_p{p + 1}.log"
            command = f"{bin_path}/{execute_name} {DATA_PATH}/{graph}.g {int(math.sqrt(len(PATTERNS[p])))} {PATTERNS[p]} 0 1>{log_name}"
            print(command, flush=True)
            result = os.system(COMMAND_PREFIX + command)
            tmp.append(read_time_cost(log_name))
            if result != 0:
                return 1
        data.append(tmp)
    write_table(data, result_path)
    return 0


if __name__ == "__main__":
    pattern_matching_gsi_cuts(is_gpu=True)