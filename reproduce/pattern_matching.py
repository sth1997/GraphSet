import os
import math
import matplotlib.pyplot as plt

from settings import *
from utils import *

PATTERNS = []
PATTERNS.append("0111010011100011100001100")  # p1
PATTERNS.append("011011101110110101011000110000101000")  # p2
PATTERNS.append("011111101000110111101010101101101010")  # p3
PATTERNS.append("011110101101110000110000100001010010")  # p4
PATTERNS.append("0111111101111111011101110100111100011100001100000")  # p5
PATTERNS.append("0111111101111111011001110100111100011000001100000")  # p6
PATTERN_SIZES = [5, 6, 6, 6, 7, 7]

GRAPHS = ["mico", "patents", "orkut", "livejournal"]


def pattern_matching(
    is_gpu: bool = False,
    is_generated_code: bool = False,
    bin_path: str = "../build/bin",
    log_path: str = "../reproduce_log/pattern_matching",
    result_path: str = "../reproduce_result",
):
    log_path = log_path + ("_gpu" if is_gpu else "_cpu")
    if is_generated_code:
        log_path += "_generated"

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    result_path = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = (
        f"{result_path}/pattern_matching" + ("_gpu" if is_gpu else "_cpu") + ".csv"
    )

    data = []
    data.append(["Pattern"] + GRAPHS)

    execute_name = "gpu_graph" if is_gpu else "pm_test"

    for p in range(len(PATTERNS)):
        tmp = [f"p{p+1}"]
        for graph in GRAPHS:
            if is_generated_code:
                execute_name = f"{graph}_p{p + 1}"
            log_name = f"{log_path}/{graph}_p{p + 1}.log"
            command = f"{bin_path}/{execute_name} {DATA_PATH}/{graph}.g {PATTERN_SIZES[p]} {PATTERNS[p]} 1>{log_name}"
            print(command, flush=True)
            result = os.system(COMMAND_PREFIX + command)
            tmp.append(read_time_cost(log_name))
            if result != 0:
                return 1
        data.append(tmp)
    write_table(data, result_path)
    return 0

def draw_pattern_matching(is_gpu: bool = False, result_path: str = "../reproduce_result"):
    with open(f"{result_path}/pattern_matching" + ("_gpu" if is_gpu else "_cpu") + ".csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        # first line is header: Pattern, [graph_names]
        header = next(reader)
        graph_names = header[1:]
        # other lines are data
        data = []
        for row in reader:
            data.append(row)
        # transpose data
        data = list(map(list, zip(*data)))
        # first column is pattern names
        pattern_names = data[0]
        # other columns are time costs
        time_costs = data[1:]
        # draw a figure for every graph, and make them in one table horizontally
        fig_size = (3 * len(graph_names), 5)
        fig, axes = plt.subplots(1, len(graph_names), figsize=fig_size)
        for i in range(len(graph_names)):
            data_y = list(map(float, time_costs[i]))
            print(data_y)
            axes[i].set_title(graph_names[i])
            axes[i].set_xlabel("Pattern")
            axes[i].set_ylabel("Time Cost (s)")
            axes[i].bar(pattern_names, data_y)
            # set y axis to log scale
            axes[i].set_yscale("log")
            # set y axis start to the minimum 10 powers smaller than the minimum time cost
            axes[i].set_ylim(bottom=10 ** (int(math.log10(min(data_y))) - 1))
        plt.tight_layout()
        # add a name 
        plt.suptitle("Pattern Matching" + (" (GPU)" if is_gpu else " (CPU)"))
        plt.savefig(f"{result_path}/pattern_matching" + ("_gpu" if is_gpu else "_cpu") + ".png")
    


if __name__ == "__main__":
    print("Start reproducing pattern matching...(Total Estimated Time Cost: 7 hours)")
    print("Reproducing pattern matching on GPU...")
    assert pattern_matching(is_gpu=True, is_generated_code=True) == 0
    draw_pattern_matching(is_gpu=True)
    print("Reproducing pattern matching on CPU...")
    assert pattern_matching(is_gpu=False) == 0
    draw_pattern_matching(is_gpu=False)
    print("Finish reproducing pattern matching!")
