import os
import math
import matplotlib.pyplot as plt

from settings import *
from utils import *

PATTERNS = []
PATTERNS.append("0111010011100011100001100")  # p3
PATTERNS.append("011011101110110101011000110000101000")  # p4
PATTERNS.append("011111101000110111101010101101101010")  # p5
PATTERNS.append("011110101101110000110000100001010010")  # p6
PATTERNS.append("0111111101111111011101110100111100011100001100000")  # p7
PATTERNS.append("0111111101111111011001110100111100011000001100000")  # p8
PATTERNS.append("011101110") # p1
PATTERNS.append("0111101111011110") # p2

GRAPH = "livejournal"


GPU_CARDS = [1, 2, 4, 8]

def pattern_matching_scalability(
    is_gpu: bool = False,
    bin_path: str = "../build/bin",
    log_path: str = "../reproduce_log/pattern_matching_scalability",
    result_path: str = "../reproduce_result",
):
    if not is_gpu:
        return NotImplementedError
    log_path = log_path + ("_gpu" if is_gpu else "_cpu")

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    result_path = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = (
        f"{result_path}/pattern_matching_scalability" + ("_gpu" if is_gpu else "_cpu") + ".csv"
    )

    data = []
    data.append(["Cards"] + GPU_CARDS)

    execute_name = "gpu_graph_static_task" if is_gpu else "pm_test"

    for p in range(len(PATTERNS)):
        tmp = [f"p{p+1}"]
        for n in range(len(GPU_CARDS)):
            log_name = f"{log_path}/{GRAPH}_p{p + 1}_n{GPU_CARDS[n]}.log"
            command = f"{bin_path}/{execute_name} {DATA_PATH}/{GRAPH}.g 0 {PATTERNS[p]} {GPU_CARDS[n]} 1>{log_name}"
            print(command, flush=True)
            result = os.system(MULTI_CARD_COMMAND_PREFIX + command)
            tmp.append(read_time_cost(log_name))
            if result != 0:
                return 1
        data.append(tmp)
    write_table(data, result_path)
    return 0

def draw_pattern_matching_scalability(is_gpu: bool = False, result_path: str = "../reproduce_result"):
    if not is_gpu:
        return NotImplementedError
    with open(f"{result_path}/pattern_matching_scalability" + ("_gpu" if is_gpu else "_cpu") + ".csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        # first line is header: Pattern, [graph_names]
        header = next(reader)
        # other lines are data
        data = []
        for row in reader:
            data.append(row)

        # first column is pattern names
        pattern_names = [data[i][0] for i in range(0, len(data))]

        # other columns are time costs
        time_costs = [data[i][1:] for i in range(0, len(data))]
        print(time_costs)
        # draw a line graph consisting of the acceleration of each pattern
        for i in range(len(PATTERNS)):
            # convert string to float
            time_costs[i] = list(map(float, time_costs[i]))
            # calculate acceleration
            time_costs[i] = [time_costs[i][0] / time_costs[i][j] for j in range(len(time_costs[i]))]
        print(time_costs)
        # draw
        plt.figure(figsize=(8, 8))
        plt.title("Pattern Matching Scalability" + (" (GPU)" if is_gpu else " (CPU)"))
        plt.xlabel("Cards")
        plt.ylabel("Acceleration")
        for i in range(len(pattern_names)):
            plt.plot(GPU_CARDS, time_costs[i], label=pattern_names[i], marker='o')
        # set x axis to log scale base 2
        plt.loglog(base=2)
        plt.legend()
        plt.savefig(f"{result_path}/pattern_matching_scalability" + ("_gpu" if is_gpu else "_cpu") + ".png")


    


if __name__ == "__main__":
    print("Reproducing pattern matching scalability on GPU... (Expected time cost: TODO) ")
    # assert pattern_matching_scalability(is_gpu=True) == 0
    draw_pattern_matching_scalability(is_gpu=True)
