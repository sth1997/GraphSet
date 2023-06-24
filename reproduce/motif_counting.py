import os

from utils import *
from settings import *

GRAPHS = ["mico", "patents", "orkut", "livejournal"]

motif_sizes = [4]


def motif_counting(
    is_gpu: bool = False,
    bin_path: str = "../build/bin",
    log_path: str = "../reproduce_log/motif_counting",
    result_path: str = "../reproduce_result",
):
    log_path = log_path + ("_gpu" if is_gpu else "_cpu")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    result_path = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = (
        f"{result_path}/motif_counting" + ("_gpu" if is_gpu else "_cpu") + ".csv"
    )

    data = []
    data.append(["Size"] + GRAPHS)

    execute_name = "gpu_mc" if is_gpu else "motif_counting_test"

    for size in motif_sizes:
        tmp = [size]
        for graph in GRAPHS:
            log_name = f"{log_path}/{graph}_mc{size}.log"
            command = f"{bin_path}/{execute_name} {DATA_PATH}/{graph}.g {size} 1>{log_name}"
            print(command, flush=True)
            result = os.system(COMMAND_PREFIX + command)
            tmp.append(read_time_cost(log_name))
            if result != 0:
                return 1
        data.append(tmp)
    write_table(data, result_path)
    return 0

if __name__ == "__main__":
    print("Reproducing motif counting results. (Time: 6 hours)")
    print("Reproducing motif counting results on CPU...")
    assert motif_counting(is_gpu=False) == 0
    print("Reproducing motif counting results on GPU...")
    assert motif_counting(is_gpu=True) == 0
    print("Reproducing motif counting results done.")
