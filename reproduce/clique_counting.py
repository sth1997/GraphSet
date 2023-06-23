import os

from utils import *
from settings import *

clique_graphs = ["patents", "mico", "livejournal", "orkut", "friendster"]
clique_sizes = [4, 5]


def clique_counting(
    is_gpu: bool = False,
    bin_path: str = "../build/bin",
    log_path: str = "../reproduce_log/clique_counting",
    result_path: str = "../reproduce_result",
):
    log_path = log_path + ("_gpu" if is_gpu else "_cpu")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    result_path = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = (
        f"{result_path}/clique_counting" + ("_gpu" if is_gpu else "_cpu") + ".csv"
    )

    execute_name = "gpu_kclique" if is_gpu else "clique_test"

    data = []
    data.append(["Size"] + clique_graphs)

    for size in clique_sizes:
        tmp = [size]
        for graph in clique_graphs:
            log_name = f"{log_path}/{graph}_cc{size}.log"
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
    print("Reproducing clique counting results. (Time: 20 minutes)")
    print("Reproducing clique counting results on CPU...")
    assert clique_counting(is_gpu=False) == 0
    print("Reproducing clique counting results on GPU...")
    assert clique_counting(is_gpu=True) == 0
    print("Reproducing clique counting results done.")
