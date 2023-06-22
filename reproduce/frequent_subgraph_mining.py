import os

from settings import *
from utils import *

fsm_graphs = ['mico', 'patents', 'youtube']
fsm_pairs = [ 
    [ # mico
        (2, [100, 300, 500]),
        (3, [10000, 13000, 15000]),
    ],
    [ # patents
        (2, [5000, 10000, 15000]),
        (3, [10000, 15000, 20000])    
    ],
    [ # youtube
        (2, [1000, 3000, 5000]),
        (3, [1000, 3000, 5000])
    ]
]

def frequent_subgraph_mining(
    is_gpu: bool = False,
    bin_path: str = "../build/bin",
    log_path: str = "../reproduce_log/frequent_subgraph_mining",
    result_path: str = "../reproduce_result",
):
    log_path = log_path + ("_gpu" if is_gpu else "_cpu")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    result_path = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = (
        f"{result_path}/frequent_subgraph_mining"
        + ("_gpu" if is_gpu else "_cpu")
        + ".csv"
    )

    execute_name = "gpu_fsm" if is_gpu else "fsm_test"

    data = []
    data.append(["size-support"] + fsm_graphs)

    size_support_list = []

    for graph_index in range(len(fsm_graphs)):
        graph = fsm_graphs[graph_index]
        for fsm_data in fsm_pairs[graph_index]:
            size = fsm_data[0]
            for support in fsm_data[1]:
                size_support_list.append(f"{size}-{support}")

    size_support_list = list(set(size_support_list))

    size_support_list.sort()

    tmp_data = [
        [size_support] + [0] * (len(data[0]) - 1) for size_support in size_support_list
    ]

    for graph_index in range(len(fsm_graphs)):
        graph = fsm_graphs[graph_index]
        for fsm_data in fsm_pairs[graph_index]:
            size = fsm_data[0]
            for support in fsm_data[1]:
                tmp_execute_name = execute_name
                if graph == "mico" and size == 3:
                    tmp_execute_name = "gpu_new_fsm" if is_gpu else "fsm_vertex_test"
                log_name = f"{log_path}/{graph}_fsm{size}+{support}.log"
                command = f"{bin_path}/{tmp_execute_name} {DATA_PATH}/{graph}.adj {size} {support}  1>{log_name}"
                print(command, flush=True)
                result = os.system(COMMAND_PREFIX + command)
                tmp_data[size_support_list.index(f"{size}-{support}")][
                    data[0].index(graph)
                ] = read_time_cost(log_name)
                if result != 0:
                    return 1
    data += tmp_data
    write_table(data, result_path)
    return 0


if __name__ == "__main__":
    assert frequent_subgraph_mining(is_gpu=False) == 0
    assert frequent_subgraph_mining(is_gpu=True) == 0