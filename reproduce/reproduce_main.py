import os
import sys
import re
import csv
import matplotlib.pyplot as plt

data_path = "/home/cqq/data"
file_dir = os.path.dirname(__file__)

command_prefix = "srun -p V100 --gres=gpu:v132p:1 --exclusive "

patterns = []
patterns.append("0111010011100011100001100") # p1
patterns.append("011011101110110101011000110000101000") # p2
patterns.append("011111101000110111101010101101101010") # p3
patterns.append("011110101101110000110000100001010010") # p4
patterns.append("0111111101111111011101110100111100011100001100000") # p5
patterns.append("0111111101111111011001110100111100011000001100000") # p6
pattern_sizes = [5, 6, 6, 6, 7, 7]

graphs = []
graphs.append("patents")
graphs.append("mico")
graphs.append("livejournal")
graphs.append("orkut")


motif_sizes = [4]

clique_sizes = [4, 5]

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


def read_time_cost(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        pattern = r"Counting time cost: (\d+\.\d+) s"
        match = re.search(pattern, content)

        if match:
            time_cost = float(match.group(1))
            return time_cost
        else:
            print(f"Time not found in file {file_path}.")
            return None

def write_table(table, csv_file):
    # Write table and header into a CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table)


def pattern_matching(is_gpu : bool = False, is_generated_code : bool = False, log_path : str = "/../reproduce_log/pattern_matching", result_path : str = "/../reproduce_result"):
    log_path = file_dir + log_path + ("_gpu" if is_gpu else "_cpu") 
    if is_generated_code:
        log_path += "_generated"

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    result_path = file_dir + result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = f"{result_path}/pattern_matching" + ("_gpu" if is_gpu else "_cpu") + ".csv"

    data = []
    data.append(["Pattern"] + graphs)

    execute_name = "gpu_graph" if is_gpu else "pm_test"
    
    for p in range(len(patterns)):
        tmp = [f"p{p+1}"]
        for graph in graphs:
            if(is_generated_code):
                execute_name = f"{graph}_p{p + 1}"
            log_name = f"{log_path}/{graph}_p{p + 1}.log"
            command = f"{file_dir}/../build/bin/{execute_name} {data_path}/{graph}.g {pattern_sizes[p]} {patterns[p]} 1>{log_name}"
            print(command, flush=True)
            result = os.system(command_prefix + command)
            tmp.append(read_time_cost(log_name))
            if result != 0:
                return 1
        data.append(tmp)
    write_table(data, result_path)
    return 0


def motif_counting(is_gpu : bool = False, log_path : str = "/../reproduce_log/motif_counting", result_path : str = "/../reproduce_result"):
    log_path = file_dir + log_path + ("_gpu" if is_gpu else "_cpu") 
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    result_path = file_dir + result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = f"{result_path}/motif_counting" + ("_gpu" if is_gpu else "_cpu") + ".csv"
    
    data = []
    data.append(["Size"] + graphs)

    execute_name = "gpu_mc" if is_gpu else "motif_counting_test"

    for size in motif_sizes:
        tmp = [size]
        for graph in graphs:
            log_name = f"{log_path}/{graph}_mc{size}.log"
            command = f"{file_dir}/../build/bin/{execute_name} {data_path}/{graph}.g {size} 1>{log_name}"
            print(command, flush=True)
            result = os.system(command_prefix + command)
            tmp.append(read_time_cost(log_name))
            if result != 0:
                return 1
        data.append(tmp)
    write_table(data, result_path)
    return 0


def clique_counting(is_gpu : bool = False, log_path : str = "/../reproduce_log/clique_counting", result_path : str = "/../reproduce_result"):
    log_path = file_dir + log_path + ("_gpu" if is_gpu else "_cpu") 
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    result_path = file_dir + result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = f"{result_path}/clique_counting" + ("_gpu" if is_gpu else "_cpu") + ".csv"

    execute_name = "gpu_kclique" if is_gpu else "clique_test"

    data = []
    data.append(["Size"] + graphs)

    for size in clique_sizes:
        tmp = [size]
        for graph in graphs:
            log_name = f"{log_path}/{graph}_cc{size}.log"
            command = f"{file_dir}/../build/bin/{execute_name} {data_path}/{graph}.g {size} 1>{log_name}"
            print(command, flush=True)
            result = os.system(command_prefix + command)
            tmp.append(read_time_cost(log_name))
            if result != 0:
                return 1
        data.append(tmp)
    write_table(data, result_path)    
    return 0

def frequent_subgraph_mining(is_gpu : bool = False, log_path : str = "/../reproduce_log/frequent_subgraph_mining", result_path : str = "/../reproduce_result"):
    log_path = file_dir + log_path + ("_gpu" if is_gpu else "_cpu") 
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    result_path = file_dir + result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = f"{result_path}/frequent_subgraph_mining" + ("_gpu" if is_gpu else "_cpu") + ".csv"

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

    tmp_data = [ [size_support] + [0] * (len(data[0]) - 1) for size_support in size_support_list ]
    
    for graph_index in range(len(fsm_graphs)):
        graph = fsm_graphs[graph_index]
        for fsm_data in fsm_pairs[graph_index]:
            size = fsm_data[0]
            for support in fsm_data[1]:
                tmp_execute_name = execute_name 
                if graph == "mico" and size == 3:
                    tmp_execute_name = "gpu_new_fsm" if is_gpu else "fsm_vertex_test"
                log_name = f"{log_path}/{graph}_fsm{size}+{support}.log"
                command = f"{file_dir}/../build/bin/{tmp_execute_name} {data_path}/{graph}.adj {size} {support}  1>{log_name}"
                print(command, flush=True)
                result = os.system(command_prefix + command)
                tmp_data[size_support_list.index(f"{size}-{support}")][data[0].index(graph)] = read_time_cost(log_name)
                if result != 0:
                    return 1
    data += tmp_data
    write_table(data, result_path)    
    return 0


if __name__ == "__main__":
    assert motif_counting(False) == 0
    assert motif_counting(True) == 0
    assert clique_counting(False) == 0
    assert clique_counting(True) == 0
    assert frequent_subgraph_mining(False) == 0
    assert frequent_subgraph_mining(True) == 0
    assert pattern_matching(True, True) == 0
    assert pattern_matching(False) == 0
    pass
