import os

data_path = "/home/cqq/data"
file_dir = os.path.dirname(__file__)

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

def pattern_matching_reproduce(is_gpu : bool = False, is_generated_code : bool = False, log_path : str = "/../reproduce_log/pattern_matching"):
    log_path = file_dir + log_path
    if is_gpu:
        log_path += "_gpu"
    else:
        log_path += "_cpu"

    if is_generated_code:
        log_path += "_generated"

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if is_gpu:
        execute_name = "gpu_graph"
    else:
        execute_name = "pm_test"
    
    for graph in graphs:
        for p in range(len(patterns)):
            if(is_generated_code):
                execute_name = f"{graph}_p{p + 1}"
            log_name = f"{log_path}/{graph}_p{p + 1}.log"
            command = f"{file_dir}/../build/bin/{execute_name} {data_path}/{graph}.g {pattern_sizes[p]} {patterns[p]} 1>{log_name}"
            print(command)
            result = os.system(command)
            if result != 0:
                return

# def motif_counting(is_gpu : bool = False, log_path : str = "/motif_counting"):
#     log_path = file_dir + log_path
#     if is_gpu:
#         log_path += "_gpu"
#     else:
#         log_path += "_cpu"

#     if not os.path.exists(log_path):
#         os.makedirs(log_path)

#     sizes = [4, 5]
#     graphs = ["patents", "mico", "livejournal", "orkut"]
#     for size in sizes:
#         for graph in graphs:
#             log_name = log_path + "/" + execute_name + "_" + graph + "_" + "p" + str(p + 1) + ".log"
#             command = "srun -p V100 --exclusive " + file_dir + "/../build/bin/" + execute_name + " " + data_path + "/" + graph + ".g " + str(pattern_sizes[p]) + " " + str(patterns[p]) + " > " + log_name
#             print(command)
#             result = os.system(command)
#             if result != 0:
#                 return



if __name__ == "__main__":
    # pattern_matching_reproduce(False)
    pattern_matching_reproduce(True, True)
