import os

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


for g_idx, graph in enumerate(graphs):
    for p in range(len(patterns)):
        output_name = f"{file_dir}/../auto/{graph}_p{p+1}.cu" 
        commands = f"{file_dir}/../build/bin/final_generator {data_path}/{graph}.g {pattern_sizes[p]} {patterns[p]} > {output_name}"
        print(commands, flush=True)
        os.system(command_prefix + commands)
os.system(f"cd {file_dir}/../auto; bash concat.sh")