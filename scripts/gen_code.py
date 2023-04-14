import os

patterns = []
patterns.append("0111010011100011100001100")
patterns.append("011011101110110101011000110000101000")
patterns.append("011111101000110111101010101101101010")
patterns.append("011110101101110000110000100001010010")
patterns.append("0111111101111111011101110100111100011100001100000")
patterns.append("0111111101111111011001110100111100011000001100000")

pattern_sizes = [5, 6, 6, 6, 7, 7]
graphs = []
graphs.append("patents")
graphs.append("mico")
graphs.append("livejournal")
graphs.append("orkut")


for g_idx, graph in enumerate(graphs):
    for p in range(len(patterns)):
        output_name = f"../auto/{graph}_p{p+1}.cu" 
        commands = f"./bin/final_generator /home/cqq/data/{graph}.g {pattern_sizes[p]} {patterns[p]} > {output_name}"
        print(commands)
        os.system(commands)
os.system("cd ../auto; bash concat.sh")