import os

def main():
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
    for graph in graphs:
        for p in range(len(patterns)):
            for node in [1, 2]:
                # for times in [1, 2, 3]:
                log_name = graph + "_" + "p" + str(p + 1) + "_" + str(node) + "node.log"
                command = "srun -N " + str(node) + " -p Big --exclusive ./bin/gpu_graph_static_task /home/cqq/data/" + graph + ".g " + str(pattern_sizes[p]) + " " + str(patterns[p]) + " > " + log_name + " 2>&1 &"
                print(command)
                result = os.system(command)
                if result != 0:
                    return


if __name__ == "__main__":
    main()