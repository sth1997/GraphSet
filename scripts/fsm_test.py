import os

min_support = [x for x in range(30000, 0, -1000)] + [x for x in range(1000, 100, -100)]
max_edges = [2, 3]
graphs = []
graphs.append("patents.adj")
graphs.append("youtube.adj")
graphs.append("mico.adj")
g = []
g.append("pt")
g.append("yt")
g.append("mc")
for g_idx, graph in enumerate(graphs):
    for b in min_support:
        for a in max_edges:
            bin_name = "fsm_test"
            log_name = g[g_idx] + "_" + str(a) + "fsm_" + str(b) + ".log"
            command = "./bin/" + bin_name + " /home/cqq/data/" + graph + " " + str(a) + " " + str(b) + " > " + log_name
            print(command)
            os.system(command)