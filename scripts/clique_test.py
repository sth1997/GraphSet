import os

clique_sizes = [4, 5]
graphs = []
graphs.append("wiki-vote_input")
graphs.append("patents_input")
graphs.append("mico_input")
graphs.append("livejournal_input")
graphs.append("orkut_input")

bin_name = "./bin/clique_test"

for g_idx, graph in enumerate(graphs):
    for p in range(len(clique_sizes)):
        input_file = "/home/ubuntu/data/" + graph
        log_name = graph + "_" + str(clique_sizes[p]) + ".log"
        os.system(bin_name + " " + input_file + " " + str(clique_sizes[p]) + " " + " > " + log_name)