import os

patterns = []
patterns.append("3 011101110")
patterns.append("3 011100100")
patterns.append("4 0111101111011110")
patterns.append("4 0111101111001100")
patterns.append("4 0111101011001000")
patterns.append("4 0111100010001000")
patterns.append("4 0110100110010110")
patterns.append("4 0110100110000100")
graphs = []
graphs.append("citeseer.g")
graphs.append("wiki-vote.g")
graphs.append("patents.g")
graphs.append("mico.g")
graphs.append("livejournal.g")
graphs.append("orkut.g")
g = []
g.append('cs')
g.append('wv')
g.append("pt")
g.append("mc")
g.append("lj")
g.append("or")
for g_idx, graph in enumerate(graphs):
    for p in range(len(patterns)):
        bin_name = g[g_idx] + "_motif_test" + str(p + 1)
        log_name = "./motif_test_log/" + graph + "_" + "p" + str(p + 1) + ".log"
        #log_name = "../auto/" + g[g_idx] + "_p" + str(p + 1) + "_inject.cu" 
        os.system("./build/bin/" + bin_name + " /home/zms/GraphMining/graph_dataset/" + graph + " " + str(patterns[p]) + " > " + log_name + " &")
