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
graphs.append("CiteSeer ~/dataset/citeseer.g")
graphs.append("Wiki-Vote ~/dataset/wiki-vote.g")
graphs.append("Patents ~/dataset/patents.g")
graphs.append("MiCo ~/dataset/mico.g")
graphs.append("LiveJournal ~/dataset/livejournal.g")
graphs.append("Orkut ~/dataset/orkut.g")

print("#!/bin/bash\nset -x\n")

for graph in graphs:
    for p in range(len(patterns)):
        log_name = "./" + graph + "_" + "motif_p" + str(p + 1) + ".log_$(date -Iseconds)"
        print("../build/bin/baseline_test " + graph + " " + patterns[p] + " > " + log_name)
