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
graphs.append("wiki-vote.g")
graphs.append("patents.g")
graphs.append("mico.g")
graphs.append("livejournal.g")
graphs.append("orkut.g")
g = []
g.append('wv')
g.append("pt")
g.append("mc")
g.append("lj")
g.append("or")
for g_idx, graph in enumerate(graphs):
    for p in range(len(patterns)):
        #log_name = graph + "_" + "p" + str(p + 1) + ".log"
        target_prefix = "../auto/" + g[g_idx] + "_p" + str(p + 1)
        inject_target = target_prefix + "_inject.cu" 
        final_target = target_prefix + ".cu"

        os.system("./bin/code_generator /home/hzx/data/" + graph + " " + str(pattern_sizes[p]) + " " + str(patterns[p]) + " > " + inject_target)
        os.system("cat ../auto/before_inj.cu > %s" % final_target)
        os.system("cat %s >> %s" % (inject_target, final_target))
        os.system("cat ../auto/after_inj.cu >> %s" % final_target)
