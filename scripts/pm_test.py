import os
from os.path import join

import time

graph_files_dict = {
    'wv': 'wiki-vote.g',
    'pt': 'patents.g',
    'mc': 'mico.g',
    'lj': 'livejournal.g',
    'ok': 'orkut.g'
}

patterns_dict = {
    'p1': (5, '0111010011100011100001100'),
    'p2': (6, '011011101110110101011000110000101000'),
    'p3': (6, '011111101000110111101010101101101010'),
    'p4': (6, '011110101101110000110000100001010010')
    # 'p5': (7, "0111111101111111011101110100111100011100001100000"),
    # 'p6': (7, "0111111101111111011001110100111100011000001100000")
}

# (graph_name, pattern_name, time_limit (in minutes))
# schemes = [
#     # ('wv', 'p5', 1),
#     # ('wv', 'p6', 1),
#     # ('pt', 'p5', 1),
#     # ('pt', 'p6', 1),
#     # ('mc', 'p5', 1),
#     ('mc', 'p6', 1),
#     # ('lj', 'p5', 6),
#     ('lj', 'p6', 3),
#     ('ok', 'p5', 2),
#     ('ok', 'p6', 1),
# ]

# schemes = [
#     ('wv', 'p5', 1),
#     ('pt', 'p5', 1),
#     ('mc', 'p5', 1),
#     ('ok', 'p5', 2),
#     ('lj', 'p5', 6)
# ]

schemes = [
    # ('wv', 'p1', 1),
    # ('wv', 'p2', 1),
    # ('wv', 'p3', 1),
    ('wv', 'p4', 1),
    ('pt', 'p1', 1),
    ('pt', 'p2', 1),
    ('pt', 'p3', 1),
    ('pt', 'p4', 1),
    ('mc', 'p1', 1),
    ('mc', 'p2', 1),
    ('mc', 'p3', 1),
    ('mc', 'p4', 1),
    ('ok', 'p1', 4),
    ('ok', 'p2', 1),
    ('ok', 'p3', 1),
    ('ok', 'p4', 4),
    ('lj', 'p1', 1),
    ('lj', 'p2', 1),
    ('lj', 'p3', 1),
    ('lj', 'p4', 1)
]

base_dir = '/home/hzx/GraphMining-GPU/'

def index(i, j, n):
    return i * n + j

def generate_permutations(perms, buf, n):
    if len(buf) >= n:
        perms.append([x for x in buf])
        return
    for i in range(n):
        if i not in buf:
            buf.append(i)
            generate_permutations(perms, buf, n)
            buf.pop()

def count_times(log_file_names):
    import re
    times, perm_ids = [], []
    for filename in log_file_names:
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line for line in lines if line.startswith('Counting time cost')]
        if len(lines) == 0:
            continue

        line = lines[0]
        t = re.findall('\d+', line)
        times.append(float(t[0] + '.' + t[1]))

        t = re.findall('\d+', filename)
        perm_ids.append(int(t[1]))
    return times, perm_ids


try:
    for graph_name, pattern_name, time_limit in schemes:
        graph_path = join('~hzx/data/', graph_files_dict[graph_name])
        n, adj_mat = patterns_dict[pattern_name]

        perms = []
        rank = [0] * n
        cur_adj_mat = ['0'] * (n * n)
        generate_permutations(perms, [], n)
        for perm_id, perm in enumerate(perms):
            # if perm_id != 8:
            #     continue
            # if perm_id != 1571:
            #     continue

            print('%s-%s perm %d' % (graph_name, pattern_name, perm_id))
            # if (graph_name == 'pt') and (pattern_name == 'p5') and (perm_id < 1045):
            #     continue
            # if (graph_name == 'mc') and (pattern_name == 'p6') and (perm_id < 3099):
            #     continue

            for i in range(n):
                rank[perm[i]] = i
            for i in range(n):
                for j in range(n):
                    cur_adj_mat[index(rank[i], rank[j], n)] = adj_mat[index(i, j, n)]
            adj_mat_str = ''.join(cur_adj_mat)
            
            # code generation
            code_gen_path = join(base_dir, 'auto/test_inj.cu')
            os.chdir(join(base_dir, 'build'))
            os.system('srun -N 1 bin/final_generator %s %d %s > %s' % (graph_path, n, adj_mat_str, code_gen_path))

            pattern_valid = True
            with open(code_gen_path, 'r') as f:
                if len(f.read()) < 100:
                    pattern_valid = False
            
            if not pattern_valid:
                print('invalid pattern')
                continue
                
            os.chdir(join(base_dir, 'auto'))
            os.system('cat before_inj.cu > test.cu')
            os.system('cat test_inj.cu >> test.cu')
            os.system('cat after_inj.cu >> test.cu')

            os.chdir(join(base_dir, 'build'))
            os.system('make -j')

            # log_name = 'logs/00-%s-%s-%d.txt' % (graph_name, pattern_name, perm_id) 
            # os.system('srun -N 1 bin/test %s %d %s | tee %s' % (graph_path, n, adj_mat_str, log_name))
            # break

            log_name = 'logs/%s-%s-%d.txt' % (graph_name, pattern_name, perm_id)
            os.system('srun -N 1 -t %d bin/test %s %d %s > %s' % (time_limit, graph_path, n, adj_mat_str, log_name))

            time.sleep(0.5)
except KeyboardInterrupt:
    print('\n\n-------------  stop --------------------\n\n')