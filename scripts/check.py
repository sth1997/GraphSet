from IPython import embed
from operator import itemgetter

def parse_kv(line, sep=':'):
    ls = line[1:].split(sep)
    return int(ls[0]), int(ls[1])

def check():
    with open('32.txt', 'r') as f:
        raw_lines1 = f.readlines()
    with open('64.txt', 'r') as f:
        raw_lines2 = f.readlines()
    
    lines1 = [line for line in raw_lines1 if line[0] == '#']
    lines2 = [line for line in raw_lines2 if line[0] == '#']

    es1 = dict(parse_kv(line) for line in lines1)
    es2 = dict(parse_kv(line) for line in lines2)

    assert(es1.keys() == es2.keys())

    outputs = []
    for k in es1.keys():
        v1, v2 = es1[k], es2[k]
        if v1 != v2:
            line = 'i: %d v1: %d v2: %d\n' % (k, v1, v2)
            outputs.append(line)
    
    txt = ''.join(outputs)
    with open('diff.txt', 'w') as f:
        f.write(txt)

def sort_data():
    with open('8.txt', 'r') as f:
        raw_lines = f.readlines()
    
    values = [tuple(map(int, line[1:].split(','))) for line in raw_lines if line[0] == '$']
    values.sort()
    for v in values:
        print('block %d edge %d thread %d val %d' % (v[0], v[1], v[2], v[3]))

if __name__ == '__main__':
    check()
