import sys
import os

if __name__ == '__main__':
    input_name = '/home/hzx/data/graphs/mico.txt'
    output_name = '/home/sth/download/GraphMining-GPU/mico.adj'

    with open(input_name, 'r') as f:
        lines = f.readlines()

    node2labels = {}
    v2real_v = {} #重编号，因为读入带label的图时默认节点编号从0到n-1
    real_v2v = {}
    edges = []

    for line_id, line in enumerate(lines):
        tokens = line.strip().split()

        tp = tokens[0]
        if tp == 'v':
            v = int(tokens[1])
            l = int(tokens[2])
            real_v = len(v2real_v)
            if v in v2real_v:
                print("重复节点")
                sys.exit()
            v2real_v[v] = real_v
            real_v2v[real_v] = v
            node2labels[real_v] = l
        elif tp == 'e':
            u, v, l = int(tokens[1]), int(tokens[2]), int(tokens[3])
            if not u in v2real_v or not v in v2real_v:
                print("节点不存在或还未被读入")
                sys.exit()
            if l != 0:
                print("目前仅支持边的label为0")
                sys.exit()
            u = v2real_v[u]
            v = v2real_v[v]
            edges.append((u, v))
        elif tp == 't':
            if line_id != 0:
                print("输入文件包含不止一组数据")
                sys.exit()
        else:
            print("未知类型")
            print("line num = " + str(line_id))
            sys.exit()

    
    print('|V| = %d |E| = %d' % (len(node2labels), len(edges)))

    new_edges = []
    edges = sorted(edges)

    u, v = edges[0]
    if u != v:
        new_edges.append((u, v))
    for i in range(len(edges) - 1):
        prev_edge = edges[i]
        cur_edge = edges[i + 1]
        if prev_edge != cur_edge:
            u, v = cur_edge
            if u != v:
                new_edges.append(cur_edge)

    print('边表去重并去自环后 |E| = %d' % len(new_edges))

    adj = {}
    for u, v in new_edges:
        def add_edge(adj, u, v):
            if adj.get(u) is None:
                adj[u] = []
            adj[u].append(v)
        
        add_edge(adj, u, v)
        add_edge(adj, v, u)
    
    #由于删除边后可能有0度的点，还需要删除这些点，对点重编号
    old_v_cnt = len(v2real_v)
    v2real_v = {}
    for v in range(old_v_cnt):
        if not adj.get(v) is None:
            real_v = len(v2real_v)
            v2real_v[v] = real_v

    print("删除0度点后 |V| = %d" % len(v2real_v))

    with open(output_name, 'w') as f:
        def wl(s):
            f.write('%s\n' % s)
        def w(s):
            f.write(s)
        wl('%d' % (len(v2real_v)))

        adj_items = sorted(adj.items())
        for u, neighbors in adj_items:
            w('%d %d' % (v2real_v[u], node2labels[u]))
            for v in neighbors:
                w(' %d' % (v2real_v[v]))
            w('\n')
    print("转换完成")

