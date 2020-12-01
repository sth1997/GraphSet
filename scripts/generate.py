'''
用于生成完全图
'''
if __name__ == '__main__':
    n = 128
    m = n * (n - 1) // 2
    txt = '%d %d\n' % (n, m)
    for i in range(n):
        for j in range(i + 1, n):
            txt += '%d %d\n' % (i+1, j+1)
    with open('clique_input', 'w') as f:
        f.write(txt)
