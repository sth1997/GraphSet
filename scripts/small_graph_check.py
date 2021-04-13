# 寻找一个能复现bug的小图。
# 如果pattern大小为N，则会先检查所有大小为N的图（共2^(N*(N-1)/2)种），再检查大小为N+1的图……直到找到发生错误
# 由于原程序中输出的东西太多，需要先修改源程序，让其将答案输出至1.out和2.out，其余内容可以输出至stdout（本程序中不会做重定向）
# 使用本程序，需要修改N、bin1和bin2，编译一个正确版本和有bug版本的
# 由于我们的dataloader要求不能存在度数为0的点，所以很多生成的图是不合法的，如果报了“vertex number error!”是正常的

import os

N = 5
bin1 = "../build/bin/gpu_graph"
res1 = "1.out"
bin2 = "../build/bin/gpu_graph_correct"
res2 = "2.out"

while (True):
    print("N = " + str(N))
    print("range = ", range(2 ** int(N * (N - 1) / 2)))
    for i in range(2 ** int(N * (N - 1) / 2)):
        x = i
        e_num = 0
        while (x != 0):
            e_num += x & 1
            x >>= 1
        f = open("graph.out", "w")
        f.write(str(N) + " " + str(e_num) + "\n")
        x = i
        for j in range(1, N):
            for k in range(j + 1, N + 1):
                if (x & 1 == 1):
                    f.write(str(j) + " " + str(k) + "\n")
                x >>= 1
        f.close()
        os.system("srun -N 1 " + bin1 + " Patents graph.out")
        os.system("srun -N 1 " + bin2 + " Patents graph.out")
        if os.path.exists(res1) and os.path.exists(res2) and os.system("diff -w -q " + res1 + " " + res2) != 0:
            break
        print("i = " + str(i) + "  Correct.")
    N += 1
        