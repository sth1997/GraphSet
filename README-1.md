# 关于GraphMad

## 环境

集群使用`slurm`进行管理。简单来说可以使用以下命令运行cuda程序：
```sh
srun -N 1 -p V100 ./path-to-your-program some-arguments
```
分配V100分区的一个节点，该分区内节点配有16G内存的V100。

使用spack加载cuda11环境：（默认已加载）
```sh
. /mnt/ssd/spack/share/spack/setup-env.sh
spack load cuda@11
```

## 此项目

项目位于`GraphMining-GPU`文件夹中。本应用的输入为两张图（Graph），我们称一个为graph，另一个为pattern，具体任务为统计graph中与pattern同构的子图出现的次数。graph一般是一个很大的图，而pattern通常只有3 ~ 7个节点。

我们目前使用的数据集有5个graph，分别为`wiki-vote`、`patents`、`mico`、`orkut`和`livejournal`，规模递增。它们的简称和文件路径关系如下表所示：

|简称|数据文件路径|
|-|-|
|wv|~hzx/data/wiki-vote.g|
|pt|~hzx/data/patents.g|
|mc|~hzx/data/mico.g|
|lj|~hzx/data/livejournal.g|
|or|~hzx/data/orkut.g|

我们使用的pattern有6个，简称为p1~p6，具体内容可以参考`scripts`中的脚本。

由于通用版程序性能不太理想，我们使用了代码生成为某一组graph和pattern生成对应的cuda kernel。生成的cuda程序代码在`GraphMining-GPU/auto`中，如`mc_p1.cu`（`graph简称_pattern简称.cu`）。

## 运行

项目构建使用`cmake`，`GraphMining-GPU/build/bin`中应该有已经构建好的二进制程序。运行生成的代码只需要给出graph的文件路径即可，如

```sh
bin/mc_p1 ~hzx/data/mico.g
bin/wv_p3 ~hzx/data/wiki-vote.g
```

运行其它的程序需要相应地更改输入文件。

通用版本程序需要给出更多的参数（pattern的点数和无向邻接矩阵），可以暂时不用管它。如果要用，可参考在`patents`上数p1的命令：
```sh
bin/gpu_graph ~hzx/data/patents.g 5 0111010011100011100001100
# executable graph_file #(nodes of pattern) adjacency_matrix_of_pattern 
```
详见`GraphMining-GPU/scripts/run.py`。

如果需要构建项目，请
```sh
source GraphMining-GPU/env.sh
cd GraphMining-GPU/build
cmake ..
make -j
```

若遇到任何问题，请与我们联系。
