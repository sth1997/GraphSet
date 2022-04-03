# GraphMining-GPU

gpu codes are in `gpu/`

Please `source env.sh` first.

Run like this `./gpu_house Wiki-Vote ~zms/wiki-vote_input`


## 构建

项目使用CMake进行构建，请先执行`source env.sh`加载环境。
然后是常见的CMake构建步骤：
```sh
mkdir build
cd build
cmake ..
make -j
```
编译好的程序在`build/bin`中。

## 项目组织

需要重构，部分新内容尚未并入主分支。只简要说明部分。

+ `auto`：自动生成的代码。可以不用管它。
+ `code_gen`：代码生成器，用于对特定的输入生成特化的CUDA代码。但这个程序比较旧，有一些新的在`code-gen-hzx`分支上，不见得有用。
+ `gpu`：重点文件是`gpu_graph.cu`，一般会跑这个程序，后面会详细介绍。

## 运行

我们目前使用的数据集有5个graph，分别为`wiki-vote`、`patents`、`mico`、`orkut`和`livejournal`，规模递增。它们的简称和文件路径关系如下表所示：

|简称|数据文件路径|
|-|-|
|wv|~hzx/data/wiki-vote.g|
|pt|~hzx/data/patents.g|
|mc|~hzx/data/mico.g|
|lj|~hzx/data/livejournal.g|
|or|~hzx/data/orkut.g|

我们使用的pattern有6个，简称为p1~p6，它们的详细信息如下：

|pattern|点数|邻接矩阵|
|-|-|-|
|p1|5|0111010011100011100001100|
|p2|6|011011101110110101011000110000101000|
|p3|6|011111101000110111101010101101101010|
|p4|6|011110101101110000110000100001010010|
|p5|7|0111111101111111011101110100111100011100001100000|
|p6|7|0111111101111111011001110100111100011000001100000|

点数为`n`的pattern常用邻接矩阵表示，它在代码中是一个长度为`n*n`的01字符串。

程序`gpu_graph`接受3个参数，分别是图文件路径、pattern点数和pattern的邻接矩阵表示。当然，pattern点数这个参数是多余的，但由于某些原因一直没改。下面是一个在`patents`数据集上数`house(p1)`的例子：
```sh
bin/gpu_graph ~hzx/data/patents.g 5 0111010011100011100001100
# executable graph_file #(nodes of pattern) adjacency_matrix_of_pattern 
```
更多内容可以参考`scripts`中的脚本`run.py`。
