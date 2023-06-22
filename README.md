# GraphSet

> The cpu codes and gpu codes are in `cpu/` and `gpu/`, respectively.
>
> The sample graphs are in `dataset/`. `*.adj` is the labeled graph for Frequent Subgraph Mining.  `*.g` is the binary unlabeled graph for other applications.
>
> Other information, such as how to run different applications and the format of graphs, will be gradually updated in `README.md` with code refactoring.


## Environments

System: Linux

Compilers: gcc@10.2.1 / cuda@11+

Hardware:

+ GPU: Nvidia Volta Architecture or newer (`sm_70` is used now), 32GB memory

Other dependencies:

+ openmpi (> 4.1.1)
+ cmake (> 3.21)

### Use `spack` as package manager

In `env.sh` ( this file should be modified according to specified machines):

Example:

```bash
spack load cuda@11.8
spack load cmake@3.24.3
spack load openmpi@4.1.1
```

### Submodules

We add GoogleTest as submodule to perform unit tests:

+ please add `--recursive` while cloning `git clone https://github.com/sth1997/GraphSet.git --recursive`,

+ or use `git submodule init`, `git submodule update` after clone.


## Build

In the root directory of the project:

```bash
mkdir build && cd build
cmake ..
make -j
```

## test 

After build stage, in the root directory of the project:

```bash
cd build/test && ctest
```

## Usage

in `build/` directory:

### Pattern Matching

#### End-to-End

GPU:

`./bin/gpu_graph <graph_file> <pattern_size> <pattern_matrix_string>`

CPU:

`./bin/pattern_matching_test <graph_file> <pattern_size> <pattern_matrix_string>`


#### Code Generation

Although it is convenient to use **end-to-end** pattern matching directly, its performance will be worse than that of **code generation**. Code generation is only available for GPU code now.

Following steps are need for generate code for specific graph and pattern:

1. build the `code_gen/final_generator.cu` by cmake.
2. provide graphs and patterns in `scripts/gen_code.py` (and note that the `data_path` and `command_prefix` should be changed according to environment)
3. run `scripts/gen_code.py`
4. modify the pattern number and graph name in `auto/CMakeLists.txt`
5. rebuild the project by cmake

Run generated code:

```bash
`./bin/patents_p1 <graph_file> <pattern_size> <pattern_matrix_string>`
```

(The pattern and graph should match the input of code generation)

### Clique Counting

GPU:

`./bin/gpu_clique <graph_file> <clique_size>`

CPU:

`./bin/clique_test <graph_file> <clique_size>`



### Frequent Subgraph Mining

CPU:

`./bin/fsm_test <labelled_graph_file> <max_edge> <min_support>`

GPU:

`./bin/gpu_fsm <labelled_graph_file> <max_edge> <min_support>` 

### Motif Counting

CPU:

`./bin/motif_counting_test <graph_file> <pattern_size>`

GPU:

`./bin/gpu_mc <graph_file> <pattern_size>`

`./bin/mc3 <graph_file>` (for 3-motif counting)


## Input Graph

Two types of graphs are used in our system:

### Labelled Graph

We use a text file for input.

The number `n` on the first line is the number of vertices in the graph.

The following `n` lines, with

+ the no. of the vertex
+ the label no. of the vertex
+ vertices that had a edge to this vertex


### Unlabelled Graph

We suppose these graphs are undirected. To speed up graph reading, we use a binary format (usually `*.g`). 

If you want to input other graphs in text format, you can read `src/dataloader.cpp` for how to change the input method.
 
## Reproduce Scripts

We provide one-step reproduce scripts. Results (tables and figures) will be in `reproduce_result`, and logs will be in `reproduce_log`. The reproduce process will take several hours.

After building the project, in `reproduce/` directory:

* run `python ./pattern_matching.py` to reproduce pattern matching results.
    * it will generate graph for Figure [TODO] as `pattern_matching_gpu.png` and `pattern_matching_cpu.png`
* run `python ./clique_counting.py` to reproduce clique counting results.
    * it will generate graph for Table [TODO] as `clique_counting_gpu.csv` and `clique_counting_cpu.csv`
* run `python ./frequent_subgraph_mining.py` to reproduce frequent subgraph mining results.
    * it will generate graph for Table [TODO] as `frequent_subgraph_mining_gpu.csv` and `frequent_subgraph_mining_cpu.csv`
* run `python ./motif_counting.py` to reproduce motif counting results.
    * it will generate graph for Table [TODO] as `motif_counting_gpu.csv` and `motif_counting_cpu.csv`
* run `python ./scalability.py` to reproduce scalability results.
    * it will generate graph for Figure [TODO] as `pattern_matching_scalability.png`


Also please note that `DATA_PATH` and `COMMAND_PREFIX, MULTI_CARD_COMMAND_PREFIX` in the `reproduce/settings.py` should be modified according to the environment of the machine.

Full dataset can be downloaded through this [link](https://1drv.ms/f/s!Agc-P1eh9RVug-IM6eVlnMCpYCGCpQ?e=LlMueg). You can also choose to preprocess it from the original Stanford Large Network Dataset Collection [link](https://snap.stanford.edu/data/).

