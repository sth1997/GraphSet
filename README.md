# GraphIndor

GraphIndor is being refactored.

> The cpu codes and gpu codes are in `cpu/` and `gpu/`, respectively.
>
> The sample graphs are in `dataset/`. `*.adj` is the labeled graph for Frequent Subgraph Mining.  `*.g` is the binary unlabeled graph for other applications.
>
> Other information, such as how to run different applications and the format of graphs, will be gradually updated in `README.md` with code refactoring.


## Environments

System: Linux

Compilers: gcc@10.2.1 / cuda@11.0.2

Hardware:

+ GPU: Nvidia Volta Architecture or newer (`sm_70` is used now)

Other dependencies:

+ openmpi
+ cmake ( > 3.21.4 )

### Use `spack` as package manager

In `env.sh` ( this file should be modified according to specified machines):

```bash
spack load cuda@11.0.2
spack load --first cmake@3.21.4
spack load openmpi
```

### Submodules

We add GoogleTest as submodule to perform unit tests:

+ please add `--recursive` while cloning `git clone https://github.com/sth1997/GraphIndor.git --recursive`,

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


Although it is convenient to use **end-to-end** pattern matching directly, its performance will be worse than that of **code generation**. We are refactoring the code generation method and will update them later.

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

Two types of graphs are used in GraphIndor:

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
 
