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

## Build

In the root directory of the project:

```bash
mkdir build && cd build
cmake ..
make -j
```

## Usage

### Pattern Matching(include Clique Counting)

#### Code Generation

### Frequent Subgraph Mining

### Motif Counting

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
 