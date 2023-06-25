# AE of SC23

**For AE reviewers, we have prepared a server containing all the datasets and codes. Please contact sth19@mails.tsinghua.edu.cn to access the server.**

## Reproduce

(i) To begin, ensure that you have a machine with a performance configuration similar to the following: Intel Xeon Platinum 8259CL CPU @ 2.50GHz, 1 socket (16 cores, hyper-threading disabled), 256 GB memory, and at least 8 NVIDIA Tesla V100 (32GB memory) GPUs for PCIe.

Our open-source project can be found at https://github.com/sth1997/GraphSet, and the steps to reproduce our work are as follows:

1. Download the repository code via git.

2. The full dataset can be accessed via https://1drv.ms/f/s!Agc-P1eh9RVug-IM6eVlnMCpYCGCpQ?e=LlMueg. Alternatively(and not recommended), you can opt to preprocess the dataset from the original Stanford Large Network Dataset Collection, which can be found at https://snap.stanford.edu/data/.

3. Install the necessary dependencies: gcc@12, CUDA@12, cmake@3.26, openmpi@4.1.5, and python@3 (preinstalled on our clusters, so you only need to load it.). Examples by using module load command:
    * `module load cuda-12.0.1-gcc-12.2.0-34tfuhe`
    * `module load openmpi-4.1.5-gcc-12.2.0-awa3vt5`
    * `module load cmake-3.26.3-gcc-12.2.0-caw7voo`

4. Build the project by executing `mkdir build; cd build; cmake ..; make -j` in the project's root folder.

5. Please note that `DATA_PATH` and `COMMAND_PREFIX, MULTI_CARD_COMMAND_PREFIX` in the `reproduce/settings.py` should be modified according to the environment of the machine.

(ii) The reproduction process:

After building the project, in `reproduce/` directory:

* run `python ./pattern_matching.py` to reproduce pattern matching results. (Time cost: 7 hours)
    * it will generate graph for Figure [TODO] as `pattern_matching_gpu.png` and `pattern_matching_cpu.png`
* run `python ./clique_counting.py` to reproduce clique counting results. (Time cost: 20 minutes)
    * it will generate file for Table [TODO] as `clique_counting_gpu.csv` and `clique_counting_cpu.csv`
* run `python ./frequent_subgraph_mining.py` to reproduce frequent subgraph mining results. (Time cost: 7 hours)
    * it will generate file for Table [TODO] as `frequent_subgraph_mining_gpu.csv` and `frequent_subgraph_mining_cpu.csv`
* run `python ./motif_counting.py` to reproduce motif counting results. (Time cost: 6 hours)
    * it will generate file for Table [TODO] as `motif_counting_gpu.csv` and `motif_counting_cpu.csv`
* run `python ./scalability.py` to reproduce scalability results (Time cost: 1 hour)
    * it will generate graph for Figure [TODO] as `pattern_matching_scalability.png`
* run `python ./gsi_cuts.py` to reproduce our results compared with GSI and cuTS. (Time cost: 5 hours)
    * it will generate file for Table [TODO] as `attern_matching_gsi_cuts_gpu.csv`

(iii) Upon completion, results (tables and figures) will be located in the `reproduce_result` folder, and the logs will be in the `reproduce_log` folder. The csv files within the `reproduce_result` folder will display the application time of our system. We expect the counting results to be consistent with those of other systems and previously provided logs/results in our repository.

(iv) The experimental running times obtained should closely match those reported in our article, thereby validating the work presented therein.

# GraphSet

> The cpu codes and gpu codes are in `cpu/` and `gpu/`, respectively.
>
> The sample graphs are in `dataset/`. `*.adj` is the labeled graph for Frequent Subgraph Mining.  `*.g` is the binary unlabeled graph for other applications.
>
> Other information, such as how to run different applications and the format of graphs, will be gradually updated in `README.md` with code refactoring.


## Environments

System: Linux

Compilers: gcc@12(minimum 10) / cuda@12(minimum 11)

Hardware:

+ GPU: Nvidia Volta Architecture or newer (`sm_70` is used now), 32GB memory

Other dependencies:

+ openmpi (> 4.1)
+ cmake (> 3.21)

### Use `spack` as package manager

In `env.sh` ( this file should be modified according to specified machines):

Example:

```bash
spack load cuda@12.0
spack load cmake@3.26
spack load openmpi@4.1.5
```

## Build

In the root directory of the project:

```bash
mkdir build && cd build
cmake ..
make -j
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
 

