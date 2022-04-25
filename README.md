# GraphMining-GPU

clique improvements

+ edge orientation
+ delete extra codes (restriction & subtraction_set & tmp_set)
+ cub prefix warp scan

usage:

(cuda 11.0+ required)

```
source ../env.sh
mkdir build && cd build
cmake ..
make -j
```

```
./bin/gpu_clique /home/hzx/data/orkut.g 4 0111101111011110
# srun -p V100 ./bin/gpu_clique /home/hzx/data/orkut.g 4 0111101111011110
./bin/gpu_clique /home/hzx/data/orkut.g 5 0111110111110111110111110
# srun -p V100 ./bin/gpu_clique /home/hzx/data/orkut.g 5 0111110111110111110111110
```

(must use doubled edge pattern & double edge graph)