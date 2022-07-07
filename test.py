import os


for m in [2, 4, 8, 16]:
    for n in [2, 4, 8, 16, 32, 64, 128, 256]:
        if(m > n):
            continue
        command1 = f"CUDA_VISIBLE_DEVICES=5 srun -p V100  ./bin/gpu_test_intersect_1 ~/data/patents.g 10 {m} {n} > result.patents.{m}.{n}.txt"
        print(command1)
        os.system(command1)

        command2 = f"CUDA_VISIBLE_DEVICES=5 srun -p V100  ./bin/gpu_test_intersect_2 ~/data/patents.g 10 {m} {n} >> result.patents.{m}.{n}.txt"
        print(command2)
        os.system(command2)

        command3 = f"CUDA_VISIBLE_DEVICES=5 srun -p V100  ./bin/gpu_test_intersect_3 ~/data/patents.g 10 {m} {n} >> result.patents.{m}.{n}.txt"
        print(command3)
        os.system(command3)

for m in [32, 64, 128, 256]:
    for n in [32, 64, 128, 256]:
        if(m > n):
            continue
        command1 = f"CUDA_VISIBLE_DEVICES=5 srun -p V100  ./bin/gpu_test_intersect_1 ~/data/patents.g 100 {m} {n} > result.patents.{m}.{n}.txt"
        print(command1)
        os.system(command1)

        command2 = f"CUDA_VISIBLE_DEVICES=5 srun -p V100  ./bin/gpu_test_intersect_2 ~/data/patents.g 100 {m} {n} >> result.patents.{m}.{n}.txt"
        print(command2)
        os.system(command2)

        command3 = f"CUDA_VISIBLE_DEVICES=5 srun -p V100  ./bin/gpu_test_intersect_3 ~/data/patents.g 100 {m} {n} >> result.patents.{m}.{n}.txt"
        print(command3)
        os.system(command3)
