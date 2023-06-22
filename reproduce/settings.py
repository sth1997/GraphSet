import os

# modify that to your own path
DATA_PATH = "/home/cqq/data"

# modify that to your own path
COMMAND_PREFIX = "srun -p V100 --gres=gpu:v132p:1 --exclusive "


MULTI_CARD_COMMAND_PREFIX = "srun -p V100 -n 1 --exclusive "