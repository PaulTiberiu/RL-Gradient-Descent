import numpy as np
import os
import torch
import torch.nn as nn
import sys

import hydra
from omegaconf import DictConfig

from bbrl_algos.models.loggers import Logger

# HYDRA_FULL_ERROR = 1
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../algos/dqn')
from dqnv2 import run_dqn

matplotlib.use("TkAgg")


def read_tensor_file():
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative path to the file
    policies_file_path = os.path.join(script_directory, "policies.txt")

    
    with open(policies_file_path, "r") as pol_file:

        pol_data = [float(line.strip()) for line in pol_file.readlines()]

    pol_values = [float(value) for value in pol_data]

    return pol_values


def get_best_policy(tensor_list):
    return np.max(tensor_list) 


def get_best_3_policies(cfg_raw,logger):
    
    best_policies = []
    for i in range(3):
        #cfg_raw.algorithm.seed.train = i+1
        cfg_raw.algorithm.seed.train = np.random.randint(1, 35) # better for random seed
        #print("youpi", cfg_raw.algorithm.seed.train)
        run_dqn(cfg_raw,logger)
        tensor_list = read_tensor_file()
        best_policy = get_best_policy(tensor_list)
        best_policies.append(best_policy)
    
    return best_policies

@hydra.main(
    config_path="../algos/dqn/configs/",
    config_name="dqn_cartpole.yaml",
    #config_name="dqn_lunar_lander.yaml", #cartpole
)  # , version_base="1.3")

def main(cfg_raw: DictConfig):
    #print(read_tensor_file())
    logger = Logger(cfg_raw)
    print(run_dqn_3_times(cfg_raw,logger))

if __name__ == "__main__":
    main()
