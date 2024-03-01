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


def load_best_model(env_name, dirname, fileroot, agent):
    search_path = os.path.join("..", "algos", "dqn", dirname, env_name)
    # Assuming only one file will match the pattern
    file_list = [f for f in os.listdir(search_path) if f.startswith(fileroot)]
    if not file_list:
        print("No model file found.")
        return None
    filename = os.path.join(search_path, file_list[0])
    return agent.load_model(filename + ".agt")

def get_best_3_policies(cfg_raw,logger,agent):
    best_policies = []
    for _ in range(3):
        cfg_raw.algorithm.seed.train = np.random.randint(1, 35) # better for random seed
        run_dqn(cfg_raw,logger)
        best = load_best_model(cfg_raw.env.name, cfg_raw.algorithm.dirname, cfg_raw.algorithm.fileroot, agent)
        best_policies.append(best)
    return best_policies



@hydra.main(
    config_path="../algos/dqn/configs/",
    config_name="dqn_cartpole.yaml",
    #config_name="dqn_lunar_lander.yaml", #cartpole
)  # , version_base="1.3")

def main(cfg_raw: DictConfig):
    #print(read_tensor_file())
    logger = Logger(cfg_raw)

    # Where can i load this in order to get the best policy?
    # Does it have to be in the dqnv2 file?
    print(load_best_model(cfg_raw.env.name, cfg_raw.algorithm.dirname, cfg_raw.algorithm.fileroot, eval_agent))
    #policies = get_best_3_policies(cfg_raw,logger)
    #print(policies)

if __name__ == "__main__":
    main()
