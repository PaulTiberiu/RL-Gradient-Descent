import numpy as np
import os
import torch
import torch.nn as nn
import sys
import re
import shutil
from datetime import datetime

import hydra
from omegaconf import DictConfig

from bbrl_algos.models.loggers import Logger

# HYDRA_FULL_ERROR = 1
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../algos/dqn')
from dqnv2 import run_dqn

matplotlib.use("TkAgg")


def extract_number_from_filename(filename):
    """
    Extracts the numerical value (XXX) from a filename of the form "CartPole-v1dqnXXX.agt"
    """
    match = re.search(r'CartPole-v1dqn(\d+\.\d+).agt', filename)
    if match:
        return float(match.group(1))
    return None
    

def get_last_chronological_folder(source_dir):
    """
    Returns the filename of the last chronological folder by time within the specified source directory
    """
    # Get a list of folders in chronological order by time
    folders = sorted([f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))],
                     key=lambda x: os.path.getmtime(os.path.join(source_dir, x)),
                     reverse=True)
    
    # Return the filename of the first folder (last chronological folder)
    return folders[0] if folders else None


def get_best_model(date, time):
    """
    Stocker les 3 meilleures politiques dans un fichier
    """
    # Path to the directory containing the best agents
    script_directory = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_directory, "tmp", "hydra", date, time, "dqn_best_agents")
    print(source_dir)
    
    # Path to the destination directory in the visualization folder
    destination_dir = os.path.join(script_directory, "tmp","dqn_best_agents")
    
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Get a list of folders in chronological order
    agt_files = [f for f in os.listdir(source_dir) if f.endswith(".agt")]
    #print(agt_files)
    
    # Extract the values from each folder and append them to an array
    values_array = [extract_number_from_filename(agt_file) for agt_file in agt_files]
    #print(values_array)    
    # Find the index of the folder with the maximum value
    max_index = values_array.index(max(values_array))
    
    # Get the folder with the maximum value
    max_value_agt_file = agt_files[max_index]
    
    # Copy the files from the folder with the max value to the destination directory
    source_file = os.path.join(source_dir, max_value_agt_file)
    destination_file = os.path.join(destination_dir, max_value_agt_file)
    shutil.copyfile(source_file, destination_file)
    
    # Print the results
    # print("Values array:", values_array)
    # print("File with max value:", max_value_agt_file)
    # print("File with max value copied successfully to:", destination_dir)


def get_best_3_policies(cfg_raw, logger):

    for _ in range(3):
        cfg_raw.algorithm.seed.train = np.random.randint(1, 35) # seeds between 1 and 35 
        print(cfg_raw.algorithm.seed.train)
        run_dqn(cfg_raw,logger)

        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%Y-%m-%d")

        script_directory = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(script_directory, "tmp", "hydra", current_date)

        current_time = get_last_chronological_folder(source_dir)
        get_best_model(current_date, current_time)


#def load_best_models(agent):


@hydra.main(
    config_path="../algos/dqn/configs/",
    config_name="dqn_cartpole.yaml",
    #config_name="dqn_lunar_lander.yaml", #cartpole
)  # , version_base="1.3")

def main(cfg_raw: DictConfig):
    #print(read_tensor_file())

    logger = Logger(cfg_raw)
    #get_best_3_policies(cfg_raw, logger)

if __name__ == "__main__":
    main()
