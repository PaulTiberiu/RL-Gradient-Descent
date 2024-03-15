import os
import re
import shutil

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import numpy as np
import gym


from bbrl_algos.models.loggers import Logger
from bbrl.agents import Agent, TemporalAgent, Agents
from bbrl_algos.algos.dqn.dqn import local_get_env_agents
from bbrl_algos.models.critics import DiscreteQAgent
from bbrl.workspace import Workspace



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


def get_best_policy(date, time):
    """
    Stocker la meilleure politique dans un fichier
    """
    # Path to the directory containing the best agents
    script_directory = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_directory, "..", "algos", "dqn", "tmp", "hydra", date, time, "dqn_best_agents")
    #print(source_dir)
    
    # Path to the destination directory in the visualization folder
    destination_dir = os.path.join(script_directory, "dqn_best_agents")
    
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


def load_best_agent():
    """
    Load all the best agents in the dqn_best_agents folder using agent.load_model and store them in a list
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_directory, "dqn_best_agents")

    # Get a list of files in the dqn_best_agents folder
    agt_files = [f for f in os.listdir(source_dir) if f.endswith(".agt")]

    # List to store the loaded agents
    loaded_agents = []

    agent = Agent()
    
    # Iterate over each file and load the agent using agent.load_model
    for file in agt_files:
        file_path = os.path.join(source_dir, file)  # Full path to the agent file
        loaded_agent = agent.load_model(file_path)
        loaded_agents.append(loaded_agent)
    
    return loaded_agents


def load_policies(loaded_agents):
    list_policies = []
    for i in range (len(loaded_agents)):
        policy = torch.nn.utils.parameters_to_vector(loaded_agents[i].parameters())
        list_policies.append(policy)
    return list_policies


def plot_triangle_with_new_point(coefficients, new_policy_reward):
    """
    Plot a triangle with vertices representing policies and a new point calculated as a weighted sum of policies.
    Color the new point based on its reward value.

    Parameters:
    - coefficients (list of float): List of coefficients (a1, a2, a3) used to calculate the new point.
    - new_policy_reward (float): Reward value for the new policy.
    """

    # Generate the vertices of the equilateral triangle
    triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])

    # Plot the triangle edges
    plt.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], 'k-')

    # Plot policy vertices
    #plt.plot(triangle_vertices[:3, 0], triangle_vertices[:3, 1], 'ko')  # Black circles
    plt.text(triangle_vertices[0, 0] - 0.05, triangle_vertices[0, 1] - 0.03, 'p1', fontsize=10)
    plt.text(triangle_vertices[1, 0] + 0.05, triangle_vertices[1, 1] - 0.03, 'p2', fontsize=10)
    plt.text(triangle_vertices[2, 0], triangle_vertices[2, 1] + 0.03, 'p3', fontsize=10)

    # Calculate the coordinates of the new point
    new_point = np.dot(np.array(coefficients), triangle_vertices[:3])

    # Normalize the reward value to range from 0 to 1
    norm = mcolors.Normalize(vmin=0, vmax=500)

    # Choose a colormap that covers the entire range from 0 to 500
    cmap = plt.get_cmap('coolwarm')

    # Plot the new point representing the weighted sum of policies, with color based on its reward
    plt.scatter(new_point[0], new_point[1], c=new_policy_reward, cmap=cmap, norm=norm, s=50)

    # Set axis limits and labels
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, np.sqrt(3)/2 + 0.1)

    # Add color bar legend
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm))
    cbar.set_label('Reward')

    # Add legend
    plt.legend(['Triangle Edges', 'Policy Vertices', 'New Point'])

    # Show the plot
    plt.show()

def plot_triangle_with_multiple_points(coefficients_list, rewards_list):
    """
    Plot a triangle with vertices representing policies and a new point calculated as a weighted sum of policies.
    Color the new point based on its reward value.

    Parameters:
    - coefficients (list of float): List of coefficients (a1, a2, a3) used to calculate the new point.
    - new_policy_reward (float): Reward value for the new policy.
    """

    # Generate the vertices of the equilateral triangle
    triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])

    # Plot the triangle edges
    plt.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], 'k-')

    # Plot policy vertices
    #plt.plot(triangle_vertices[:3, 0], triangle_vertices[:3, 1], 'ko')  # Black circles
    plt.text(triangle_vertices[0, 0] - 0.05, triangle_vertices[0, 1] - 0.03, 'p1', fontsize=10)
    plt.text(triangle_vertices[1, 0] + 0.05, triangle_vertices[1, 1] - 0.03, 'p2', fontsize=10)
    plt.text(triangle_vertices[2, 0], triangle_vertices[2, 1] + 0.03, 'p3', fontsize=10)


    # Normalize the reward value to range from 0 to 1
    norm = mcolors.Normalize(vmin=0, vmax=500)

    # Choose a colormap that covers the entire range from 0 to 500
    cmap = plt.get_cmap('viridis')

    # Calculate the coordinates of the new point
    for coefs, reward in zip(coefficients_list, rewards_list):
        new_point = np.dot(np.array(coefs), triangle_vertices[:3])
        plt.scatter(new_point[0], new_point[1], c=reward, cmap=cmap, norm=norm, s=50)


    # Plot the new point representing the weighted sum of policies, with color based on its reward

    # Set axis limits and labels
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, np.sqrt(3)/2 + 0.1)

    # Add color bar legend
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm))
    cbar.set_label('Reward')

    # Add legend
    plt.legend(['Triangle Edges'])

    # Show the plot
    plt.show()


def update_policy_with_coefficients(list_policies, coefficients):
    """
    Returns policy = a1 * p1 + a2 * p2 + a3 * p3
    """

    # Ensure the number of coefficients matches the number of policy tensors
    if len(list_policies) != len(coefficients):
        raise ValueError("Number of policy tensors and coefficients must match.")

    # Initialize the updated policy tensor with zeros
    updated_policy = torch.zeros_like(list_policies[0])

    # Update the policy tensor using the provided coefficients
    for coeff, policy in zip(coefficients, list_policies):
        updated_policy += coeff * policy

    return updated_policy

def load_reward(eval_agent):
    workspace = Workspace()

    eval_agent(workspace, t=0, stop_variable="env/done",choose_action=True)

    rewards = workspace["env/cumulated_reward"][-1]
    mean_reward = rewards.mean()

    return mean_reward

def get_policy_reward(coefficients, loaded_agents):
    rewards = []

    for i in range(len(loaded_agents)):
        rewards.append(load_reward(loaded_agents[i]))
        
    new_policy_reward = 0
    if(len(coefficients) == len(rewards)):
        for j in range(len(coefficients)):
            new_policy_reward += coefficients[j] * rewards[j]

    return new_policy_reward


def generate_coefficients_list(num_points):

    # Generate random coefficients for each point
    coefficients_list = []
    for _ in range(num_points):
        coefficients = np.random.dirichlet(np.ones(3), size=1).flatten()
        coefficients_list.append(coefficients.tolist())

    return coefficients_list

def save_policy_and_get_agent(policy):
    # Save the policy tensor to the specified file
    script_directory = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_directory, "Cartpole-v1.agt")
    torch.save(policy, source_dir)

    #print(source_dir)

    # Initialize a new agent instance
    new_agent = Agent()

    # Load the policy tensor from the file into the agent
    new_agent.load_model(source_dir)

    return new_agent


@hydra.main(
    config_path="../algos/dqn/configs/",
    config_name="dqn_cartpole.yaml",
    #config_name="dqn_lunar_lander.yaml", #cartpole
)  # , version_base="1.3")

def main(cfg_raw: DictConfig):
    #print(read_tensor_file())

    date = "2024-03-10" 
    time = "10-44-08"
    get_best_policy(date, time)

    date = "2024-03-10" 
    time = "10-49-00"
    get_best_policy(date, time)

    date = "2024-03-10" 
    time = "10-53-36"
    get_best_policy(date, time)

    loaded_agents = load_best_agent()
    #print(loaded_agents)

    list_policies = load_policies(loaded_agents)


    coefficients = [0, 0, 1]
    new_policy_reward = get_policy_reward(coefficients, loaded_agents)
    print(new_policy_reward)

    #plot_triangle_with_new_point(coefficients, new_policy_reward)
    #coefficients_list = [[0, 0, 1], [0, 1, 0], [1, 0, 0],[0.3, 0.5, 0.2]]

    coefficients_list = generate_coefficients_list(500)
    #print(coefficients_list)

    rewards_list = [get_policy_reward(coeff, loaded_agents) for coeff in coefficients_list]
    plot_triangle_with_multiple_points(coefficients_list, rewards_list)


    #new_policy = update_policy_with_coefficients(list_policies, coefficients)
    #print(new_policy)


    # new_agent = save_policy_and_get_agent(new_policy)
    # print(new_agent)



if __name__ == "__main__":
    main()

