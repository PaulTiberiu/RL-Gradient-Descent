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
from datetime import datetime



from bbrl_algos.models.loggers import Logger
from bbrl.agents import Agent, TemporalAgent, Agents
########################################################################################################################
from bbrl_algos.algos.dqn.dqn import local_get_env_agents
# TO CHANGE
#from bbrl_algos.models.critics import DiscreteQAgent
########################################################################################################################
from bbrl.workspace import Workspace
import plotly.graph_objects as go



def extract_number_from_filename(filename):
    """
    Extracts the numerical value (XXX) from a filename of the form "CartPole-v1dqnXXX.agt"
    """
    match = re.search(r'Swimmer-v3-td3-(\d+\.\d+).agt', filename)
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


def get_best_policy(date, time, suffix):
    """
    Stocker la meilleure politique dans un fichier
    """
    # Path to the directory containing the best agents
    script_directory = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_directory, "..", "algos", "td3", "tmp", "hydra", date, time, "td3_best_agents")
    
    # Path to the destination directory in the visualization folder
    destination_dir = os.path.join(script_directory, "td3_best_agents")
    
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Get a list of folders in chronological order
    agt_files = [f for f in os.listdir(source_dir) if f.endswith(".agt")]
    
    # Extract the values from each folder and append them to an array
    values_array = [extract_number_from_filename(agt_file) for agt_file in agt_files]
    
    # Find the index of the folder with the maximum value
    max_index = values_array.index(max(values_array))
    
    # Get the folder with the maximum value
    max_value_agt_file = agt_files[max_index]
    
    # Copy the files from the folder with the max value to the destination directory
    source_file = os.path.join(source_dir, max_value_agt_file)
    
    # Modify the destination file name to include the suffix
    base_name, extension = os.path.splitext(max_value_agt_file)
    destination_file = os.path.join(destination_dir, f"{base_name}_{suffix}{extension}")
    
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
    source_dir = os.path.join(script_directory, "td3_best_agents")

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


def plot_triangle_with_multiple_points(coefficients_list, rewards_list):
    """
    Plot a triangle with vertices representing policies and a new point calculated as a weighted sum of policies.
    Color the new point based on its reward value.

    Parameters:
    - coefficients_list (list of lists): List of coefficients (a1, a2, a3) used to calculate the new point.
    - rewards_list (list of float): List of reward values for each point.
    """

    # Generate the vertices of the equilateral triangle
    triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2],[0,0]])

    # Plot the triangle edges
    plt.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], 'k-')

    # Plot policy vertices
    plt.text(triangle_vertices[0, 0] - 0.05, triangle_vertices[0, 1] - 0.03, 'p1', fontsize=10)
    plt.text(triangle_vertices[1, 0] + 0.05, triangle_vertices[1, 1] - 0.03, 'p2', fontsize=10)
    plt.text(triangle_vertices[2, 0], triangle_vertices[2, 1] + 0.03, 'p3', fontsize=10)

    #norm = mcolors.Normalize(vmin=0, vmax=500) #TO TO HAVE THE VALUES FROM 0 TO 500
    norm = mcolors.Normalize(vmin=min(rewards_list), vmax=max(rewards_list))

    # Choose a colormap that covers the entire range of rewards
    cmap = plt.get_cmap('RdBu_r')
    #cmap = plt.get_cmap('viridis')

    # Plot the points with adjusted positions and transparency
    for coefs, reward in zip(coefficients_list, rewards_list):
        new_point = np.dot(np.array(coefs), triangle_vertices[:3])
        #jitter = np.random.normal(0, 0.01, 2)  # Add a small random jitter to avoid superposition
        #plt.scatter(new_point[0] + jitter[0], new_point[1] + jitter[1], c=reward, cmap=cmap, norm=norm, alpha=0.5, s=250)
        plt.scatter(new_point[0], new_point[1], c=reward, cmap=cmap, norm=norm, s=60)
    # Set axis limits and labels
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, np.sqrt(3)/2 + 0.1)

    # Add color bar legend
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=plt.gca())
    cbar.set_label('Reward')

    # Add legend
    plt.legend(['Triangle Edges'])


    script_directory = os.path.dirname(os.path.abspath(__file__))


    save_directory = os.path.join(script_directory, "..", "..","..", "visualization_results")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Then save the plot using the save_directory
    save_path = os.path.join(save_directory, f"swimmer_policy_{date_time}.png")

    plt.savefig(save_path)

    # Show the plot
    plt.show()



def plot_triangle_with_multiple_points_plotly(coefficients_list, rewards_list):
    """
    Plot a triangle with vertices representing policies and a new point calculated as a weighted sum of policies.
    Color the new point based on its reward value.

    Parameters:
    - coefficients_list (list of lists): List of coefficients (a1, a2, a3) used to calculate the new point.
    - rewards_list (list of torch.Tensor): List of reward tensors for each point.
    """

    # Convert Torch tensors to standard Python numbers
    rewards_list = [reward.item() for reward in rewards_list]

    # Generate the vertices of the equilateral triangle
    triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2],[0,0]])

    # Create trace for triangle edges
    triangle_edges = go.Scatter(
        x=triangle_vertices[:, 0],
        y=triangle_vertices[:, 1],
        mode='lines',
        line=dict(color='black'),
        showlegend=False,  # Set this to False to hide from legend
    )

    # Create trace for policy vertices
    policy_vertices = go.Scatter(
        x=[triangle_vertices[0, 0], triangle_vertices[1, 0], triangle_vertices[2, 0]],
        y=[triangle_vertices[0, 1], triangle_vertices[1, 1], triangle_vertices[2, 1]],
        mode='text',
        text=['P1', 'P2', 'P3'],
        textposition='top center',
        textfont=dict(size=20),
        showlegend=False,  # Set this to False to hide from legend
    )

    # Create trace for new points
    min_reward = min(rewards_list)
    max_reward = max(rewards_list)


    # Create trace for new points with the updated colorbar attributes
    new_points = []
    for coefs, reward in zip(coefficients_list, rewards_list):
        new_point = np.dot(np.array(coefs), triangle_vertices[:3])
        hover_text = f"Reward: {reward:.2f}<br>Coordinates: ({new_point[0]:.2f}, {new_point[1]:.2f})<br>Alphas: {coefs}"
        new_points.append(go.Scatter(
            x=[float(new_point[0])],
            y=[float(new_point[1])],
            mode='markers',
            marker=dict(
                color=[reward],  # Enclose reward in a list to define its color based on the 'RdBu' scale
                size=10,
                colorbar=dict(
                    title='Reward',
                    tickvals=[min_reward, max_reward],
                    ticktext=[f'{min_reward:.2f}', f'{max_reward:.2f}']
                ),
                colorscale='RdBu_r',  # Use the built-in Red-Blue color scale
                cmin=min_reward,  # Explicitly set the min for color scaling
                cmax=max_reward,  # Explicitly set the max for color scaling
                showscale=True  # Ensure that the colorscale is shown
            ),
            text=[hover_text],
            hoverinfo='text',
            showlegend=False,
        ))


    # Create layout
    layout = go.Layout(
        title='Triangle Plot',
        xaxis=dict(title='X-axis', range=[-0.1, 1.1]),
        yaxis=dict(title='Y-axis', range=[-0.1, np.sqrt(3)/2 + 0.1]),
        showlegend=True
    )

    # Plot
    fig = go.Figure(data=[triangle_edges, policy_vertices] + new_points, layout=layout)
    fig.show()


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

def create_new_TD3_agent_and_evaluate(cfg, env_agent, theta):

    # Create the agent
    obs_size, act_size = env_agent.get_obs_and_actions_sizes()
    # TO CHANGE ############################################################################################################
    print(globals()) # voir c'est quoi le nom de l'agent
    #########################################################################################################################
    policy = globals()["DiscreteQAgent"](
        obs_size, cfg.algorithm.architecture.critic_hidden_sizes, act_size
    )
    ev_agent = Agents(env_agent, policy)
    eval_agent = TemporalAgent(ev_agent)

    # Calculate the reward
    torch.nn.utils.vector_to_parameters(theta, eval_agent.parameters())

    workspace = Workspace()

    eval_agent(workspace, t=0, stop_variable="env/done", render=False)

    rewards = workspace["env/cumulated_reward"][-1]
    mean_reward = rewards.mean()

    return mean_reward


def intersection_point(m, b, x_coord):
    "Calculates the intersection point of an y = mx'+b axis with the x = x_coord axis"
    # Calculate y-coordinate of intersection point
    y = m * (x_coord) + b
    return x_coord, y

def find_axis_through_point(point, slope):
    # Point coordinates
    x_h, y_h = point
    
    # Equation of the axis passing through H: y = mx + b
    # We know that the slope is the same as the given slope
    # So, b = y - mx
    b = y_h - slope * x_h
    
    # Return the equation of the axis: y = slope * x + b
    return slope, b

def generate_left_edge_points(num_points):
    # Generate x values linearly spaced between 0 and 1/2
    x_values = np.linspace(0, 0.5, num_points)
    
    # Calculate corresponding y values based on the equation sqrt(3) * x
    y_values = np.sqrt(3) * x_values
    
    # Combine x and y values into a list of points
    points = list(zip(x_values, y_values))
    
    return points

def generate_lower_edge_points(num_points):
    # Generate x values linearly spaced between 0 and 1
    x_values = np.linspace(0, 1, num_points)
    
    # Set y values to zero for all points
    y_values = np.zeros_like(x_values)
    
    # Combine x and y values into a list of points
    points = list(zip(x_values, y_values))
    
    return points

def get_alphas_from_point(x,y):
    triangle_vertices = [[0, 0],[1, 0], [0.5, np.sqrt(3)/2]]
    #print("triangle_vertices : ", triangle_vertices)
    A = np.vstack([triangle_vertices[0], triangle_vertices[1], triangle_vertices[2]]).T
    #print("A : ", A)

    A = np.vstack([A, np.ones(3)])

    b = np.array([x, y, 1])  

    # Résolution du système d'équations
    alphas = np.linalg.lstsq(A, b, rcond=None)[0]

    # Normalisation pour s'assurer que les alphas sont positifs ou nuls et somment à 1
    alphas = np.maximum(alphas, 0)  # Les valeurs négatives deviennent 0
    alphas /= np.sum(alphas)  # Normalisation pour que la somme soit égale à 1

    #print("alphas: ",alphas)

    return alphas

def is_inside_triangle(point, A, B, C):

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(point, A, B) < 0.0
    b2 = sign(point, B, C) < 0.0
    b3 = sign(point, C, A) < 0.0

    return ((b1 == b2) and (b2 == b3))

def policies_visualization(cfg, env_agent, num_points, loaded_agents):

    #1. Calculating alphas

    points = generate_left_edge_points(num_points)
    points2 = generate_lower_edge_points(num_points)

    axis_equations_points = []
    i=0
    slope = -np.sqrt(3)/3
    for point in points:
        new_slope, b = find_axis_through_point(point,slope)
        axis_equations_points.append((new_slope, b))

    intersection = []
    i = 0
    j = 0
    for i in range(len(axis_equations_points)):
        for j in range(len(points2)):
            slope, b = axis_equations_points[i]
            x_coord = points2[j][0]
            intersection.append(intersection_point(slope, b, x_coord))

    #print(intersection)

    # Extract x and y coordinates from points
    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]

    # Extract x and y coordinates from points2
    x_points2 = [point[0] for point in points2]
    y_points2 = [point[1] for point in points2]

    # Extract x and y coordinates from intersection
    x_intersection = [point[0] for point in intersection]
    y_intersection = [point[1] for point in intersection]

    alphas = []
    i = 0
    for i in range(len(x_points)):
        if(is_inside_triangle([x_points[i],y_points[i]], [0, 0], [1, 0], [0.5, np.sqrt(3)/2])):
            alphas.append(get_alphas_from_point(x_points[i],y_points[i]))
    j = 0
    for j in range(len(x_points2)):
        if(is_inside_triangle([x_points2[j],y_points2[j]], [0, 0], [1, 0], [0.5, np.sqrt(3)/2])):
            alphas.append(get_alphas_from_point(x_points2[j],y_points2[j]))
    k = 0
    for k in range(len(x_intersection)):
        if(is_inside_triangle([x_intersection[k],y_intersection[k]], [0, 0], [1, 0], [0.5, np.sqrt(3)/2])):
            alphas.append(get_alphas_from_point(x_intersection[k],y_intersection[k]))

    #print(len(alphas))

    #2. Calculating rewards
    rewards_list = []
    cpt=0
    print(len(alphas))
    for alpha in alphas:
        #print(alpha)
        theta = update_policy_with_coefficients(loaded_agents, alpha) # a1P1+a2P2+a3P3
        rewards_list.append(create_new_TD3_agent_and_evaluate(cfg, env_agent, theta))
        cpt+=1
        print(cpt)


    #3. Plotting the triangle with the points
    plot_triangle_with_multiple_points_plotly(alphas, rewards_list)
    plot_triangle_with_multiple_points(alphas, rewards_list)



@hydra.main(
    config_path="../algos/td3/configs/",
    config_name="td3_swimmer.yaml",
    #config_name="dqn_lunar_lander.yaml", #cartpole
)  # , version_base="1.3")

def main(cfg_raw: DictConfig):
    #print(read_tensor_file())

    date = "2024-03-23" 
    time = "19-05-08"
    get_best_policy(date, time, 1)

    date = "2024-03-23" 
    time = "19-08-30"
    get_best_policy(date, time, 2)

    date = "2024-03-23" 
    time = "19-20-13"
    get_best_policy(date, time, 3)

    loaded_agents = load_best_agent()
    print(loaded_agents)

    #new_theta = update_policy_with_coefficients(load_policies(loaded_agents), [0.5, 0.3, 0.2])
    #print(new_theta)
    
    # A METTRE LA METHODE NOUS PERMETTANT DE FAIRE CA DANS TD3.PY ##################################################################################
    _, eval_env_agent = local_get_env_agents(cfg_raw)
    ################################################################################################################################################
    
    #reward = create_new_DQN_agent_and_evaluate(cfg_raw, eval_env_agent, new_theta) 
    #print(reward)

    policies_visualization(cfg_raw, eval_env_agent, 80, load_policies(loaded_agents))


if __name__ == "__main__":
    main()

