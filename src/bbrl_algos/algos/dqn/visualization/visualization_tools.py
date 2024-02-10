import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import plotly.graph_objects as go
from datetime import datetime


"""
def calculate_euclidean_distance(policy1, policy2):
    distance = 0.0
    for i in range (len(policy1)):
        distance += torch.norm(policy1[i] - policy2[i], p=2) #on fait torch.norm ou vraiment (policy1[i] - policy2[i])**2

    return distance
"""

def calculate_euclidean_distance(policy1, policy2):
    distance = torch.norm(policy1 - policy2, p=2)
    return distance.item()  # Convert to Python float


def calculate_gradient_normv2(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = torch.norm(param.grad.data, p=2)
            total_norm += param_norm.item()

    # or grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in model.parameters()]))

    return total_norm


def calculate_gradient_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = torch.norm(param.grad.data, p=2)
            total_norm += param_norm.item() ** 2 # Add the squared norm
    total_norm = total_norm ** 0.5 # Take the square root of the sum of squared norms

    # or grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in model.parameters()]))

    return total_norm

def write_in_file(filename, data):

    if(filename == "loss_values.txt"):
        # Path to the visualisation folder
        #visualization_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization")

        loss_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loss_values.txt")

        with open(loss_file_path, "a") as loss_file:
            loss_file.write(f"{data}\n")

    if(filename == "distances.txt"):
        # Path to the visualisation folder
        #visualization_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_files")

        distances_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "distances.txt")


        with open(distances_file_path, "a") as distances_file:
            distances_file.write(f"{data}\n")

    if(filename == "gradient_norm.txt"):
        # Path to the visualisation folder
        #visualization_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization")

        gradient_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradient_norm.txt")

        with open(gradient_file_path, "a") as gradient_file:
            gradient_file.write(f"{data}\n")

    if(filename == "facteurs_k.txt"):
        # Path to the visualisation folder
        #visualization_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization")

        k_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "facteurs_k.txt")

        with open(k_file_path, "a") as k_file:
            k_file.write(f"{data}\n")

    if(filename == "differences_grad_distances.txt"):
        # Path to the visualisation folder
        #visualization_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization")

        diff_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "differences_grad_distances.txt")

        with open(diff_file_path, "a") as diff_file:
            diff_file.write(f"{data}\n")

    

def delete_file(filename):
    # Get the absolute path of the file
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    # Check if the file exists
    if os.path.exists(file_path):
        # If the file exists, delete it
        os.remove(file_path)
    

def dynamic_histograms(filename):
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative path to the file
    if filename == "distances.txt":
        distance_file_path = os.path.join(script_directory, filename)

        # Read the file
        with open(distance_file_path, "r") as distance_file:
            distance_data = [float(line.strip()) for line in distance_file.readlines()]

        # Create histogram
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=distance_data))

        # Update layout
        fig.update_layout(
            title='Répartition des valeurs de la distance',
            xaxis=dict(title='Intervalles de valeurs pris par la distance'),
            yaxis=dict(title='Nombre de valeurs par intervalle'),
            bargap=0.05,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinates
        )

        # Show plot
        fig.show()



    elif filename == "gradient_norm.txt":
        grad_file_path = os.path.join(script_directory, filename)

        with open(grad_file_path, "r") as grad_file:
            grad_data = [float(line.strip()) for line in grad_file.readlines()]

        # Create histogram
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=grad_data))

        # Update layout
        fig.update_layout(
            title='Répartition des valeurs du gradient',
            xaxis=dict(title='Intervalles de valeurs pris par le gradient'),
            yaxis=dict(title='Nombre de valeurs par intervalle'),
            bargap=0.05,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinates
        )

        # Show plot
        fig.show()

    elif filename == "loss_values.txt":
        loss_file_path = os.path.join(script_directory, filename)

        with open(loss_file_path, "r") as loss_file:
            loss_data = [float(line.strip()) for line in loss_file.readlines()]

        # Create histogram
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=loss_data))

        # Update layout
        fig.update_layout(
            title='Répartition des valeurs de la loss',
            xaxis=dict(title='Intervalles de valeurs pris par la loss'),
            yaxis=dict(title='Nombre de valeurs par intervalle'),
            bargap=0.05,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinates
        )

        # Show plot
        fig.show()
    
    elif filename == "facteurs_k.txt":
        k_file_path = os.path.join(script_directory, filename)

        with open(k_file_path, "r") as k_file:
            k_data = [float(line.strip()) for line in k_file.readlines()]

        # Create histogram
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=k_data))

        # Update layout
        fig.update_layout(
            title='Répartition des valeurs du facteur k',
            xaxis=dict(title='Intervalles de valeurs pris par le facteur k'),
            yaxis=dict(title='Nombre de valeurs par intervalle'),
            bargap=0.05,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinates
        )

        # Show plot
        fig.show()

    elif filename == "differences_grad_distances.txt":
        diff_file_path = os.path.join(script_directory, filename)

        with open(diff_file_path, "r") as diff_file:
            diff_data = [float(line.strip()) for line in diff_file.readlines()]

        # Create histogram
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=diff_data))

        # Update layout
        fig.update_layout(
            title='Répartition des valeurs de la difference entre le facteur k et learning rate',
            xaxis=dict(title='Intervalles de valeurs pris par la difference entre le facteur k et learning rate'),
            yaxis=dict(title='Nombre de valeurs par intervalle'),
            bargap=0.05,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinates
        )

        # Show plot
        fig.show()



def histograms(filename):

    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative path to the file
    if(filename == "distances.txt"):
        distance_file_path = os.path.join(script_directory, filename)

        # Read the file
        with open(distance_file_path, "r") as distance_file:
            distance_data = [float(line.strip()) for line in distance_file.readlines()]

        # Convert the distance values
        distance_values = [float(value) for value in distance_data]
        #distance_values = [value/10 for value in distance_values]

        num_bins = 20

        # Plot the histogramm 
        plt.figure(figsize=(10, 6))
        plt.hist(distance_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs de la distance')
        plt.xlabel('Intervalles de valeurs pris par la distance')
        plt.ylabel('Nombre de valeurs par intervalle')
        #plt.show()

        # Get current date and time
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        save_directory = os.path.join(script_directory, "..", "..", "..", "visualization_results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Then save the plot using the save_directory
        save_path = os.path.join(save_directory, f"{filename.split('.')[0]}_{date_time}_histogram.png")
        plt.savefig(save_path)
        plt.show()

    if(filename == "gradient_norm.txt"):
        grad_file_path = os.path.join(script_directory, filename)

        with open(grad_file_path, "r") as grad_file:
            grad_data = [float(line.strip()) for line in grad_file.readlines()]

        grad_values = [float(value) for value in grad_data]
        #grad_values = [value/10 for value in grad_values]

        num_bins = 20

        plt.figure(figsize=(10, 6))
        plt.hist(grad_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs du gradient')
        plt.xlabel('Intervalles de valeurs pris par le gradient')
        plt.ylabel('Nombre de valeurs par intervalle')
        #plt.show()

        # Get current date and time
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        save_directory = os.path.join(script_directory, "..", "..", "..", "visualization_results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Then save the plot using the save_directory
        save_path = os.path.join(save_directory, f"{filename.split('.')[0]}_{date_time}_histogram.png")
        plt.savefig(save_path)
        plt.show()

    if(filename == "loss_values.txt"):
        loss_file_path = os.path.join(script_directory, filename)

        with open(loss_file_path, "r") as loss_file:
            loss_data = [float(line.strip()) for line in loss_file.readlines()]

        loss_values = [float(value) for value in loss_data]
        #grad_values = [value/10 for value in grad_values]

        num_bins = 20

        plt.figure(figsize=(10, 6))
        plt.hist(loss_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs de la loss')
        plt.xlabel('Intervalles de valeurs pris par la loss')
        plt.ylabel('Nombre de valeurs par intervalle')
        #plt.show()

        # Get current date and time
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        save_directory = os.path.join(script_directory, "..", "..", "..", "visualization_results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Then save the plot using the save_directory
        save_path = os.path.join(save_directory, f"{filename.split('.')[0]}_{date_time}_histogram.png")
        plt.savefig(save_path)
        plt.show()


    if(filename == "facteurs_k.txt"):
        k_file_path = os.path.join(script_directory, filename)

        with open(k_file_path, "r") as k_file:
            #loss_data = [line.split(", ")[1].split(": ")[1] for line in loss_file.readlines()]
            k_data = [float(line.strip()) for line in k_file.readlines()]

        k_values = [float(value) for value in k_data]
        #loss_values = [value/10 for value in loss_values]

        num_bins = 20

        plt.figure(figsize=(10, 6))
        plt.hist(k_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs du facteur k')
        plt.xlabel('Intervalles de valeurs pris par le facteur k')
        plt.ylabel('Nombre de valeurs par intervalle')
        #plt.show()

        # Get current date and time
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        save_directory = os.path.join(script_directory, "..", "..", "..", "visualization_results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Then save the plot using the save_directory
        save_path = os.path.join(save_directory, f"{filename.split('.')[0]}_{date_time}_histogram.png")
        plt.savefig(save_path)
        plt.show()

    if(filename == "differences_grad_distances.txt"):
        diff_file_path = os.path.join(script_directory, filename)

        with open(diff_file_path, "r") as loss_file:
            #loss_data = [line.split(", ")[1].split(": ")[1] for line in loss_file.readlines()]
            diff_data = [float(line.strip()) for line in loss_file.readlines()]

        diff_values = [float(value) for value in diff_data]
        #loss_values = [value/10 for value in loss_values]

        num_bins = 20

        plt.figure(figsize=(10, 6))
        plt.hist(diff_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs de la difference entre le facteur k et learning rate')
        plt.xlabel('Intervalles de valeurs pris par la difference entre le facteur k et learning rate')
        plt.ylabel('Nombre de valeurs par intervalle')
        #plt.show()

        # Get current date and time
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        save_directory = os.path.join(script_directory, "..", "..", "..", "visualization_results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Then save the plot using the save_directory
        save_path = os.path.join(save_directory, f"{filename.split('.')[0]}_{date_time}_histogram.png")
        plt.savefig(save_path)
        plt.show()

    plt.show()


def is_grad_norm_proportional_to_distance(cfg):
    
    eps = 0.02
    script_directory = os.path.dirname(os.path.abspath(__file__))

    distance_file_path = os.path.join(script_directory, "distances.txt")

    with open(distance_file_path, "r") as distance_file:
        distance_data = [float(line.strip()) for line in distance_file.readlines()]
    
    distance_values = [float(value) for value in distance_data]

    grad_file_path = os.path.join(script_directory, "gradient_norm.txt")

    with open(grad_file_path, "r") as grad_file:
        grad_data = [float(line.strip()) for line in grad_file.readlines()]
    
    grad_values = [float(value) for value in grad_data]

    for i in range(len(distance_values)):
        k = distance_values[i] / grad_values[i]
        write_in_file("facteurs_k.txt", k)
        write_in_file("differences_grad_distances.txt", abs(k - cfg.optimizer.lr))


        if(abs(k - cfg.optimizer.lr) > eps):
            return False
            #print(abs(k - cfg.optimizer.lr))

    return True
    
