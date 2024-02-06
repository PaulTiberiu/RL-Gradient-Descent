import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn


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
        distance_values = [value/10 for value in distance_values]

        num_bins = 20

        # Plot the histogramm 
        plt.figure(figsize=(10, 6))
        plt.hist(distance_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs de la distance')
        plt.xlabel('Intervalles de valeurs pris par la distance')
        plt.ylabel('Nombre de valeurs par intervalle')
        #plt.show()

        save_directory = os.path.join(script_directory, "..", "..", "..", "visualization_results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Then save the plot using the save_directory
        save_path = os.path.join(save_directory, "distance_histogram.png")
        plt.savefig(save_path)

    if(filename == "gradient_norm.txt"):
        grad_file_path = os.path.join(script_directory, filename)

        with open(grad_file_path, "r") as grad_file:
            grad_data = [float(line.strip()) for line in grad_file.readlines()]

        grad_values = [float(value) for value in grad_data]
        grad_values = [value/10 for value in grad_values]

        num_bins = 20

        plt.figure(figsize=(10, 6))
        plt.hist(grad_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs du gradient')
        plt.xlabel('Intervalles de valeurs pris par le gradient')
        plt.ylabel('Nombre de valeurs par intervalle')
        #plt.show()

        save_directory = os.path.join(script_directory, "..", "..", "..", "visualization_results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Then save the plot using the save_directory
        save_path = os.path.join(save_directory, "gradient_histogram.png")
        plt.savefig(save_path)


    if(filename == "loss_values.txt"):
        loss_file_path = os.path.join(script_directory, filename)

        with open(loss_file_path, "r") as loss_file:
            #loss_data = [line.split(", ")[1].split(": ")[1] for line in loss_file.readlines()]
            loss_data = [float(line.strip()) for line in loss_file.readlines()]

        loss_values = [float(value) for value in loss_data]
        loss_values = [value/10 for value in loss_values]

        num_bins = 20

        plt.figure(figsize=(10, 6))
        plt.hist(loss_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs de la loss')
        plt.xlabel('Intervalles de valeurs pris par la loss')
        plt.ylabel('Nombre de valeurs par intervalle')
        #plt.show()

        save_directory = os.path.join(script_directory, "..", "..", "..","visualization_results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Then save the plot using the save_directory
        save_path = os.path.join(save_directory, "loss_histogram.png")
        plt.savefig(save_path)

    # Connect the on_mouse_move function to the figure's event manager
    # plt.gcf().canvas.mpl_connect('motion_notify_event', on_mouse_move)

    # Keep the plot interactive
    # plt.show(block=True)

    # plt.show()

"""
def on_mouse_move(event):
    global data
    if event.inaxes is not None:
        if event.button == 'left' and event.xdata is not None:
            new_data = np.random.randn(100) + event.xdata
            update_histogram(new_data)


def update_histogram(new_data):
    global counts, bins
    new_counts, _ = np.histogram(new_data, bins=bins)
    counts += new_counts
    for rect, h in zip(plt.gca().patches, counts):
        rect.set_height(h)
    plt.gca().relim()
    plt.gca().autoscale_view()
    plt.draw()
"""
