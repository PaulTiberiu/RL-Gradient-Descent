# Visualization Tools for Reinforcement Learning algorithms
This tool was completed for the P-ANDROIDE project during the first year of the Master's program in ANDROIDE computer science at Sorbonne University. Our project showcases two main components:

## Histograms
In this part, we are focusing on analyzing the loss, the norm of distances between two policies, and the norm of gradients within the framework of the gradient descent algorithm, specifically with the use of the DQN algorithm.

Using the histograms, we will be able to verify certain hypothesis such as proportionalities between, for example, the norm of the gradient is proportional to the distance between two successive policies. Examples of histograms (distances between 2 consecutive policies, the norm of gradients and loss values) can be seen below:
<img src="https://github.com/PaulTiberiu/RL-Gradient-Visualization-Tool/assets/123265734/4497a166-3c3c-4d3c-a901-cdd9cd14b5df" width="500">
<img src="https://github.com/PaulTiberiu/RL-Gradient-Visualization-Tool/assets/123265734/b59cab8e-2456-44f9-ab2d-1f4867ae89cb" width="500">
<img src="https://github.com/PaulTiberiu/RL-Gradient-Visualization-Tool/assets/123265734/de20094c-b07b-4f7c-b9f0-9481a16be674" width="500">

In order to get this results, you just need to execute the histograms.py script located at the following path: /src/bbrl_algos/visualization, that launches a run of the dqn algorithm and provides the histograms.

## Policies visualization tool
This tool will allow us to display the performance of policies within a selected subspace of the policy space.

Below is an illustration depicting policies with rewards of 500, using the cartpole environment and the dqn algorithm:
![policy_2024-03-20_12-44-23](https://github.com/PaulTiberiu/RL-Gradient-Visualization-Tool/assets/123265734/f8a1b8f7-2744-4083-8642-b1cd305200f0)

In order to enhance the interactivity of our plots, we utilized Plotly, a data visualization library. This enables us to create more sophisticated visualizations, such as adjusting the plot size and displaying the exact reward value and its corresponding weighted sum coefficients for each point on the plot.

In order to get this results, firstly you need to execute 3 times the dqnv2.py script(/src/bbrl_algos/algos/dqn), then check in the folder tmp/hydra for the date and time you executed the code. Then, you just need to modify the dates and times in the code of policies_visualization.py (path: /src/bbrl_algos/visualization) before executing it.

For multiple plots you can check the visualization_results folder of our Github.

# BBRL - ALGOS
Below, you can find information regarding the purpose and installation of the BBRL - ALGOS library, which our tool utilizes:

## Description

This library is designed for education purposes, it is mainly used to perform some practical experiences with various RL algorithms. It facilitates using optuna for tuning hyper-parameters and using rliable and statistical tests for analyzing the results.

## Installation

git clone https://github.com/osigaud/bbrl_algos.git

cd bbrl_algos

pip install -e .

We suggest using your favorite python environment (conda, venv, ...) as some further installations might be necessary

## Usage

go to src/bbrl_algos, choose your algorithm and run python3 your_algorithm.py
