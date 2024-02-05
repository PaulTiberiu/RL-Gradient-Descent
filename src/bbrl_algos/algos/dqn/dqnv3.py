#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import copy
import os
import numpy as np
from typing import Callable, List

import hydra
#import optuna
from omegaconf import DictConfig

# %%
import torch
import torch.nn as nn

# %%
import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import AutoResetWrapper

# %%
from bbrl import get_arguments, get_class
from bbrl.agents import TemporalAgent, Agents, PrintAgent
from bbrl.workspace import Workspace

from bbrl_algos.models.exploration_agents import EGreedyActionSelector
from bbrl_algos.models.critics import DiscreteQAgent
from bbrl_algos.models.loggers import Logger
from bbrl_algos.models.utils import save_best

from bbrl.visu.plot_critics import plot_discrete_q, plot_critic
from bbrl_algos.models.hyper_params import launch_optuna

from bbrl.utils.chrono import Chrono

# HYDRA_FULL_ERROR = 1
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from bbrl_gymnasium.envs.maze_mdp import MazeMDPEnv
from bbrl_algos.wrappers.env_wrappers import MazeMDPContinuousWrapper
from bbrl.agents.gymnasium import make_env, ParallelGymAgent
from functools import partial


matplotlib.use("TkAgg")


def local_get_env_agents(cfg):
    eval_env_agent = ParallelGymAgent(
        partial(
            make_env,
            cfg.gym_env.env_name,
            autoreset=False,
        ),
        cfg.algorithm.nb_evals,
        include_last_state=True,
        seed=cfg.algorithm.seed.eval,
    )
    train_env_agent = ParallelGymAgent(
        partial(
            make_env,
            cfg.gym_env.env_name,
            autoreset=True,
        ),
        cfg.algorithm.n_envs,
        include_last_state=True,
        seed=cfg.algorithm.seed.train,
    )
    return train_env_agent, eval_env_agent


# %%
def compute_critic_loss(
    discount_factor, reward, must_bootstrap, action, q_values, q_target=None
):
    """Compute critic loss
    Args:
        discount_factor (float): The discount factor
        reward (torch.Tensor): a (2 × T × B) tensor containing the rewards
        must_bootstrap (torch.Tensor): a (2 × T × B) tensor containing 0 if the episode is completed at time $t$
        action (torch.LongTensor): a (2 × T) long tensor containing the chosen action
        q_values (torch.Tensor): a (2 × T × B × A) tensor containing Q values
        q_target (torch.Tensor, optional): a (2 × T × B × A) tensor containing target Q values

    Returns:
        torch.Scalar: The loss
    """
    if q_target is None:
        q_target = q_values
    max_q = q_target[1].amax(dim=-1).detach()
    target = reward[1] + discount_factor * max_q * must_bootstrap[1]
    act = action[0].unsqueeze(dim=-1)
    qvals = q_values[0].gather(dim=1, index=act)
    qvals = qvals.squeeze(dim=1)
    return nn.MSELoss()(qvals, target)


# %%
def create_dqn_agent(cfg_algo, train_env_agent, eval_env_agent):
    # obs_space = train_env_agent.get_observation_space()
    # obs_shape = obs_space.shape if len(obs_space.shape) > 0 else obs_space.n

    # act_space = train_env_agent.get_action_space()
    # act_shape = act_space.shape if len(act_space.shape) > 0 else act_space.n

    state_dim, action_dim = train_env_agent.get_obs_and_actions_sizes()

    critic = DiscreteQAgent(
        state_dim=state_dim,
        hidden_layers=list(cfg_algo.architecture.hidden_sizes),
        action_dim=action_dim,
        seed=cfg_algo.seed.q,
    )

    explorer = EGreedyActionSelector(
        name="action_selector",
        epsilon=cfg_algo.explorer.epsilon_start,
        epsilon_end=cfg_algo.explorer.epsilon_end,
        epsilon_decay=cfg_algo.explorer.decay,
        seed=cfg_algo.seed.explorer,
    )
    q_agent = TemporalAgent(critic)

    tr_agent = Agents(train_env_agent, critic, explorer)  # , PrintAgent())
    ev_agent = Agents(eval_env_agent, critic)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)

    return train_agent, eval_agent, q_agent


# %%
# Configure the optimizer over the q agent
def setup_optimizer(optimizer_cfg, q_agent):
    optimizer_args = get_arguments(optimizer_cfg)
    parameters = q_agent.parameters()
    optimizer = get_class(optimizer_cfg)(parameters, **optimizer_args)
    return optimizer


# %%
def run_dqn(cfg, logger, trial=None):

    policy_distances = []

    best_reward = float("-inf")
    if cfg.collect_stats:
        directory = "./dqn_data/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + "dqn_" + cfg.gym_env.env_name + ".data"
        fo = open(filename, "wb")
        stats_data = []

    # 1) Create the environment agent
    train_env_agent, eval_env_agent = local_get_env_agents(cfg)

    # 2) Create the DQN-like Agent
    train_agent, eval_agent, q_agent = create_dqn_agent(
        cfg.algorithm, train_env_agent, eval_env_agent
    )

    # 3) Create the training workspace
    train_workspace = Workspace()  # Used for training

    # 5) Configure the optimizer
    optimizer = setup_optimizer(cfg.optimizer, q_agent)

    # 6) Define the steps counters
    nb_steps = 0
    tmp_steps_eval = 0

    prev_policy = None

    while nb_steps < cfg.algorithm.n_steps:
        # Decay the explorer epsilon
        explorer = train_agent.agent.get_by_name("action_selector")
        assert len(explorer) == 1, "There should be only one explorer"
        explorer[0].decay()

        # Execute the agent in the workspace
        if nb_steps > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace,
                t=1,
                n_steps=cfg.algorithm.n_steps_train - 1,
            )
        else:
            train_agent(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps_train,
            )

        transition_workspace: Workspace = train_workspace.get_transitions(
            filter_key="env/done"
        )

        # Only get the required number of steps
        steps_diff = cfg.algorithm.n_steps - nb_steps
        if transition_workspace.batch_size() > steps_diff:
            for key in transition_workspace.keys():
                transition_workspace.set_full(
                    key, transition_workspace[key][:, :steps_diff]
                )

        nb_steps += transition_workspace.batch_size()

        # The q agent needs to be executed on the rb_workspace workspace (gradients are removed in workspace).
        q_agent(transition_workspace, t=0, n_steps=2, choose_action=False)

        q_values, terminated, reward, action = transition_workspace[
            "critic/q_values",
            "env/terminated",
            "env/reward",
            "action",
        ]

        # Determines whether values of the critic should be propagated
        # True if the task was not terminated.
        must_bootstrap = ~terminated

        critic_loss = compute_critic_loss(
            cfg.algorithm.discount_factor,
            reward,
            must_bootstrap,
            action,
            q_values,
        )

        # Store the loss
        logger.add_log("critic_loss", critic_loss, nb_steps)

        # Open a file for appending the critic loss values
        loss_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loss_values.txt")
        with open(loss_file_path, "a") as loss_file:
            loss_file.write(f"Step: {nb_steps}, Critic Loss: {critic_loss.item()}\n")
        '''
        with open(loss_file_path, "r") as loss_file:
            loss_data = [line.split(", ")[1].split(": ")[1] for line in loss_file.readlines()]

# Convertir les valeurs de perte en nombres flottants
        loss_values = [float(value) for value in loss_data]
        loss_values = [value/10 for value in loss_values]

        num_bins = 20

        # Tracer l'histogramme de la répartition des valeurs de perte
        plt.figure(figsize=(10, 6))
        plt.hist(loss_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs de la loss')
        plt.xlabel('Intervalles de valeurs pris par la loss')
        plt.ylabel('Nombre de valeurs par intervalle')
        plt.show()
	'''

        optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            q_agent.parameters(), cfg.algorithm.max_grad_norm
        )

        optimizer.step()

        # Evaluate the agent
        if nb_steps - tmp_steps_eval > cfg.algorithm.eval_interval:
            tmp_steps_eval = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done",
                choose_action=True,
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            logger.log_reward_losses(rewards, nb_steps)
            mean = rewards.mean()

            if mean > best_reward:
                best_reward = mean

            print(
                f"nb_steps: {nb_steps}, reward: {mean:.02f}, best: {best_reward:.02f}"
            )

            if trial is not None:
                trial.report(mean, nb_steps)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if cfg.save_best and best_reward == mean:
                save_best(
                    eval_agent,
                    cfg.gym_env.env_name,
                    best_reward,
                    "./dqn_best_agents/",
                    "dqn",
                )
                if cfg.plot_agents:
                    critic = eval_agent.agent.agents[1]
                    plot_discrete_q(
                        critic,
                        eval_env_agent,
                        best_reward,
                        "./dqn_plots/",
                        cfg.gym_env.env_name,
                        input_action="policy",
                    )
                    plot_discrete_q(
                        critic,
                        eval_env_agent,
                        best_reward,
                        "./dqn_plots2/",
                        cfg.gym_env.env_name,
                        input_action=None,
                    )
            if cfg.collect_stats:
                stats_data.append(rewards)

            if trial is not None:
                trial.report(mean, nb_steps)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        
        # Backward pass and optimization
        optimizer.step()

        # Calculate Euclidean distance between successive policies
        if prev_policy is not None:
            current_policy = torch.nn.utils.parameters_to_vector(eval_agent.parameters())
            #print("prev_policy ", prev_policy)
            #print("current_policy", current_policy)
            distance = calculate_euclidean_distance(prev_policy, current_policy)
            #print("Distance between successive policies:", distance)
            policy_distances.append(distance)

        # Updating the previous policy
        prev_policy = torch.nn.utils.parameters_to_vector(eval_agent.parameters())

        grad_norm = calculate_gradient_norm(q_agent)
        print("Gradient Norm:", grad_norm)

        # Get the directory of the current script
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Construct the relative path to the file
        grad_file_path = os.path.join(script_directory, "gradient_norm.txt")

        # Save the file using the relative path
        with open(grad_file_path, "a") as grad_file:
            grad_file.write(f"{grad_norm}\n")
        '''
        with open(grad_file_path, "r") as grad_file:
            grad_data = [float(line.strip()) for line in grad_file.readlines()]

        # Convertir les valeurs de perte en nombres flottants
        grad_values = [float(value) for value in grad_data]
        grad_values = [value/10 for value in grad_values]

        num_bins = 20

        # Tracer l'histogramme de la répartition des valeurs du gradient
        plt.figure(figsize=(10, 6))
        plt.hist(grad_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs du gradient')
        plt.xlabel('Intervalles de valeurs pris par le gradient')
        plt.ylabel('Nombre de valeurs par intervalle')
        plt.show()
	'''



    if cfg.collect_stats:
        # All rewards, dimensions (# of evaluations x # of episodes)
        stats_data = torch.stack(stats_data, axis=-1)
        print(np.shape(stats_data))
        #print("policy_distances ", policy_distances)
        np.savetxt(filename, stats_data.numpy())
        fo.flush()
        fo.close()
        # Get the directory of the current script
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Construct the relative path to the file
        distance_file_path = os.path.join(script_directory, "distances.txt")
        
        with open(distance_file_path, "a") as distance_file:
            distance_file.write(f"{distance}\n")

        with open(distance_file_path, "r") as distance_file:
            distance_data = [float(line.strip()) for line in distance_file.readlines()]

        # Convertir les valeurs de distance en nombres flottants
        distance_values = [float(value) for value in distance_data]
        distance_values = [value/10 for value in distance_values]

        num_bins = 20

        # Tracer l'histogramme de la répartition des valeurs de la distance 
        plt.figure(figsize=(10, 6))
        plt.hist(distance_values, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Répartition des valeurs de la distance')
        plt.xlabel('Intervalles de valeurs pris par la distance')
        plt.ylabel('Nombre de valeurs par intervalle')
        plt.show()


    return best_reward

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
    """Calculate the L2 norm of gradients in the model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        float: The L2 norm of gradients.
    """

    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = torch.norm(param.grad.data, p=2)
            total_norm += param_norm.item() ** 2 # Add the squared norm
    total_norm = total_norm ** 0.5 # Take the square root of the sum of squared norms

    # or grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in model.parameters()]))

    return total_norm


# %%
@hydra.main(
    config_path="configs/",
    config_name="dqn_cartpole.yaml",
    #config_name="dqn_lunar_lander.yaml", #cartpole
)  # , version_base="1.3")
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_dqn)
    else:
        logger = Logger(cfg_raw)
        run_dqn(cfg_raw, logger)


if __name__ == "__main__":
    main()