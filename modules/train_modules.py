import numpy as np
import os

from datetime import datetime
from scipy.stats import norm
from modules.agents.dql_agent import DQLAgent
from modules.agents.ppo_agent import PPOAgent
import config
from modules.hex_env import HexEnv


def gaussian_probabilities(n, shift=2):
    # Generate an array of indices
    x = np.linspace(-3, 3, n)  # -3 and 3 are arbitrary bounds for the Gaussian
    # Shift the mean of the Gaussian to the right
    x = x - np.min(x) - shift
    # Generate Gaussian distribution values for these indices
    probabilities = norm.pdf(x)
    # Normalize the probabilities so they sum to 1
    probabilities /= probabilities.sum()
    return probabilities


def get_sorted_models(folder_path):
    # Get all files in the directory
    files = os.listdir(folder_path)

    fixed_files = []

    for file in files:
        # Files that end with -actor -critic
        suffix = file.split("-")[-1]
        if suffix == "actor" or suffix == "critic":
            file = file.replace(f"-{suffix}", "")
        fixed_files.append(file)

    model_files = list(set(fixed_files))

    # Sort the model files based on the timestamp in their name
    sorted_models = sorted(
        model_files,
        reverse=True,
    )

    return sorted_models


def get_policies(folder_path, agent_class, env, include_most_recent=False):

    sorted_models = get_sorted_models(folder_path)
    opponent_policies = []
    model_files = []

    if not include_most_recent and len(sorted_models) > 1:
        sorted_models = sorted_models[1:]

    if len(sorted_models) >= 1:
        for model in sorted_models:
            agent = agent_class(
                hidden_layers=config.hidden_layers,
                hidden_size=config.hidden_size,
                use_conv=config.use_conv,
            )
            agent.set_env(env)
            agent.load(f"{folder_path}/{model}")
            opponent_policies.append(agent.get_action)
            model_files.append(model)

    else:
        agent = agent_class(
            hidden_layers=config.hidden_layers,
            hidden_size=config.hidden_size,
            use_conv=config.use_conv,
        )
        agent.load(f"{folder_path}/{sorted_models[0]}")
        opponent_policies.append(agent.get_action)
        model_files.append(sorted_models[0])

    return opponent_policies, model_files


def get_opponent_policy(policies, model_files, number_of_policies=10):
    number_of_policies = min(number_of_policies, len(policies))
    probabilities = gaussian_probabilities(len(policies))

    indices = np.random.choice(
        len(policies), number_of_policies, p=probabilities, replace=False
    )
    selected_policies = [policies[i] for i in indices]
    selected_model_files = [model_files[i] for i in indices]

    print(f"  --- Sampling from {model_files}")
    print(f"  --- Using {number_of_policies} policies")
    print(f"  --- Selected models: {selected_model_files}\n")

    return lambda board, action_set, current_game: selected_policies[
        (current_game // 2)
        % number_of_policies  # // 2 to ensure that the same policy is used for two games (both as white and black)
    ](board)


def get_agents(board_size, root_folder=None, as_path_agent_dict=False):
    if root_folder is None:
        root_folder = f"./models/{board_size}x{board_size}"
    os.makedirs(root_folder, exist_ok=True)
    env = HexEnv(size=board_size)

    agents = {}
    path_agent_dict = {}

    for model in config.models:
        model_folder = f"{root_folder}/{model}"
        os.makedirs(model_folder, exist_ok=True)
        agent_class = None

        if model == "dql":
            agent_class = DQLAgent

        elif model == "ppo":
            agent_class = PPOAgent

        agents[model] = {}

        for architecture_folder in os.listdir(model_folder):
            print(architecture_folder)
            architectre_path = f"{model_folder}/{architecture_folder}"
            if os.path.isdir(architectre_path):
                print(f"Loading {model} agents from {architecture_folder}")
                layers, size, architectre = architecture_folder.split("_")
                hidden_layers = int(layers)
                hidden_size = int(size)
                use_conv = architectre == "conv"

                agents[model][architecture_folder] = []

                if agent_class is not None:
                    for model_file in os.listdir(architectre_path):
                        agent = agent_class(
                            hidden_layers=hidden_layers,
                            hidden_size=hidden_size,
                            use_conv=use_conv,
                        )
                        agent.set_env(env)
                        agent.load(f"{architectre_path}/{model_file}")
                        agents[model][architecture_folder].append(
                            {"agent": agent, "path": f"{architectre_path}/{model_file}"}
                        )
                        path_agent_dict[f"{architectre_path}/{model_file}"] = agent

    if as_path_agent_dict:
        return path_agent_dict
    return agents
