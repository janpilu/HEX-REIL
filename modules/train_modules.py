import numpy as np
import os

from datetime import datetime
from scipy.stats import norm
from modules.ppo_agent import PPOAgent


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

    # Filter out files that are not models (not ending with .zip)
    model_files = [f for f in files if f.endswith(".zip")]

    # Sort the model files based on the timestamp in their name
    sorted_models = sorted(
        model_files,
        reverse=True,
    )

    return sorted_models


def get_policies(folder_path, include_most_recent=False):

    sorted_models = get_sorted_models(folder_path)
    opponent_policies = []
    model_files = []

    if not include_most_recent and len(sorted_models) > 1:
        sorted_models = sorted_models[1:]

    if len(sorted_models) >= 1:
        for model in sorted_models:
            agent = PPOAgent()
            agent.load(f"{folder_path}/{model}")
            opponent_policies.append(agent.get_action)
            model_files.append(model)

    else:
        agent = PPOAgent()
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
    ](board, action_set)
