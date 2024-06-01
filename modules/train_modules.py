import numpy as np
import os

from datetime import datetime
from scipy.stats import norm
from modules.ppo_agent import PPOAgent


def gaussian_probabilities(n):
    # Generate an array of indices
    x = np.linspace(-3, 3, n)  # -3 and 3 are arbitrary bounds for the Gaussian
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


def get_policies(folder_path):
    
    sorted_models = get_sorted_models(folder_path)
    opponent_policies = []
    
    # Ignore the first model as it is the most recent and used by the agent and later for evaluation
    if len(sorted_models) > 1:
    
        for model in sorted_models[1:]:
            agent = PPOAgent()
            agent.load(f"{folder_path}/{model}")
            opponent_policies.append(agent.get_action)
    
    else:
        agent = PPOAgent()
        agent.load(f"{folder_path}/{sorted_models[0]}")
        opponent_policies.append(agent.get_action)
        
    return opponent_policies


def get_opponent_policy(policies, number_of_policies=10):
    
    number_of_policies = min(number_of_policies, len(policies))

    print(f"  --- Using {number_of_policies} policies\n")

    probabilities = gaussian_probabilities(len(policies))

    selected_policies = [
        policies[i]
        for i in np.random.choice(len(policies), number_of_policies, p=probabilities)
    ]

    return lambda board, action_set, current_game: selected_policies[
        current_game % number_of_policies
    ](board, action_set)

