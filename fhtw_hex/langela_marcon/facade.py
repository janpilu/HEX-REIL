# stupid example
import os

from .modules.hex_env import HexEnv
from .modules.agents.dql_agent import DQLAgent
from .modules.agents.ppo_agent import PPOAgent


def agent(board, action_set):
    agent_dir = "./langela_marcon/models/7x7/winner"
    agent = None
    for model in os.listdir(agent_dir):
        model_folder = f"{agent_dir}/{model}"
        for architecture_folder in os.listdir(model_folder):
            architectre_path = f"{model_folder}/{architecture_folder}"
            if os.path.isdir(architectre_path):
                layers, size, architectre = architecture_folder.split("_")
                hidden_layers = int(layers)
                hidden_size = int(size)
                use_conv = architectre == "conv"

                agent_class = DQLAgent if model == "dql" else PPOAgent

                files = os.listdir(architectre_path)

                fixed_files = []

                for file in files:
                    # Files that end with -actor -critic
                    suffix = file.split("-")[-1]
                    if suffix == "actor" or suffix == "critic":
                        file = file.replace(f"-{suffix}", "")
                    fixed_files.append(file)

                model_files = list(set(fixed_files))

                model_file = model_files[0]

                env = HexEnv(size=7)

                agent = agent_class(
                    hidden_layers=hidden_layers,
                    hidden_size=hidden_size,
                    use_conv=use_conv,
                )
                agent.set_env(env)
                agent.load(f"{architectre_path}/{model_file}")
    return agent.policy(board, action_set)


# Here should be the necessary Python wrapper for your model, in the form of a callable agent, such as above.
# Please make sure that the agent does actually work with the provided Hex module.
