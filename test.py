import os
from modules.ppo_agent import PPOAgent
from modules.hex_env import HexEnv

board_size = 7
model = "ppo"
model_path = f"models/{model}_{board_size}x{board_size}"
model_file_path = f"{model_path}.zip"

agent = PPOAgent()

# Load if you have a trained model

env = HexEnv(size=board_size, opponent_policy=agent.get_action)
agent.set_env(env)

if os.path.exists(model_file_path):
    print("Loading model")
    agent.load(model_path)
else:
    print("Creating new model")
    agent.init_model()

print("Training model")
agent.train(100)

print("Saving model")
agent.save(model_path)
