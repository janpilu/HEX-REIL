import os
from modules.ppo_agent import PPOAgent
from modules.hex_env import HexEnv

board_size = 7
model = "ppo"
model_path = f"models/{board_size}x{board_size}/{model}"  # Add timestamp
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
agent.train(100000)

print("Saving model")
agent.save(model_path)

agent.evaluate(1000)

## TODOS:
# Timestamp,
# evaluate models after training,
# only better models are saved,
# sample action selection from previous models,
# extrend step function to allow for multiple games to be played
