import os
from modules.ppo_agent import PPOAgent
from modules.hex_env import HexEnv

board_size = 4
model = "ppo"
model_path = f"models/{board_size}x{board_size}/{model}"  # Add timestamp
model_file_path = f"{model_path}.zip"
skip_training = True
against_random = True

agent = PPOAgent()

# Load if you have a trained model

env = None

if os.path.exists(model_file_path):
    print("Loading model")
    agent.load(model_path)
    opponent_policy = agent.get_random_action if against_random else agent.get_action
    env = HexEnv(size=board_size, opponent_policy=opponent_policy)
    agent.set_env(env)
else:
    print("Creating new model")
    env = HexEnv(size=board_size, opponent_policy=agent.get_random_action)
    agent.set_env(env)
    agent.init_model()


if not skip_training:
    print("Training model")
    agent.train(200000)

    print("Saving model")
    agent.save(model_path)

agent.evaluate_games(100)

## TODOS:
# Timestamp,
# evaluate models after training,
# only better models are saved,
# sample action selection from previous models,
# extrend step function to allow for multiple games to be played
