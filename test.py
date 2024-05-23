import os
import sys
import argparse
from modules.ppo_agent import PPOAgent
from modules.hex_env import HexEnv
from datetime import datetime


def main(args):
    
    # create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print(f"Starting training at {timestamp}")
    
    # assign the arguments to the variables
    board_size = args.board_size
    model = args.model
    model_path = f"./models/{board_size}x{board_size}/{timestamp}_{model}"
    model_file_path = f"{model_path}.zip"

    agent = PPOAgent()

    # Load if you have a trained model
    env = None

    if os.path.exists(model_file_path):
        
        print("Loading model")
        agent.load(model_path)
        opponent_policy = agent.get_random_action if args.against_random == "true" else agent.get_action
        env = HexEnv(size=board_size, opponent_policy=opponent_policy)
        agent.set_env(env)
    
    else:
        print("Creating new model")
        env = HexEnv(size=board_size, opponent_policy=agent.get_random_action)
        agent.set_env(env)
        agent.init_model()

    if args.skip_training == "false":
        print("Training model")
        agent.train(args.training_steps)

        print("Saving model")
        agent.save(model_path)

    agent.evaluate_games(args.evaluation_steps)

## TODOS:
# Timestamp,
# evaluate models after training,
# only better models are saved,
# sample action selection from previous models,
# extrend step function to allow for multiple games to be played


if __name__ == "__main__":
    
    # define the parser
    parser = argparse.ArgumentParser(description='Training PPO Agent')
    
    # Add argument for board size
    parser.add_argument('-b', '--board_size', type=int, default=4, help='Size of the board')
    
    # Add argument for model
    parser.add_argument('-m', '--model', type=str, default='ppo', help='Model to use')
    
    # Add argument for training steps
    parser.add_argument('-ts', '--training_steps', type=int, default=200000, help='Number of training steps')
    
    # Add argument if training should be skipped
    parser.add_argument('-st', '--skip_training', type=str, default="false", help='Skip training')
    
    # Add argument for evaluation steps
    parser.add_argument('-es', '--evaluation_steps', type=int, default=100, help='Number of evaluation steps')
    
    # Add argument if agent should play against random
    parser.add_argument('-ar', '--against_random', type=str, default="false", help='Play against random')
    
    
    # parse the arguments
    args = parser.parse_args()
    
    # call the main function
    main(args)