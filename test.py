import os
import sys
import argparse
import numpy as np


from pathlib import Path
from modules.ppo_agent import PPOAgent
from modules.hex_env import HexEnv
from datetime import datetime

# load train module
import modules.train_modules as train_modules


def main(args):
    
    # create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Start training at {timestamp}")

    # assign the arguments to the variables
    board_size = args.board_size
    model = args.model
    
    # main folder for models
    os.makedirs("./models", exist_ok=True)
    
    model_folder = f"./models/{board_size}x{board_size}"
    os.makedirs(model_folder, exist_ok=True)
    
    model_path = f"{model_folder}/{timestamp}_{model}"

    models = train_modules.get_sorted_models(model_folder)
    most_recent_model = models[0] if len(models) > 0 else None

    if len(models) > 0 and not args.against_random:
        print("Loading models")
        policies = train_modules.get_policies(model_folder)
        opponent_policy = train_modules.get_opponent_policy(policies, args.number_of_policies)
    else:
        if not args.against_random:
            print("No models found, playing against random")
        else:
            print("Playing against random")
        opponent_policy = lambda board, action_set, current_game: np.random.choice(action_set)  # <-- Modified to accept three arguments

    agent = PPOAgent()

    # Load if you have a trained model

    env = HexEnv(size=board_size, opponent_policy=opponent_policy)

    if most_recent_model is not None:
        print("Loading most recent model")
        agent.load(f"{model_folder}/{most_recent_model}")
        agent.set_env(env)
        print(f"Model loaded, continuing training from {most_recent_model}")
        
        evaluation_agent = PPOAgent()
        evaluation_agent.load(f"{model_folder}/{most_recent_model}")
        evaluation_agent.set_env(env)
    
    else:
        print("Creating new model")
        agent.set_env(env)
        agent.init_model()
        
        # define random agent for evaluation
        evaluation_agent = PPOAgent()
        evaluation_agent.set_env(env)
        evaluation_agent.init_model()  # Ensure evaluation model is initialized

    score = -1
    training_round = 0
    

    while score < args.evaluation_threshold / 100:
        
        if training_round % args.resampling_threshold == 0 and training_round != 0:
            print(f"{args.resampling_threshold} rounds have passed without reaching threshold")
            print("Resampling opponent policy")
            opponent_policy = train_modules.get_opponent_policy(policies, args.number_of_policies)
            agent.set_opponent_policy(opponent_policy)
        
        training_round += 1
        
        if not args.skip_training:
            
            if score > -1:
                print(f"Score: {score*100:.2f}%, did not reach threshold of {args.evaluation_threshold}%")
            
            print("Training model")
            agent.train(args.training_steps)
            
        print(f"Evaluating model against most recent model ({most_recent_model})")
        score = agent.evaluate_games(args.evaluation_steps, evaluation_agent.get_action)

    print("Saving model")
    agent.save(model_path)


## TODOS:
# Timestamp,
# evaluate models after training,
# only better models are saved,
# sample action selection from previous models,
# extrend step function to allow for multiple games to be played


if __name__ == "__main__":

    # define the parser
    parser = argparse.ArgumentParser(description="Training PPO Agent")

    # Add argument for board size
    parser.add_argument(
        "-b", "--board_size", type=int, default=4, help="Size of the board"
    )

    # Add argument for model
    parser.add_argument(
        "-m", "--model", type=str, default="ppo", help="Model to use"
    )

    # Add argument for training steps
    parser.add_argument(
        "-ts",
        "--training_steps",
        type=int,
        default=20000,
        help="Number of training steps",
    )

    # Add argument if training should be skipped
    parser.add_argument(
        "-st",
        "--skip_training",
        default=False,
        help="Skip training",
        action="store_true",
    )

    # Add argument for evaluation steps
    parser.add_argument(
        "-es",
        "--evaluation_steps",
        type=int,
        default=100,
        help="Number of evaluation steps",
    )

    # Add argument for evaluation steps
    parser.add_argument(
        "-et",
        "--evaluation_threshold",
        type=int,
        default=60,
        help="Percent to reach before stopping training",
    )

    # Add argument for resampling threshold
    parser.add_argument(
        "-rt",
        "--resampling_threshold",
        type=int,
        default=5,
        help="Number of training rounds before resampling opponent policy",
    )

    # Add argument if agent should play against random
    parser.add_argument(
        "-ar",
        "--against_random",
        default=False,
        action="store_true",
        help="Play against random",
    )

    parser.add_argument(
        "-d",
        "--debug",
        default=True,
        action="store_true",
        help="Debug flag",
    )

    parser.add_argument(
        "-np",
        "--number_of_policies",
        type=int,
        default=10,
        help="Number of policies to use",
    )

    # parse the arguments
    args = parser.parse_args()

    # call the main function
    main(args)
