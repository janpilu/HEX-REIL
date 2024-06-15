import os
import argparse
import numpy as np
import random

from datetime import datetime
from modules.ppo_agent import PPOAgent
from modules.hex_env import HexEnv
from pathlib import Path
from sb3_contrib.common.wrappers import ActionMasker


def human_input_to_coordinates(human_input):
    # This function translates human terminal input into the proper array indices.
    number_translated = 27
    letter_translated = 27
    names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(human_input) > 0:
        letter = human_input[0].upper()
    if len(human_input) > 1:
        number1 = human_input[1]
    if len(human_input) > 2:
        number2 = human_input[2]
    for i in range(26):
        if names[i] == letter:
            letter_translated = i
            break
    if len(human_input) > 2:
        for i in range(10, 27):
            if number1 + number2 == "{}".format(i):
                number_translated = i - 1
    else:
        for i in range(1, 10):
            if number1 == "{}".format(i):
                number_translated = i - 1
    return (number_translated, letter_translated)


def main(args):
    # Set the board size
    board_size = args.board_size

    # Load the model
    agent = PPOAgent()
    agent.load(args.model_path)

    # Initialize the environment
    env = HexEnv(size=board_size)
    env = ActionMasker(env, lambda env: env.unwrapped.get_masked_actions())
    agent.set_env(env)

    # Decide who starts randomly
    human_player_starts = random.choice([True, False])
    human_player = 1 if human_player_starts else -1

    print(f"\nHuman is playing as {'White' if human_player == 1 else 'Black'}")
    print(f"Model is playing as {'Black' if human_player == 1 else 'White'}\n")

    current_player = 1 if human_player_starts else -1

    # Game loop
    env.reset()
    done = False
    while not done:
        env.render()
        if current_player == 1:  # Human's turn
            valid_actions = env.get_valid_actions()
            action = None
            while action not in valid_actions:
                user_input = input("Enter your move (e.g. 'A1'): ")
                coordinates = human_input_to_coordinates(
                    user_input
                )  # convert user input to coordinates
                action = env.hex.coordinate_to_scalar(
                    coordinates
                )  # convert coordinates to scalar
        else:
            # AI's turn
            board = env.hex.board
            action_set = env.get_valid_actions()
            action = agent.get_action(board, action_set)

        _, _, done, _, _ = env.step(action)
        current_player *= -1

    env.render()

    if env.hex.winner == human_player:
        print("Human wins!")
    else:
        print("Model wins!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Hex against a trained PPO Agent")

    parser.add_argument(
        "-b", "--board_size", type=int, default=5, help="Size of the board"
    )

    parser.add_argument(
        "-m", "--model_path", type=str, required=True, help="Path to the trained model"
    )

    args = parser.parse_args()

    main(args)
