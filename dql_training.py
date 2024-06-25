import os
import sys
import argparse
import numpy as np


from pathlib import Path
from modules.agents.ppo_agent import PPOAgent
from modules.logger import EvaluationLogger
from modules.agents.dql_agent import DQLAgent
from modules.hex_env import HexEnv
from datetime import datetime

from sb3_contrib.common.wrappers import ActionMasker

# load train module
import modules.train_modules as train_modules

import config

agent_class = DQLAgent if config.model == "dql" else PPOAgent


def main(args):

    number_of_runs = args.run_numbers

    for run_number in range(number_of_runs):

        # create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        print(f"Start training at {timestamp}\n")

        # assign the arguments to the variables
        board_size = args.board_size
        model = config.model

        # main folder for models
        os.makedirs("./models", exist_ok=True)

        conv_flag = "conv" if config.use_conv else "fc"

        model_folder = f"./models/{board_size}x{board_size}/{config.model}/{config.hidden_layers}_{config.hidden_size}_{conv_flag}"
        os.makedirs(model_folder, exist_ok=True)

        model_path = f"{model_folder}/{timestamp}_{model}"

        models = train_modules.get_sorted_models(model_folder)
        most_recent_model = models[0] if len(models) > 0 else None

        policies = []

        env = HexEnv(size=board_size)

        include_most_recent = run_number % 2 == 0

        if len(models) > 0 and not args.against_random:
            print("Loading models")
            policies, model_files = train_modules.get_policies(
                model_folder, agent_class, env, include_most_recent=include_most_recent
            )
            opponent_policy = train_modules.get_opponent_policy(
                policies, model_files, args.number_of_policies
            )

        else:
            if not args.against_random:
                print("No models found, playing against random")
            else:
                print("Playing against random")
            opponent_policy = lambda board, action_set, current_game: np.random.choice(
                action_set
            )  # <-- Modified to accept three arguments

        env.set_opponent_policy(opponent_policy)
        logger = EvaluationLogger(f"training-{timestamp}.csv")
        agent = agent_class(
            logger=logger,
            hidden_layers=config.hidden_layers,
            hidden_size=config.hidden_size,
            use_conv=config.use_conv,
        )

        # Load if you have a trained model
        # env = ActionMasker(env, lambda env: env.unwrapped.get_masked_actions())

        if most_recent_model is not None:
            print("\nLoading most recent model")
            agent.set_env(env)
            agent.load(f"{model_folder}/{most_recent_model}")
            print(f"Model loaded, continuing training from {most_recent_model}")

            evaluation_agent = agent_class(
                hidden_layers=config.hidden_layers,
                hidden_size=config.hidden_size,
                use_conv=config.use_conv,
            )

            evaluation_agent.set_env(env)
            if args.against_random:
                print("Playing against random")
                evaluation_agent.init_model()
            else:
                print("Loading most recent model for evaluation")
                print(f"Loading {model_folder}/{most_recent_model}")
                evaluation_agent.load(f"{model_folder}/{most_recent_model}")

        else:
            print("Creating new model")
            agent.set_env(env)
            agent.init_model()

            # define random agent for evaluation
            evaluation_agent = agent_class()
            evaluation_agent.set_env(env)
            evaluation_agent.init_model()  # Ensure evaluation model is initialized

        agent_checkpoint = agent

        score = -1
        training_round = 0
        scores = []

        while score < args.evaluation_threshold / 100:

            if training_round >= 10:
                mean_score = np.mean(scores[-10:])
                if mean_score < 0.35:
                    print("Score is less than 35%, stopping training")
                    agent = agent_checkpoint

            if training_round != 0:

                if (
                    len(policies) > args.number_of_policies
                    and training_round % (args.lr_update_threshold * 3) == 0
                ):
                    print("Resampling opponent policy")
                    opponent_policy = train_modules.get_opponent_policy(
                        policies, model_files, args.number_of_policies
                    )
                    agent.set_opponent_policy(opponent_policy)

            training_round += 1

            if not args.skip_training:

                if score > -1:
                    print(
                        f"Score: {score*100:.2f}%, did not reach threshold of {args.evaluation_threshold}%\n"
                    )

                print("\nTraining model")
                agent.train(args.training_steps)

            print(f"Evaluating model against most recent model ({most_recent_model})")
            agent.focus_on_player([1, -1])
            results = agent.evaluate_games(
                args.number_of_policies * 20,
                # args.evaluation_steps,
                # lambda board, _, __: evaluation_agent.get_action(board),
                verbose=3,
            )
            results_self = None
            if config.model == "ppo":
                results_self = agent.evaluate_games(
                    args.evaluation_steps,
                    lambda board, _, __: evaluation_agent.get_action(board),
                    verbose=3,
                )
            else:
                results_self = agent.evaluate_games(
                    2,
                    lambda board, _, __: evaluation_agent.get_action(board),
                    verbose=3,
                )

            n_eval_games = args.number_of_policies * 20 + 2
            if config.model == "ppo":
                n_eval_games += args.evaluation_steps - 2

            results_total = {
                "win_rate": (results["win_rate"] + results_self["win_rate"]) / 2,
                "white_wins": results["white_wins"] + results_self["white_wins"],
                "black_wins": results["black_wins"] + results_self["black_wins"],
            }

            results = results_total

            score = results["win_rate"]
            scores.append(score)

            if (
                results["black_wins"]
                < n_eval_games * score * args.focus_threshold / 100
            ):
                print(
                    f"Black wins less than {args.focus_threshold}% focus on black player"
                )
                agent.focus_on_player([-1])
            elif (
                results["white_wins"]
                < n_eval_games * score * args.focus_threshold / 100
            ):
                print(
                    f"White wins less than {args.focus_threshold}% focus on white player"
                )
                agent.focus_on_player([1])
            else:
                print(f"White and black wins greater than {args.focus_threshold}%")
                agent.focus_on_player([1, -1])

            if args.skip_training:
                break

        if args.skip_training:
            break
        print("\nSaving model!\n")
        agent.save(model_path)
        print(f"Run {run_number + 1} out of {number_of_runs} completed\n\n\n")
    logger.close()


## TODOS:
# Timestamp,
# evaluate models after training,
# only better models are saved,
# sample action selection from previous models,
# extrend step function to allow for multiple games to be played


if __name__ == "__main__":

    # define the parser
    parser = argparse.ArgumentParser(description="Training PPO Agent")

    # Add argument for number of runs
    parser.add_argument(
        "-r", "--run_numbers", type=int, default=20, help="Number of runs"
    )

    # Add argument for board size
    parser.add_argument(
        "-b", "--board_size", type=int, default=7, help="Size of the board"
    )

    # Add argument for model
    parser.add_argument("-m", "--model", type=str, default="ppo", help="Model to use")

    # Add argument for training steps
    parser.add_argument(
        "-ts",
        "--training_steps",
        type=int,
        default=1000,
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
        default=200,
        help="Number of evaluation steps",
    )

    # Add argument for evaluation steps
    parser.add_argument(
        "-et",
        "--evaluation_threshold",
        type=int,
        default=65,
        help="Percent to reach before stopping training",
    )

    # Add argument for resampling threshold
    parser.add_argument(
        "-rt",
        "--lr_update_threshold",
        type=int,
        default=4,
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

    parser.add_argument(
        "-ft",
        "--focus_threshold",
        type=int,
        default=30,
        help="Threshold for focusing on player",
    )

    # parse the arguments
    args = parser.parse_args()

    # call the main function
    main(args)
