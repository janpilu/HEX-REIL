# from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO as PPO
import numpy as np
import os

from fhtw_hex.hex_engine import hexPosition


class PPOAgent:

    def __init__(self):
        self.env = None
        self.model = None

    def set_env(self, env):
        self.env = env
        if self.model is not None:
            self.model.set_env(self.env)

    def set_opponent_policy(self, opponent_policy):
        self.env.set_opponent_policy(opponent_policy)

    def init_model(self):
        self.model = PPO("MlpPolicy", self.env, verbose=0)

    def train(self, steps):
        self.model.learn(total_timesteps=steps, progress_bar=True)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path)

    def evaluate(self, steps):
        obs, *vals = self.env.reset()
        for i in range(steps):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info, *vals = self.env.step(action)
            self.env.render()

    def evaluate_games(self, games, opponent_policy=None, verbose=1):
        training_opponent_policy = self.env.unwrapped.opponent_policy
        if opponent_policy is not None:
            self.env.unwrapped.opponent_policy = opponent_policy

        obs, *vals = self.env.reset()
        wins = []
        winner = 0
        board = None
        player = 0

        for i in range(games):
            done = False
            while not done:
                action, _states = self.model.predict(
                    obs, action_masks=self.env.unwrapped.get_masked_actions(obs)
                )
                obs, rewards, done, info, winner_dict = self.env.step(action)
                # print(winner_dict)
                winner = winner_dict.get("winner", 0)
                board = winner_dict.get("board", None)
                player = winner_dict.get("player", 0)

            if winner == player:
                wins.append(player)

            if verbose >= 2:
                print(f"Game {i + 1} over")

            if verbose >= 3:
                print("Winner: ", winner)
                print("Player:", player)
                self.print_board(board)

        if verbose >= 1:
            print(
                f"\nWin rate: {( len(wins) / games)*100:.2f}% - ({len(wins)}/{games})"
            )

            print(f"White wins: {wins.count(1)}")
            print(f"Black wins: {wins.count(-1)}")

        self.env.unwrapped.opponent_policy = training_opponent_policy
        return len(wins) / games

    def get_action(self, board, *args, **kwargs):
        action, _states = self.model.predict(
            board, action_masks=self.env.unwrapped.get_masked_actions(board)
        )
        return action

    def get_random_action(self, board, action_set):
        return action_set[np.random.randint(len(action_set))]

    def print_board(self, board):
        engine = hexPosition(size=len(board))
        engine.board = board
        engine.print()
        print("\n")
