from stable_baselines3 import PPO
import numpy as np
import os


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
        training_opponent_policy = self.env.opponent_policy
        if opponent_policy is not None:
            self.env.opponent_policy = opponent_policy
        obs, *vals = self.env.reset()
        winners = []
        for i in range(games):
            done = False
            while not done:
                action, _states = self.model.predict(obs)
                obs, rewards, done, info, *vals = self.env.step(action)
            if verbose >= 2:
                print(f"Game {i + 1} over")
                print("Winner: ", self.env.hex.winner)
            winners.append(self.env.hex.winner)
            if verbose >= 3:
                self.env.render()
        if verbose >= 1:
            print(f"Win rate: {(winners.count(1) / games):.2f}% - ({winners.count(1)}/{games})")
        
        self.env.opponent_policy = training_opponent_policy
        return winners.count(1) / games

    def get_action(self, board, action_set, *args, **kwargs):
        valid_action = False
        while not valid_action:
            action, _states = self.model.predict(board)
            if action in action_set:
                return action

    def get_random_action(self, board, action_set):
        return action_set[np.random.randint(len(action_set))]
