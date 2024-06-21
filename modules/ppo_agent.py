import numpy as np
import torch
from modules.ppo import PPO  # Import your custom PPO implementation
from tqdm import tqdm

from fhtw_hex.hex_engine import hexPosition


class PPOAgent:

    def __init__(self, logger=None):
        self.env = None
        self.model = None
        self.logger = logger
        if self.logger is not None:
            logger.set_lr(0.0003)

    def set_env(self, env):
        self.env = env
        if self.model is not None:
            self.model.env = (
                self.env
            )  # Ensure the environment is set in your custom PPO

    def set_opponent_policy(self, opponent_policy):
        self.env.set_opponent_policy(opponent_policy)

    def init_model(self):
        input_dims = self.env.observation_space.shape
        n_actions = self.env.action_space.n
        self.model = PPO(
            n_actions,
            input_dims,
            alpha=0.0003,
            batch_size=64,
            ent_coef=0.01,
            use_conv=True,
        )

    def train(self, steps):
        obs = self.env.reset()[0]
        game = 0
        done = False
        for _ in tqdm(range(steps), desc="Collecting Game Data"):
            action, log_probs, value = self.model.choose_action(
                obs, action_mask=self.get_masked_actions(obs)
            )
            new_obs, reward, done, _, info = self.env.step(action)
            self.model.remember(obs, action, log_probs, value, reward, done)
            obs = new_obs

            if done:
                obs = self.env.reset()[0]
                game += 1
                # self.model.learn()
                # self.model.memory.clear_memory()
                if game % 500 == 0:
                    self.model.learn()
        while not done:
            action, log_probs, value = self.model.choose_action(
                obs, action_mask=self.get_masked_actions(obs)
            )
            new_obs, reward, done, _, info = self.env.step(action)
            self.model.remember(obs, action, log_probs, value, reward, done)
            obs = new_obs

        self.env.reset()
        self.model.learn()

    def save(self, path):
        self.model.save_models()

    def load(self, path):
        self.model.load_models()

    def evaluate(self, steps):
        obs, *vals = self.env.reset()
        for i in range(steps):
            action, _, _ = self.model.choose_action(
                obs, action_mask=self.get_masked_actions(obs)
            )
            obs, rewards, dones, info, *vals = self.env.step(action)
            self.env.render()

    def update_lr(self):
        # learning_rates = [0.00005, 0.0001, 0.0003]
        # current_lr = self.model.actor.optimizer.param_groups[0]["lr"]
        # new_lr_index = (learning_rates.index(current_lr) + 1) % len(learning_rates)
        # new_lr = learning_rates[new_lr_index]
        # self.model.actor.optimizer.param_groups[0]["lr"] = new_lr
        # self.model.critic.optimizer.param_groups[0]["lr"] = new_lr
        # self.logger.set_lr(new_lr)
        pass

    def evaluate_games(self, games, opponent_policy=None, verbose=1):
        training_opponent_policy = self.env.unwrapped.opponent_policy
        if opponent_policy is not None:
            self.env.unwrapped.opponent_policy = opponent_policy

        obs, *vals = self.env.reset()
        wins = []
        winner = 0
        board = None
        player = 0

        for i in tqdm(range(games), desc="Evaluation Progress"):
            done = False
            while not done:
                action, _, _ = self.model.choose_action(
                    obs, action_mask=self.get_masked_actions(obs)
                )
                obs, rewards, done, info, winner_dict = self.env.step(action)
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
                f"\nWin rate: {(len(wins) / games) * 100:.2f}% - ({len(wins)}/{games})"
            )
            print(f"White wins: {wins.count(1)}")
            print(f"Black wins: {wins.count(-1)}")

        self.env.unwrapped.opponent_policy = training_opponent_policy
        return {
            "win_rate": len(wins) / games,
            "white_wins": wins.count(1),
            "black_wins": wins.count(-1),
        }

    def get_action(self, board, *args, **kwargs):
        action, _, _ = self.model.choose_action(
            board, action_mask=self.get_masked_actions(board)
        )
        return action

    def get_random_action(self, board, action_set):
        return action_set[np.random.randint(len(action_set))]

    def print_board(self, board):
        engine = hexPosition(size=len(board))
        engine.board = board
        engine.print()
        print("\n")

    def focus_on_player(self, players):
        self.env.unwrapped.focus_on_player(players)

    def get_masked_actions(self, board):
        size = len(board)
        hex = hexPosition(size=size)

        action_masks = np.ones(size**2)
        for i in range(size):
            for j in range(size):
                if board[i][j] != 0:
                    action_masks[hex.coordinate_to_scalar((i, j))] = 0
        return action_masks
