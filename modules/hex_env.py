import gymnasium as gym
from gymnasium import spaces
import numpy as np
from fhtw_hex.hex_engine import hexPosition


class HexEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, opponent_policy=None, size=5):
        super(HexEnv, self).__init__()
        self.hex = hexPosition(size=size)
        self.size = size
        self.action_space = spaces.Discrete(size * size)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(size, size), dtype=np.int8
        )
        self.opponent_policy = opponent_policy
        self.current_game = 0

    def set_opponent_policy(self, opponent_policy):
        self.opponent_policy = opponent_policy

    def reset(self, **kwargs):
        self.hex.reset()
        return np.array(self.hex.board, dtype=np.int8), {}

    def step(self, action):
        if self.hex.winner != 0:
            self.hex.reset()
            self.current_game += 1

        if action not in self.get_valid_actions():
            return (
                np.array(self.hex.board, dtype=np.int8),
                -5,
                False,
                False,
                {"invalid_action": True},
            )

        coordinates = self.hex.scalar_to_coordinates(action)
        self.hex.move(coordinates)

        done = self.hex.winner != 0

        if not done and self.opponent_policy is not None:
            recoded_board = self.hex.recode_black_as_white()
            opponent_action = self.opponent_policy(
                recoded_board,
                self.get_valid_actions(board=recoded_board),
                self.current_game,  # <-- Ensure third argument is passed
            )
            while opponent_action not in self.get_valid_actions(board=recoded_board):
                opponent_action = self.opponent_policy(
                    recoded_board,
                    self.get_valid_actions(board=recoded_board),
                    self.current_game,  # <-- Ensure third argument is passed
                )

            opponent_action = self.hex.recode_coordinates(
                self.hex.scalar_to_coordinates(opponent_action)
            )

            self.hex.move(opponent_action)
            self.hex.recode_black_as_white()
            done = self.hex.winner != 0

        reward = 0
        if done:
            reward = 1 if self.hex.winner == 1 else -1
        return np.array(self.hex.board, dtype=np.int8), reward, done, done, {}

    def render(self, mode="human", close=False):
        if mode == "human":
            self.hex.print()

    def close(self):
        pass

    def get_valid_actions(self, board=None):
        if board is None:
            board = self.hex.board
        valid_actions = []
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    valid_actions.append(self.hex.coordinate_to_scalar((i, j)))
        return valid_actions

    def sample_action(self):
        return np.random.choice(self.valid_actions)
