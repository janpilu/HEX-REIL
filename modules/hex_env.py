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
        self.current_player = 1

    def set_opponent_policy(self, opponent_policy):
        self.opponent_policy = opponent_policy

    def reset(self, **kwargs):
        # print("Resetting environment")
        self.hex.reset()
        return self.get_corrected_board(), {}

    def set_player(self, player):
        self.current_player = player

    def step(self, action):

        coordinates = (
            self.hex.scalar_to_coordinates(action)
            if self.current_player == 1
            else self.hex.recode_coordinates(self.hex.scalar_to_coordinates(action))
        )

        self.hex.move(coordinates)

        done = self.hex.winner != 0

        if not done and self.opponent_policy is not None:
            self.opponent_action()
            done = self.hex.winner != 0

        reward = 0

        winner = 0
        winning_board = None

        if done:
            reward = 1 if self.hex.winner == self.current_player else -1
            winner = self.hex.winner
            winning_board = self.hex.board
            self.hex.reset()
            self.current_game += 1
            self.current_player *= -1
            # print(f"Game {self.current_game} over")
            # print("Winner: ", winner)
            # print("Player:", self.current_player)
            if self.current_player == -1:
                self.opponent_action()

        return (
            self.get_corrected_board(),
            reward,
            done,
            done,
            {"winner": winner, "board": winning_board, "player": self.current_player},
        )

    def get_corrected_board(self):
        return (
            self.hex.board
            if self.current_player == 1
            else self.hex.recode_black_as_white()
        )

    def opponent_action(self):
        board = (
            self.hex.board
            if self.current_player == -1
            else self.hex.recode_black_as_white()
        )

        opponent_action = self.opponent_policy(
            board,
            self.get_valid_actions(board=board),
            self.current_game,  # <-- Ensure third argument is passed
        )
        # while opponent_action not in self.get_valid_actions(board=recoded_board):
        #     opponent_action = self.opponent_policy(
        #         recoded_board,
        #         self.get_valid_actions(board=recoded_board),
        #         self.current_game,  # <-- Ensure third argument is passed
        #     )

        opponent_action = (
            self.hex.scalar_to_coordinates(opponent_action)
            if self.current_player == -1
            else self.hex.recode_coordinates(
                self.hex.scalar_to_coordinates(opponent_action)
            )
        )

        self.hex.move(opponent_action)

    def render(self, mode="human", close=False):
        if mode == "human":
            self.hex.print()

    def close(self):
        pass

    def get_valid_actions(self, board=None):
        if board is None:
            board = (
                self.hex.board
                if self.current_player == 1
                else self.hex.recode_black_as_white()
            )
        valid_actions = []
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    valid_actions.append(self.hex.coordinate_to_scalar((i, j)))
        return valid_actions

    def get_masked_actions(self, board=None):
        if board is None:
            board = (
                self.hex.board
                if self.current_player == 1
                else self.hex.recode_black_as_white()
            )

        action_masks = np.ones(self.hex.size**2)
        for i in range(self.hex.size):
            for j in range(self.hex.size):
                if board[i][j] != 0:
                    action_masks[self.hex.coordinate_to_scalar((i, j))] = 0
        return action_masks
