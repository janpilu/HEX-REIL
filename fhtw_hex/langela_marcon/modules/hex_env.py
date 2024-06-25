import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ..fhtw_hex.hex_engine import hexPosition
import copy


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
        self.current_player_index = 0
        self.players = [1, -1]

    def set_opponent_policy(self, opponent_policy):
        self.opponent_policy = opponent_policy

    def reset(self, **kwargs):
        self.hex.reset()
        opponent_state, opponent_action, opponent_new_state = (
            None,
            None,
            None,
        )
        if self.current_player == -1:
            opponent_state, opponent_action = self.opponent_action()
        return self.get_corrected_board(), {
            "opponent_state": opponent_state,
            "opponent_action": opponent_action,
        }

    def set_player(self, player):
        self.current_player = player

    def is_board_empty(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.hex.board[i][j] != 0:
                    return False
        return True

    def step(self, action):
        # Convert action to board coordinates and move
        coordinates = self.convert_action_to_coordinates(action)
        self.hex.move(coordinates)

        # Check if the game is done
        done = self.hex.winner != 0

        opponent_state, opponen_action = (
            None,
            None,
        )

        # If the game is not done, let the opponent make a move
        if not done:
            opponent_state, opponen_action = self.opponent_action()
            done = self.hex.winner != 0
        else:
            opponent_state = self.get_corrected_board(opponent=True)

        reward = 0
        winner = 0
        player = 0
        winning_board = None
        opponent_winning_board = None

        # If the game is done, calculate the reward and reset the board
        if done:
            reward = 1 if self.hex.winner == self.current_player else -1
            winner = self.hex.winner
            player = self.current_player
            winning_board = copy.deepcopy(self.hex.board)
            opponent_winning_board = self.get_corrected_board(opponent=True)
            self.current_game += 1
            self.current_player_index = self.current_player_index + 1
            if self.current_player_index >= len(self.players):
                self.current_player_index = 0
            self.current_player = self.players[self.current_player_index]
            self.reset()

        return (
            self.get_corrected_board(),  # Return the current board state without correction
            reward,
            done,
            done,
            {
                "winner": winner,
                "board": winning_board,
                "player": player,
                "opponent_state": opponent_state,
                "opponent_action": opponen_action,
                "opponent_winning_board": opponent_winning_board,
            },
        )

    def opponent_action(self):
        board = self.get_corrected_board(opponent=True)

        action = self.opponent_policy(
            board,
            self.get_valid_actions(opponent=True),
            self.current_game,
        )

        coordinates = self.convert_action_to_coordinates(action, opponent=True)

        self.hex.move(coordinates)

        return board, action

    def render(self, mode="human", close=False):
        if mode == "human":
            self.hex.print()

    def get_valid_actions(self, opponent=False, board=None):
        if board is None:
            board = self.get_corrected_board(opponent=opponent)

        valid_actions = []
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    valid_actions.append(self.hex.coordinate_to_scalar((i, j)))
        return valid_actions

    def get_masked_actions(self, board=None):
        if board is None:
            board = self.get_corrected_board()

        action_masks = np.ones(self.hex.size**2)
        for i in range(self.hex.size):
            for j in range(self.hex.size):
                if board[i][j] != 0:
                    action_masks[self.hex.coordinate_to_scalar((i, j))] = 0
        return action_masks

    def get_corrected_board(self, opponent=False):
        if opponent:
            return (
                copy.deepcopy(self.hex.board)
                if self.current_player == -1
                else copy.deepcopy(self.hex.recode_black_as_white())
            )

        return (
            copy.deepcopy(self.hex.board)
            if self.current_player == 1
            else copy.deepcopy(self.hex.recode_black_as_white())
        )

    def convert_action_to_coordinates(self, action, opponent=False):
        if opponent:
            return (
                self.hex.scalar_to_coordinates(action)
                if self.current_player == -1
                else self.hex.recode_coordinates(self.hex.scalar_to_coordinates(action))
            )

        return (
            self.hex.scalar_to_coordinates(action)
            if self.current_player == 1
            else self.hex.recode_coordinates(self.hex.scalar_to_coordinates(action))
        )

    def focus_on_player(self, players):
        self.players = players

    def close(self):
        pass
