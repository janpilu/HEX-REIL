from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np

from ...fhtw_hex.hex_engine import hexPosition


class Agent(ABC):
    def __init__(self, logger=None):
        self.env = None
        self.model = None
        self.logger = logger

    def set_env(self, env):
        self.env = env

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        if self.model is None:
            self.init_model()

        self.model.load(path)

    def evaluate_games(self, games, opponent_policy=None, verbose=1):
        training_opponent_policy = self.env.opponent_policy
        if opponent_policy is not None:
            self.env.opponent_policy = opponent_policy

        obs = self.env.reset()[0]
        wins = []
        winner = 0
        board = None
        player = 0

        print(f"Evaluating {games} games")
        for i in tqdm(range(games), desc="Evaluation Progress"):
            done = False
            while not done:
                action = self.get_action(obs)
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

        self.env.opponent_policy = training_opponent_policy
        return {
            "win_rate": len(wins) / games,
            "white_wins": wins.count(1),
            "black_wins": wins.count(-1),
        }

    def focus_on_player(self, players):
        self.env.focus_on_player(players)

    def get_masked_actions(self, board):
        size = len(board)
        hex = hexPosition(size=size)

        action_masks = np.ones(size**2)
        for i in range(size):
            for j in range(size):
                if board[i][j] != 0:
                    action_masks[hex.coordinate_to_scalar((i, j))] = 0
        return action_masks

    def print_board(self, board):
        engine = hexPosition(size=len(board))
        engine.board = board
        engine.print()
        print("\n")

    def set_opponent_policy(self, opponent_policy):
        self.env.set_opponent_policy(opponent_policy)

    def check_current_player(self, board):
        black = 0
        white = 0
        for row in board:
            black += row.count(-1)
            white += row.count(1)

        return 1 if white == black else -1

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def train(self, steps):
        pass

    @abstractmethod
    def get_action(
        self,
        board,
        **kwargs,
    ):
        pass

    @abstractmethod
    def policy(self, board, action_set):
        pass
