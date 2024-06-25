import numpy as np
from tqdm import tqdm

from .agent import Agent
from ..ppo import PPO
from ...fhtw_hex.hex_engine import hexPosition


class PPOAgent(Agent):
    def __init__(self, logger=None, hidden_layers=2, hidden_size=256, use_conv=False):
        self.env = None
        self.model = None
        self.logger = logger
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.use_conv = use_conv

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
            use_conv=self.use_conv,
            hidden_layers=self.hidden_layers,
            hidden_size=self.hidden_size,
        )

    def train(self, episodes=1000):
        obs = self.env.reset()[0]

        for episode in tqdm(range(episodes), desc="Collecting Game Data"):
            done = False
            while not done:
                action, log_probs, value = self.model.choose_action(
                    obs, action_mask=self.get_masked_actions(obs)
                )
                new_obs, reward, done, _, info = self.env.step(action)
                self.model.remember(obs, action, log_probs, value, reward, done)
                obs = new_obs

            obs = self.env.reset()[0]

            if episode % 10 == 0:
                self.model.learn()

    def get_action(self, board):
        action, _, _ = self.model.choose_action(
            board, action_mask=self.get_masked_actions(board)
        )
        return action

    def get_random_action(self, board, action_set):
        return action_set[np.random.randint(len(action_set))]

    def policy(self, board, action_set):
        engine = hexPosition(len(board))
        engine.board = board
        current_player = self.check_current_player(board)
        action = None
        coordinates = []
        if current_player == 1:
            action = self.get_action(board)
            coordinates = engine.scalar_to_coordinates(action)
        else:
            action = self.get_action(engine.recode_black_as_white())
            coordinates = engine.recode_coordinates(
                engine.scalar_to_coordinates(action)
            )
        return coordinates
