import math
from tqdm import tqdm
from fhtw_hex.hex_engine import hexPosition
from modules.agents.agent import Agent
from modules.dql import DQL


class DQLAgent(Agent):
    def __init__(self, logger=None, hidden_layers=2, hidden_size=256, use_conv=False):
        super().__init__(logger)
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.use_conv = use_conv

    def init_model(self):
        input_dims = self.env.observation_space.shape
        n_actions = self.env.action_space.n
        self.model = DQL(
            input_dims=input_dims,
            n_actions=n_actions,
            lr=0.0003,
            batch_size=64,
            use_conv=self.use_conv,
            hidden_layers=self.hidden_layers,
            hidden_size=self.hidden_size,
        )

    def train(self, num_episodes=1000):
        learn_every = 4
        episode_counter = 0
        epsilon = 1
        epsilon_end = 0.01
        epsilon_decay = 0.995

        for episode in tqdm(range(num_episodes), desc="Training Progress"):
            state, info = self.env.reset()
            done = False
            episode_memory = []
            opponent_episode_memory = []
            opponent_state, opponent_action, opponent_next_state = (
                None,
                None,
                None,
            )

            if info["opponent_state"] is not None:
                opponent_state = info["opponent_state"]
                opponent_action = info["opponent_action"]

            while not done:
                action = self.model.choose_action(
                    state, action_mask=self.get_masked_actions(state), epsilon=epsilon
                )

                next_state, reward, done, _, info = self.env.step(action)

                opponent_next_state = info["opponent_state"]

                opponent_reward = -1 if reward == 1 else 0

                episode_memory.append((state, action, reward, next_state, done))
                if opponent_state is not None and opponent_action is not None:
                    opponent_episode_memory.append(
                        (
                            opponent_state,
                            opponent_action,
                            opponent_reward,
                            opponent_next_state,
                            False,
                        )
                    )

                opponent_action = info["opponent_action"]

                state = next_state
                opponent_state = opponent_next_state
                opponent_next_state = info["opponent_winning_board"]

            if reward == -1:
                opponent_episode_memory.append(
                    (
                        opponent_state,
                        opponent_action,
                        -reward,
                        opponent_next_state,
                        done,
                    )
                )

            self.model.remember(episode_memory)
            self.model.remember(opponent_episode_memory)
            episode_counter += 1
            if episode_counter % learn_every == 0:
                self.model.learn()
                if epsilon > epsilon_end:
                    epsilon *= epsilon_decay

    # def train(self, episodes=1000):
    #     obs = self.env.reset()[0]
    #     epsilon = 1

    #     for episode in tqdm(range(episodes), desc="Collecting Game Data"):
    #         done = False
    #         while not done:
    #             action = self.model.choose_action(
    #                 obs, action_mask=self.get_masked_actions(obs), epsilon=epsilon
    #             )
    #             new_obs, reward, done, _, _ = self.env.step(action)
    #             self.model.remember(obs, action, reward, new_obs, done)
    #             obs = new_obs

    #         epsilon = max(0.1, epsilon - 1 / episodes * 0.9)
    #         obs = self.env.reset()[0]

    #         if episode % 50 == 0:
    #             self.model.learn()

    #         if episode % self.model.target_update == 0:
    #             self.model.update_target_network()
    #     self.model.learn()
    #     self.model.update_target_network()
    #     self.env.reset()

    def get_action(self, board, epsilon=0.00):
        action = self.model.choose_action(
            board, action_mask=self.get_masked_actions(board), epsilon=epsilon
        )
        return action

    def policy(self, board, action_set):
        engine = hexPosition(len(board))
        engine.board = board
        current_player = self.check_current_player(board)
        action = None
        coordinates = []
        if current_player == 1:
            action = self.get_action(board, epsilon=0)
            coordinates = engine.scalar_to_coordinates(action)
        else:
            action = self.get_action(engine.recode_black_as_white(), epsilon=0)
            coordinates = engine.recode_coordinates(
                engine.scalar_to_coordinates(action)
            )
        return coordinates
