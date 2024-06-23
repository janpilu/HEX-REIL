import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import itertools


class QNetwork(nn.Module):
    def __init__(
        self, input_dims, n_actions, hidden_layers=2, hidden_size=256, use_conv=False
    ):
        super(QNetwork, self).__init__()

        self.fc = nn.Sequential(
            *itertools.chain.from_iterable(
                [
                    [
                        (
                            nn.Linear(input_dims[0] * input_dims[1], hidden_size)
                            if i == 0
                            else nn.Linear(hidden_size, hidden_size)
                        ),
                        nn.ReLU(),
                    ]
                    for i in range(hidden_layers)
                ]
            ),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, state):
        features = state.view(state.size(0), -1)
        return self.fc(features)


class DQL:
    def __init__(
        self,
        input_dims,
        n_actions,
        hidden_size=256,
        hidden_layers=2,
        use_conv=False,
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        buffer_size=15000,
        target_update=10,
    ):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory = deque(maxlen=buffer_size)

        self.q_network = QNetwork(
            input_dims,
            n_actions,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            use_conv=use_conv,
        )
        self.target_network = QNetwork(
            input_dims,
            n_actions,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            use_conv=use_conv,
        )

        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr)

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, action_mask, epsilon=0.1):
        if random.random() < epsilon:
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            return random.choice(valid_actions)

        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state)

        valid_actions = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)
        masked_q_values = q_values.masked_fill(~valid_actions, -float("inf"))

        return torch.argmax(masked_q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))


# # Example usage
# env = gym.make("CartPole-v1")
# input_dims = env.observation_space.shape[0]
# n_actions = env.action_space.n
# agent = DQL(input_dims=input_dims, n_actions=n_actions)

# num_episodes = 1000
# for episode in range(num_episodes):
#     state = env.reset()[0]
#     done = False
#     total_reward = 0

#     while not done:
#         action = agent.act(state)
#         next_state, reward, done, *_ = env.step(action)
#         agent.remember(state, action, reward, next_state, done)
#         agent.learn()
#         state = next_state
#         total_reward += reward

#     if episode % agent.target_update == 0:
#         agent.update_target_network()

#     print(f"Episode {episode}, Total Reward: {total_reward}")
