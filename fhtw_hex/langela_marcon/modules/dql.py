import os
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

        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            # Calculate the size of the flattened feature map after convolutions
            conv_out_size = input_dims[0] * input_dims[1] * 64
        else:
            conv_out_size = input_dims[0] * input_dims[1]

        self.fc = nn.Sequential(
            *itertools.chain.from_iterable(
                [
                    [
                        (
                            nn.Linear(conv_out_size, hidden_size)
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
        if self.use_conv:
            state = state.unsqueeze(1)  # Add channel dimension
            features = self.conv(state)
            features = features.view(features.size(0), -1)  # Flatten the features
        else:
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
        tau=0.01,
    ):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.tau = tau

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

        self.update_target_network(hard_update=True)

    def update_target_network(self, hard_update=False):
        if hard_update:
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            for target_param, param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

    def remember(self, episode):
        self.memory.append(episode)

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
        for episode in batch:
            states, actions, rewards, next_states, dones = zip(*episode)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
            rewards = torch.FloatTensor(np.array(rewards))
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(np.array(dones))

            q_values = self.q_network(states).gather(1, actions).squeeze(1)
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_target_network()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))
