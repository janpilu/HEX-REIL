import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim import lr_scheduler
import math
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def init_weights(module, gain=1.0):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(
            0, n_states, self.batch_size
        )  # E.g.: [0, 32, 64, 96, ...]
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [
            indices[i : i + self.batch_size] for i in batch_start
        ]  # E.g.: [[0, 1, 2, ..., 31], [32, 33, 34, ..., 63], ...]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.logprobs),
            np.array(self.values),
            np.array(self.rewards),
            np.array(self.is_terminals),
            batches,
        )

    def store_memory(self, state, action, probs, values, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.is_terminals.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []


class CustomFeatureExtractor(nn.Module):
    def __init__(self, input_dims, conv_channels=[16, 32, 64], fc_dims=256):
        super(CustomFeatureExtractor, self).__init__()

        # Dynamically create convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_dims[0]

        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            self.conv_layers.append(nn.ReLU())
            in_channels = out_channels

        conv_out_size = conv_channels[-1] * input_dims[1] * input_dims[2]

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, fc_dims),
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten the tensor correctly
        x = x.view(batch_size, -1)  # Flatten the tensor

        x = self.fc_layers(x)
        return x


class PPOActor(nn.Module):
    def __init__(
        self,
        n_actions,
        alpha,
        features_extractor=None,
        input_dims=1,
        fc1_dims=256,
        fc2_dims=256,
        checkpoint_dir="tmp/ppo",
    ):
        super(PPOActor, self).__init__()

        self.features_extractor = features_extractor
        self.checkpoint_file = os.path.join(checkpoint_dir, "actor_ppo")

        self.fc2 = nn.Sequential(
            nn.Linear(input_dims[0] * input_dims[1], fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1),
        )

        if features_extractor is not None:
            self.fc2 = nn.Sequential(
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1),
            )

        self.optimizer = optim.AdamW(self.parameters(), lr=alpha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.apply(init_weights)

    def forward(self, state):
        features = state
        if self.features_extractor is not None:
            state = state.unsqueeze(1)
            features = self.features_extractor(state)
        else:
            features = state.view(state.size(0), -1)
        dist = self.fc2(features)
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class PPOCritic(nn.Module):
    def __init__(
        self,
        alpha,
        features_extractor=None,
        input_dims=1,
        fc1_dims=256,
        fc2_dims=256,
        checkpoint_dir="tmp/ppo",
    ):
        super(PPOCritic, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, "critic_ppo")

        self.features_extractor = features_extractor
        self.fc2 = nn.Sequential(
            nn.Linear(input_dims[0] * input_dims[1], fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )

        if features_extractor is not None:
            self.fc2 = nn.Sequential(
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1),
            )

        self.apply(init_weights)
        self.optimizer = optim.AdamW(self.parameters(), lr=alpha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        features = state
        if self.features_extractor is not None:
            state = state.unsqueeze(1)
            features = self.features_extractor(state)
        else:
            features = state.view(state.size(0), -1)

        value = self.fc2(features)
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class PPO:
    def __init__(
        self,
        n_actions,
        input_dims,
        use_conv=True,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        gamma=0.99,
        alpha=0.0003,
        peak_alpha=1e-3,  # Peak learning rate
        min_alpha=1e-5,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        fc1_dims=256,
        fc2_dims=256,
        linear_increase_steps=4000,  # Number of steps for linear increase
        total_timesteps=20000,  # Total number of timesteps
        log_dir="logs",  # Directory to save logs
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.memory = PPOMemory(batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a shared feature extractor
        if use_conv:
            input_dims = (1, *input_dims)  # Add channel dimension if using convolution
            self.features_extractor = CustomFeatureExtractor(
                input_dims, conv_channels=[16, 32, 64], fc_dims=fc1_dims
            ).to(self.device)
        else:
            self.features_extractor = None

        self.actor = PPOActor(
            n_actions,
            input_dims=input_dims,
            alpha=alpha,
            features_extractor=self.features_extractor,
            fc1_dims=fc1_dims,
            fc2_dims=fc2_dims,
        )
        self.critic = PPOCritic(
            alpha=alpha,
            input_dims=input_dims,
            features_extractor=self.features_extractor,
            fc1_dims=fc1_dims,
            fc2_dims=fc2_dims,
        )

        # Define custom schedulers
        self.linear_increase_steps = linear_increase_steps
        self.total_timesteps = total_timesteps
        self.peak_alpha = peak_alpha
        self.min_alpha = min_alpha
        self.actor_scheduler = lr_scheduler.LambdaLR(
            self.actor.optimizer, lr_lambda=self.lr_schedule
        )
        self.critic_scheduler = lr_scheduler.LambdaLR(
            self.critic.optimizer, lr_lambda=self.lr_schedule
        )

        # Tensorboard logging
        self.writer = SummaryWriter(log_dir)

    def log_metrics(self, loss, actor_loss, critic_loss, entropy_loss, step):
        self.writer.add_scalar("loss/total_loss", loss.item(), step)
        self.writer.add_scalar("loss/actor_loss", actor_loss.item(), step)
        self.writer.add_scalar("loss/critic_loss", critic_loss.item(), step)
        self.writer.add_scalar("loss/entropy_loss", entropy_loss.item(), step)
        self.writer.add_scalar(
            "learning_rate", self.actor_scheduler.get_last_lr()[0], step
        )

    def log_gradients(self, model, step):
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(
                    f"gradients/{name}", param.grad.cpu().data.numpy(), step
                )

    def log_parameters(self, model, step):
        for name, param in model.named_parameters():
            self.writer.add_histogram(
                f"parameters/{name}", param.cpu().data.numpy(), step
            )

    def lr_schedule(self, step):
        if step < self.linear_increase_steps:
            return (
                step
                / self.linear_increase_steps
                * (self.peak_alpha / self.actor.optimizer.defaults["lr"])
            )
        else:
            lr = self.peak_alpha * math.exp(
                -0.1 * (step - self.linear_increase_steps) / self.total_timesteps
            )
        return max(lr, self.min_alpha)

    def remember(self, state, action, probs, values, reward, done):
        self.memory.store_memory(state, action, probs, values, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, action_mask=None):
        state = torch.tensor([observation], dtype=torch.float32).to(self.device)

        logits = self.actor(state)

        if action_mask is not None:
            mask = torch.tensor(action_mask, dtype=torch.float32).to(self.device)
            # Apply mask: set masked logits to a very large negative value
            masked_logits = (
                logits + (mask + 1e-10).log()
            )  # Log of mask to create -inf for 0 values

            dist = Categorical(logits=masked_logits)
        else:
            dist = Categorical(logits=logits)

        value = self.critic(state)
        action = dist.sample()

        log_probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, log_probs, value

    def learn(self):
        step = 0
        for _ in range(self.n_epochs):
            (
                state_arr,
                action_arr,
                old_logprobs_arr,
                values_arr,
                reward_arr,
                dones_arr,
                batches,
            ) = self.memory.generate_batches()

            values = values_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Calculate advantages using GAE
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (
                        reward_arr[k]
                        + self.gamma * values[k + 1] * (1 - int(dones_arr[k]))
                        - values[k]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)

            # Convert arrays to tensors outside of the loop
            state_tensor = torch.tensor(state_arr, dtype=torch.float32).to(
                self.actor.device
            )
            old_logprobs_tensor = torch.tensor(old_logprobs_arr).to(self.actor.device)
            action_tensor = torch.tensor(action_arr).to(self.actor.device)

            for batch in batches:
                states = state_tensor[batch]
                old_logprobs = old_logprobs_tensor[batch]
                actions = action_tensor[batch]

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                dist = Categorical(logits=dist)
                new_logprobs = dist.log_prob(actions)
                prob_ratio = torch.exp(new_logprobs - old_logprobs)

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantage[batch]
                )

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = ((returns - critic_value) ** 2).mean()

                entropy_loss = dist.entropy().mean()  # Entropy bonus
                total_loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    - self.ent_coef * entropy_loss
                )

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                total_loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor.optimizer.step()
                self.critic.optimizer.step()
                # Step the schedulers
                self.actor_scheduler.step()
                self.critic_scheduler.step()

                # self.log_metrics(
                #     total_loss, actor_loss, critic_loss, entropy_loss, step
                # )
                # self.log_gradients(self.actor, step)
                # self.log_gradients(self.critic, step)
                # self.log_parameters(self.actor, step)
                # self.log_parameters(self.critic, step)
                step += 1

        self.memory.clear_memory()
