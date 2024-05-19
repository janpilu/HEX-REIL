from stable_baselines3 import PPO


class PPOAgent:
    def set_env(self, env):
        self.env = env

    def init_model(self):
        self.model = PPO("MlpPolicy", self.env, verbose=0)

    def train(self, steps):
        self.model.learn(total_timesteps=steps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path)
        self.model.set_env(self.env)

    def evaluate(self, steps):
        obs = self.env.reset()
        for i in range(steps):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)
            self.env.render()

    def get_action(self, board, action_set):
        valid_action = False
        while not valid_action:
            action, _states = self.model.predict(board)
            if action in action_set:
                return action
