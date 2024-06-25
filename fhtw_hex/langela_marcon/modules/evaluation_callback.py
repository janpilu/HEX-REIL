from stable_baselines3.common.callbacks import BaseCallback


class EvalCallback(BaseCallback):
    def __init__(self, eval_logger, verbose=0):
        super(EvalCallback, self).__init__(verbose)
        self.eval_logger = eval_logger
        self.episode = 0

    def _on_step(self) -> bool:
        # Collect metrics after each episode
        if self.locals.get("dones")[0]:
            episode_reward = self.locals.get("infos")[0].get("episode")["r"]
            episode_length = self.locals.get("infos")[0].get("episode")["l"]
            # loss = self.model.logger.get_log_dict().get('train/loss')
            # policy_entropy = self.model.logger.get_log_dict().get('train/policy_entropy')
            # value_loss = self.model.logger.get_log_dict().get('train/value_loss')
            # action_probability = self.model.logger.get_log_dict().get('train/action_probability')

            # avg_reward = episode_reward / episode_length
            # win_rate = np.sum(np.array(episode_rewards) > 0) / len(episode_rewards)  # Example calculation

            self.eval_logger.log(
                self.episode,
                episode_length,
                episode_reward,
                # loss,
                # 0,  # Evaluation score is not available in training
                # policy_entropy,
                # value_loss,
                # action_probability
            )
            self.episode += 1

        return True
