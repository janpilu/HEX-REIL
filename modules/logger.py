import csv

class EvaluationLogger:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.lr = 0
        fieldnames = [
            'episode',
            'episode_length',
            'episode_reward',
            'learning_rate'
            # 'loss',
            # 'evaluation_score',
            # 'policy_entropy',
            # 'value_loss',
            # 'action_probability'
        ]
        self.writer.writerow(fieldnames)
    
    def set_lr(self, lr):
        self.lr = lr

    def log(self, episode, episode_length, episode_reward, 
            # loss, evaluation_score, policy_entropy, value_loss, action_probability
            ):
        data = [
            episode,
            episode_length,
            episode_reward,
            self.lr
            # loss,
            # evaluation_score,
            # policy_entropy,
            # value_loss,
            # action_probability
        ]
        self.writer.writerow(data)

    def close(self):
        self.file.close()
