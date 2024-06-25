import os
import pandas as pd
from .modules.train_modules import get_agents
from .tournament import evaluate_agent, score_to_df


def train_group(src, tgt, board_size):

    score_df = pd.DataFrame()

    group = 0

    for group in range(20):
        agent_dict = get_agents(board_size, src, as_path_agent_dict=True)
        agent_eval_map = get_agents(board_size, src)

        policies = [agent.get_action for agent in list(agent_dict.values())]
        policy = lambda board, action_set, current_game: policies[
            (current_game // 2)
            % len(
                policies
            )  # // 2 to ensure that the same policy is used for two games (both as white and black)
        ](board)

        for path, agent in agent_dict.items():
            done = False
            round = 0
            agent_dir = f'./fhtw_hex/langela_marcon/models/{board_size}x{board_size}/{tgt}/{group}/{path.split("/")[-3]}/{path.split("/")[-2]}'
            agent_path = f'{agent_dir}/{path.split("/")[-1]}'
            best_score = 0
            new_score = 0
            while not done:

                agent.env.opponent_policy = policy
                agent.train(1000)

                score = evaluate_agent(
                    board_size=board_size,
                    agent=agent,
                    opponents=agent_eval_map,
                    path=path,
                )

                summary_df = score_to_df(score)

                score_df = pd.concat([score_df, summary_df])
                new_score = score_df["total_wins"].max()

                if new_score > best_score:
                    print("New score is better than previous score")
                    print("Saving checkpoint")

                    best_score = new_score

                    # Save model checkpoint
                    if os.path.exists(f"{agent_dir}/checkpoint"):
                        os.remove(f"{agent_dir}/checkpoint")
                    agent.save(f"{agent_dir}/checkpoint")

                if new_score > len(policies) * 1.2:
                    print("Saving model")

                    done = True
                    agent.save(f"{agent_path}-{new_score}-group-{group}")

                if round > 25:
                    print("Failed to train model")
                    done = True
                    agent.load(f"{agent_dir}/checkpoint")
                    agent.save(f"{agent_path}-failed")
                round += 1

        src = (
            f"./fhtw_hex/langela_marcon/models/{board_size}x{board_size}/{tgt}/{group}"
        )
