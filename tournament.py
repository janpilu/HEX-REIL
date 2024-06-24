import os
from modules.train_modules import get_agents
from fhtw_hex.hex_engine import hexPosition
import pandas as pd


def evaluate_agent(board_size, agent, path, opponents):
    results = []
    agent_policy = agent.policy
    engine = hexPosition(board_size)
    for model, architectures in opponents.items():
        for architecture, opponent_datas in architectures.items():
            for opponent_data in opponent_datas:
                opponent = opponent_data["agent"]
                opponent_policy = opponent.policy

                # Agent plays as white
                engine.machine_vs_machine(
                    agent_policy, opponent_policy, verbose=False, autoplay=True
                )
                winner = engine.winner
                results.append(
                    {
                        "agent_path": path,
                        "opponent_path": opponent_data["path"],
                        "color": "white",
                        "win": winner == 1,
                    }
                )

                # Agent plays as black
                engine.machine_vs_machine(
                    opponent_policy, agent_policy, verbose=False, autoplay=True
                )
                winner = engine.winner
                results.append(
                    {
                        "agent_path": path,
                        "opponent_path": opponent_data["path"],
                        "color": "black",
                        "win": winner == -1,
                    }
                )

    return results


def score_to_df(score):
    # Create DataFrame
    df = pd.DataFrame(score)

    # Aggregate results to get wins and losses
    summary_df = df.groupby(["agent_path", "color"]).win.sum().unstack().fillna(0)
    summary_df["total_wins"] = summary_df["white"] + summary_df["black"]

    # Replace NaN values with 0
    summary_df = summary_df.fillna(0).astype(int)

    # Reset index for a cleaner dataframe
    summary_df.reset_index(inplace=True)

    # Sort by total wins
    summary_df = summary_df.sort_values("total_wins", ascending=False)
    return summary_df


def tournament(board_size):
    agents = get_agents(board_size)
    agent_path_map = {}
    all_results = []
    for model, architectures in agents.items():
        for architecture, agent_datas in architectures.items():
            for agent_data in agent_datas:
                agent = agent_data["agent"]
                agent_path = agent_data["path"]
                agent_path_map[agent_path] = agent
                results = evaluate_agent(board_size, agent, agent_path, agents)
                all_results.extend(results)

    summary_df = score_to_df(all_results)

    save_best_models(summary_df, agent_path_map)

    return summary_df


def save_best_models(summary_df, agents_path_map):
    # Get the best 3 models for each category (total wins, white wins, black wins) without duplicates
    summary_df.sort_values("total_wins", ascending=False, inplace=True)
    best_models = summary_df.head(3)
    # copy excluding the head 3
    summary_df = summary_df.tail(-3)

    summary_df.sort_values("white", ascending=False, inplace=True)
    best_white_models = summary_df.head(3)
    summary_df = summary_df.tail(-3)

    summary_df.sort_values("black", ascending=False, inplace=True)
    best_black_models = summary_df.head(3)

    best_merge = pd.concat([best_models, best_white_models, best_black_models])
    print(best_merge)

    # Copy the best models to a winner folder
    for index, row in best_merge.iterrows():
        agent_path = row["agent_path"]
        agent = agents_path_map[agent_path]
        agent_dir = f'models/{agent_path.split("/")[-4]}/winner/{agent_path.split("/")[-3]}/{agent_path.split("/")[-2]}'
        agent_path = f'{agent_dir}/{agent_path.split("/")[-1]}'

        os.makedirs(agent_dir, exist_ok=True)

        if not os.path.exists(agent_path):
            agent.save(agent_path)

    print("Best models saved!")


# tournament_df = tournament(7)
# # delete models with a total win less than 100

# paths = tournament_df[tournament_df["total_wins"] < 100]["agent_path"]

# for path in paths:
#     os.remove(path)
#     print(f"Deleted {path}")
