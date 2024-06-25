import os

from tqdm import tqdm
from .modules.train_modules import get_agents
from .fhtw_hex.hex_engine import hexPosition
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


def merge_agent_dicts(agents1, agents2):
    for model, architectures in agents2.items():
        if model not in agents1:
            agents1[model] = architectures
        else:
            for architecture, agent_datas in architectures.items():
                if architecture not in agents1[model]:
                    agents1[model][architecture] = agent_datas
                else:
                    agents1[model][architecture].extend(agent_datas)
    return agents1


def tournament(board_size, include_dojo=False):
    agents = get_agents(board_size)

    if include_dojo:
        dojo_dir = f"./fhtw_hex/langela_marcon/models/{board_size}x{board_size}/dojo"
        for level in os.listdir(dojo_dir):
            if not os.path.isdir(f"{dojo_dir}/{level}"):
                continue
            level_agents = get_agents(board_size, f"{dojo_dir}/{level}")
            agents = merge_agent_dicts(agents, level_agents)

    agent_path_map = {}
    all_results = []
    for model, architectures in tqdm(agents.items(), desc="Tournament", position=0):
        for architecture, agent_datas in tqdm(
            architectures.items(),
            desc="Evaluating architectures",
            position=1,
            leave=False,
        ):
            for agent_data in tqdm(
                agent_datas, desc="Evaluating agents", position=2, leave=False
            ):
                agent = agent_data["agent"]
                agent_path = agent_data["path"]
                agent_path_map[agent_path] = agent
                results = evaluate_agent(board_size, agent, agent_path, agents)
                all_results.extend(results)

    summary_df = score_to_df(all_results)

    save_best_models(summary_df, agent_path_map)

    return summary_df


def save_best_models(summary_df, agents_path_map, top=4, board_size=7):
    # Get the best X models for each category (total wins, white wins, black wins) without duplicates
    summary_df.sort_values("total_wins", ascending=False, inplace=True)
    best_models = summary_df.head(top)

    summary_df = summary_df.tail(-top)

    summary_df.sort_values("white", ascending=False, inplace=True)
    best_white_models = summary_df.head(top)
    summary_df = summary_df.tail(-top)

    summary_df.sort_values("black", ascending=False, inplace=True)
    best_black_models = summary_df.head(top)

    best_merge = pd.concat([best_models, best_white_models, best_black_models])
    print(best_merge)

    # Copy the best models to a winner folder
    for index, row in best_merge.iterrows():
        agent_path = row["agent_path"]
        agent = agents_path_map[agent_path]
        agent_dir = f'./fhtw_hex/langela_marcon/models/{board_size}x{board_size}/winner/{agent_path.split("/")[-3]}/{agent_path.split("/")[-2]}'
        agent_path = f'{agent_dir}/{agent_path.split("/")[-1]}'

        os.makedirs(agent_dir, exist_ok=True)

        if not os.path.exists(agent_path):
            agent.save(agent_path)

    print("Best models saved!")


if __name__ == "__main__":
    tournament_df = tournament(7)
    print(tournament_df.head())
    print(tournament_df.tail())

    # print(tournament_df["total_wins"].mean())
    # print(tournament_df["total_wins"].median())
    # delete models with a total win less than 100

    # paths = tournament_df[tournament_df["total_wins"] < 100]["agent_path"]

    # for path in paths:
    #     if path.split("/")[3] == "ppo":
    #         os.remove(f"{path}-actor")
    #         os.remove(f"{path}-critic")
    #         print(f"Deleted {path}-actor")
    #         print(f"Deleted {path}-critic")
    #     else:
    #         os.remove(path)
    #         print(f"Deleted {path}")
