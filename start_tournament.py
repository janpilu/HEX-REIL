import os
from fhtw_hex.langela_marcon.tournament import tournament

tournament_df = tournament(7, include_dojo=True)
print(tournament_df.head())
print(tournament_df.tail())
# Print total wins, min wins, max wins, mean wins, and standard deviation of wins
print(tournament_df["total_wins"].describe())
# Print the top 3 models for total wins

# delete the models with a total win less than 100
paths = tournament_df[tournament_df["total_wins"] < 100]["agent_path"]
for path in paths:
    if path.split("/")[3] == "ppo":
        os.remove(f"{path}-actor")
        os.remove(f"{path}-critic")
        print(f"Deleted {path}-actor")
        print(f"Deleted {path}-critic")
    else:
        os.remove(path)
        print(f"Deleted {path}")

tournament_df.to_csv("tournament_results.csv", index=False)
