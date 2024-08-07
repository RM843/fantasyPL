import pandas as pd

from build_team import get_top_players
from xgboost_trainer import get_xgboost_model, predict

df = pd.read_csv("data/cleaned_merged_seasons.csv")
# master_df =pd.DataFrame()
# seasons = list_folders("data")
# for season in seasons:
#     players  = list_folders("data/"+season+"/players")
#     for player in players:
#         df = pd.read_csv("data/"+season+"/players/"+player+"/gw.csv")
#         df["GW"] = list(df.index+1)
#         df["season"] = season
#         assert df.shape[0] ==38,f"{df.shape[0]},{season}"
#         master_df = pd.concat([master_df,df])
cat_features = ["position"]
df[cat_features] = df[cat_features].astype('category')

features = cat_features
target = "total_points"

model = get_xgboost_model(df, features, target)
next_n = 5
all_gws = list(set(list(pd.MultiIndex.from_frame(df[["season_x", "GW"]]))))
df["prediction"] = predict(model=model, dtest=df)
projections = pd.DataFrame()
for season, gw in all_gws:
    # players =  df.loc[(df["season_x"] == season), "element"].unique()
    # for player in players:
        # assert df.loc[(df["season_x"] == season)& (df["element"] == player)].shape[0]<=38
    tmp = get_top_players(df=df,season=season,current_gw=gw)
    tmp["GW"] = season+"_"+str(gw)
    projections = pd.concat([projections,tmp])

for season, gw in all_gws:
    print("here")
