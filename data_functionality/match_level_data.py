import os

import pandas as pd
from tqdm import tqdm

from data_functionality.database import get_and_save_df, get_data_from_rds_file, \
    get_data_from_r, fb_advanced_match_stats
from helper_methods import df_to_nested_dict
from run_r import run_r_method

TEMP_LIM = 15
def get_match_data_for_seasons_between_inc(season_from = "2021-2022",season_to="2022-2023",country="ENG",tier="1st",gender="M"):


    df = get_and_save_df(create_df_function=lambda:_load_match_results(country),
                         file_path=fr"data/worldfootballR/misc/matches_{country}.csv")
    df.drop_duplicates(inplace=True)
    df["Season"] = (df["Season_End_Year"] - 1).astype(str) + "-" + df["Season_End_Year"].astype(str)
    df = df[(df["Country"] == country) &
            (df["Gender"] == gender) &
            (df["Round"] == tier) &
            (df["Season_End_Year"] <= int(season_to[-4:])) &
            (df["Season_End_Year"] > int(season_from[:4]))]


    df2 = get_match_data_fb_advanced_stats(season_from = season_from,season_to=season_to,country=country,tier=tier)
    df = df.merge(df2,on="MatchURL",how="outer")
    df3 = get_match_stats_for_match_urls(match_urls=list(df["MatchURL"].unique())[:TEMP_LIM])
    return df

def get_match_data_fb_advanced_stats(season_from = "2021-2022",season_to="2022-2023",country="ENG",tier="1st"):
    return fb_advanced_match_stats(season_from = season_from, season_to=season_to, country=country, tier=tier).iloc[:,
          :19].drop_duplicates()
def _load_match_results(country):
    base_url = f"https://github.com/JaseZiv/worldfootballR_data/releases/download/match_results/{country}_match_results.rds"
    df = get_data_from_rds_file(base_url)
    df.drop(columns=[5], inplace=True)
    cols = list(get_data_from_r("load_match_results", country=f"'{country}'", gender="'M'", season_end_year=2020,
                                tier=f"'1st'").columns)
    df.columns = cols
    return df

def get_match_stats_for_match_urls(match_urls,update=False):

    stats = get_data_from_match_url_method("fb_team_match_stats", match_url=match_urls)
    lineups = get_lineups_for_match(match_url=match_urls)
    df = stats.merge(lineups,on="MatchURL",how="inner")
    assert( df.shape[0]==stats.shape[0])&( df.shape[0]==lineups.shape[0])
    return df

def get_data_from_match_url_method(method_name,match_url):
    file_path = fr"data/worldfootballR/misc/{method_name}.csv"
    df = pd.DataFrame(columns=["MatchURL"])
    if os.path.exists(file_path) :
        df = pd.read_csv(file_path)

    if match_url is str:
        match_url = [match_url]
    previously_complete = df["MatchURL"].isin(match_url)
    if df[~previously_complete].shape[0]==0:
        return df[previously_complete]
    else:
        for url in tqdm([url for url in match_url if url not in df["MatchURL"].to_list()]):
            out = run_r_method(method_name, **dict(match_url=f"'{url}'"))
            out.rename(columns={"Game_URL":"MatchURL"},inplace=True)
            assert "MatchURL" in out
            df = pd.concat([df,out])
            df.to_csv(file_path, index=False)

        return df[df["MatchURL"].isin(match_url)]

def get_lineups_for_match(match_url):
    lineups = get_data_from_match_url_method("fb_match_lineups", match_url=match_url)
    lineups = lineups.groupby(["MatchURL", "Home_Away", "Starting"]).agg({'Player_Name': ', '.join}).reset_index()
    lineups = lineups.merge(lineups.loc[(lineups["Home_Away"]=="Home")&(lineups["Starting"]=="Pitch"),["MatchURL","Player_Name"]],
                                                                                           on="MatchURL",how='left')
    lineups.rename(columns={"Player_Name_x":"Player_Name","Player_Name_y":"Home_Lineup"},inplace=True)
    assert lineups.shape[0]!=0

    lineups = lineups.merge(
        lineups.loc[(lineups["Home_Away"] == "Away") & (lineups["Starting"] == "Pitch"), ["MatchURL", "Player_Name"]],
        on="MatchURL", how='left')
    lineups.rename(columns={"Player_Name_x": "Player_Name", "Player_Name_y": "Away_Lineup"}, inplace=True)
    assert lineups.shape[0] != 0

    lineups = lineups.merge(
        lineups.loc[(lineups["Home_Away"] == "Home") & (lineups["Starting"] == "Bench"),["MatchURL","Player_Name"]],on="MatchURL", how='left')
    assert lineups.shape[0] != 0
    lineups.rename(columns={"Player_Name_x": "Player_Name", "Player_Name_y": "Home_Subs"}, inplace=True)

    lineups = lineups.merge(
        lineups.loc[(lineups["Home_Away"] == "Away") & (lineups["Starting"] == "Bench"),["MatchURL","Player_Name"]],on="MatchURL", how='left')
    assert lineups.shape[0] != 0
    lineups.rename(columns={"Player_Name_x": "Player_Name", "Player_Name_y": "Away_Subs"}, inplace=True)

    return lineups[["MatchURL","Home_Lineup","Home_Subs","Away_Lineup","Away_Subs"]].drop_duplicates()

if __name__ == '__main__':
    df = get_match_data_for_seasons_between_inc()
    print("HERE")