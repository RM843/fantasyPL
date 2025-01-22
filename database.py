from tqdm import tqdm

from run_r import run_r_method

import os
import pandas as pd

STAT_TYPES = ["standard", "shooting", "passing", "passing_types", "gca", "defense", "possession", "playing_time", "misc", "keepers", "keepers_adv"]
def get_and_save_df(file_path, create_df_function, update=False):
    """
    Save or update a DataFrame to a specified file location.

    Parameters:
        file_path (str): Path to save or load the DataFrame.
        create_df_function (callable): A function that creates and returns a DataFrame.
        update (bool): If True, updates the file even if it exists.

    Returns:
        pd.DataFrame: The DataFrame loaded from file or created by create_df_function.
    """
    # Ensure the directory structure exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Check if the file exists
    if os.path.exists(file_path) and not update:
        print(f"File exists. Loading DataFrame from: {file_path}")
        return pd.read_csv(file_path)

    # Create the DataFrame
    df = create_df_function()

    # Save the DataFrame to the specified file
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to: {file_path}")

    return df


    print(f"An error occurred: {str(e)}")
def get_data(method_name,**method_kwargs):
    method = lambda:run_r_method(method_name,**method_kwargs)
    args_formatted = "!".join([f"{str(k)}%{str(v)}".replace('\'','') for k,v in method_kwargs.items() if k[-3:]!="url" and k!="time_pause"])
    args_formatted = args_formatted if args_formatted!="" else "table"
    return get_and_save_df(create_df_function=method, file_path=fr"data/worldfootballR/{method_name}/{args_formatted}.csv", update=False)


def get_league_urls_fbref(country,season_from,season_to,tier="1st",gender="M"):
    df = get_and_save_df(create_df_function =lambda: pd.read_csv(r"https://raw.githubusercontent.com/JaseZiv/worldfootballR_data/master/raw-data/all_leages_and_cups/all_competitions.csv"),
                         file_path=fr"data/worldfootballR/misc/leagues.csv")

    df.drop_duplicates(inplace=True)
    df = df[(df["country"] == country) &
           (df["gender"] == gender) &
           (df["tier"] == tier) &
           (df["season_end_year"] <= int(season_to[-4:])) &
           (df["season_end_year"] > int(season_from[:4]))]

    return dict(zip(df["seasons"], df["seasons_urls"]))

def get_league_url_fbref(country,season,tier="1st",gender="M"):
    lst = list(get_league_urls_fbref(country=country,season_to=season,season_from=season,tier=tier,gender=gender).values())
    assert len(lst) ==1
    return lst[0]


def get_teams_for_league_season(country,season,tier="1st",gender="M"):
    league_url = get_league_url_fbref(country=country,season=season,tier=tier,gender=gender)

    # need to set save path, so not using get_data()
    method = lambda: run_r_method("fb_teams_urls",league_url=f"'{league_url}'", time_pause=3)
    df = get_and_save_df(create_df_function=method, file_path=fr"data/worldfootballR/teams/{season}.csv",
                    update=False)
    df["team"] = df["x"].apply(lambda x: x.split("/")[-1][:-6].replace("-"," "))
    df.rename(columns={"x":"team_season_url"},inplace=True)
    df = df[["team","team_season_url"]]
    return df

def get_seasons_between_inc(season_from,season_to):
    return [f"{x-1}-{x}" for x in range(int(season_from[-4:]),int(season_to[-4:])+1)]

def get_teams_for_seasons_between_inc(season_from = "2021-2022",season_to="2022-2023",country="ENG",tier="1st"):
    all_seasons = pd.DataFrame()
    for season in get_seasons_between_inc( season_from =season_from,season_to=season_to):
        teams =  get_teams_for_league_season(country = country,   season =season,tier=tier)
        teams["season"] = season
        all_seasons = pd.concat([all_seasons,teams])
    return all_seasons

def get_match_data_for_seasons_between_inc(season_from = "2021-2022",season_to="2022-2023",country="ENG",tier="1st"):
    all_seasons = pd.DataFrame()
    for season in get_seasons_between_inc(season_from=season_from, season_to=season_to):
        matches = get_data("load_match_results",country=f"'{country}'", gender="'M'", season_end_year=int(season[-4:]), tier =f"'{tier}'")
        matches["season"] = season
        all_seasons = pd.concat([all_seasons, matches])
    return all_seasons

def get_match_shooting_data(country="ENG",tier="1st"):
    df = get_data("load_fb_match_shooting",
                  country=f"'{country}'",
                  gender="'M'",
                  tier=f"'{tier}'"
                  )
    df["season"] = (df["Season_End_Year"] - 1).astype(str) + "-" + df["Season_End_Year"].astype(str)
    df["Event_Type"] = "Shot"
    return df

def get_match_big_events_data(country="ENG",tier="1st"):
    df = get_data("load_fb_match_summary",
                  country=f"'{country}'",
                  gender="'M'",
                  tier=f"'{tier}'"
                  )
    df["season"] = (df["Season_End_Year"] - 1).astype(str) + "-" + df["Season_End_Year"].astype(str)
    return df

def get_player_match_data(season_from = "2021-2022",season_to="2022-2023",country="ENG",tier="1st"):
    all_player_data=None
    for stat_type in tqdm(STAT_TYPES):
        df = get_data("load_fb_advanced_match_stats",
        country = f"'{country}'",
        gender = "'M'",
        tier = f"'{tier}'",
        stat_type=f"'{stat_type}'", team_or_player="'player'")
        if df.shape[0]==0:
            continue
        df["season"] =  (df["Season_End_Year"] -1).astype(str)+"-"+df["Season_End_Year"].astype(str)
        if all_player_data is None:
            all_player_data=df
        else:
            on =  ["MatchURL","Player_Href"]
            all_player_data =pd.merge(all_player_data, df[[x for x in df if( x not in all_player_data) or (x in on)]], on=on, how='outer')
            assert all_player_data[all_player_data.duplicated(subset=on)].shape[0] ==0
    return all_player_data[(all_player_data["Season_End_Year"]<=int(season_to[-4:]))&(all_player_data["Season_End_Year"]>=int(season_from[-4:]))]

def get_match_event_data_for_seasons_between_inc(season_from = "2021-2022",season_to="2022-2023",country="ENG",tier="1st"):
    shooting_events = get_match_shooting_data(country=country,tier=tier)
    big_events = get_match_big_events_data(country=country, tier=tier)
    df = pd.concat([shooting_events,big_events])
    return df[(df["Season_End_Year"] <= int(season_to[-4:])) & (
                df["Season_End_Year"] >= int(season_from[-4:]))]

def get_player_data_for_seasons_between_inc(season_from = "2021-2022",season_to="2022-2023"):
    all_player_data=None
    for stat_type in tqdm(STAT_TYPES):
        df = get_data("load_fb_big5_advanced_season_stats", stat_type=f"'{stat_type}'", team_or_player="'player'")
        if df.shape[0]==0:
            continue
        df["season"] =  (df["Season_End_Year"] -1).astype(str)+"-"+df["Season_End_Year"].astype(str)
        if all_player_data is None:
            all_player_data=df
        else:
            on =  ['season', 'Url',"Squad"]
            all_player_data =pd.merge(all_player_data, df[[x for x in df if( x not in all_player_data) or (x in on)]], on=on, how='outer')
            assert all_player_data[all_player_data.duplicated(subset=on)].shape[0] ==0
    return all_player_data[(all_player_data["Season_End_Year"]<=int(season_to[-4:]))&(all_player_data["Season_End_Year"]>=int(season_from[-4:]))]

if __name__ == '__main__':
    # df = get_data("load_fb_advanced_match_stats",
    #              country=f"'ENG'",
    #              gender="'M'",
    #              stat_type=f"'{STAT_TYPES[0]}'",
    #              tier=f"'1st'",team_or_player = "'player'"
    #              )
    df = get_player_match_data()

    match_event_data = get_match_event_data_for_seasons_between_inc()

    player_data =get_player_data_for_seasons_between_inc()

    teams_by_season = get_teams_for_seasons_between_inc()

    match_data = get_match_data_for_seasons_between_inc()

    print("HEre")


    # teams
    # players
    # matches
