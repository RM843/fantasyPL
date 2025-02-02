import tempfile
import os
os.environ['R_HOME'] = 'C:\Program Files\R\R-4.4.2'

import rpy2.robjects as ro

import requests
from tqdm import tqdm

from run_r import run_r_method


import pandas as pd

STAT_TYPES = ["standard", "shooting", "passing", "passing_types", "gca", "defense", "possession", "playing_time", "misc", "keepers", "keepers_adv"]
COUNTRY_TO_LEAGUE = {"ENG": "EPL"}
TEAMS_MAPPING = {
    'Manchester United': 'Manchester Utd',
    'Leicester': 'Leicester City',
    'Queens Park Rangers': 'QPR',
    'Stoke': 'Stoke City',
    'West Bromwich Albion': 'West Brom',
    'Newcastle United': 'Newcastle Utd',
    'Swansea': 'Swansea City',
    'Hull': 'Hull City',
    'Norwich': 'Norwich City',
    'Wolverhampton Wanderers': 'Wolves',
    'Cardiff': 'Cardiff City',
    'Sheffield United': 'Sheffield Utd',
    'Leeds': 'Leeds United',
    'Nottingham Forest': "Nott'ham Forest",
    'Ipswich':"Ipswich Town",
    'Luton': 'Luton Town'
}

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
        # print(f"File exists. Loading DataFrame from: {file_path}")
        return pd.read_csv(file_path)

    # Create the DataFrame
    df = create_df_function()

    # Save the DataFrame to the specified file
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to: {file_path}")

    return df


    print(f"An error occurred: {str(e)}")
def get_data_from_r(method_name, **method_kwargs):
    method = lambda:run_r_method(method_name,**method_kwargs)
    args_formatted = "!".join([f"{str(k)}%{str(v)}".replace('\'','') for k,v in method_kwargs.items() if k[-3:]!="url" and k!="time_pause"])
    args_formatted = args_formatted if args_formatted!="" else "table"
    return get_and_save_df(create_df_function=method, file_path=fr"data/worldfootballR/{method_name}/{args_formatted}.csv", update=False)

def get_data_from_rds_file(base_url):
    response = requests.get(base_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".rds") as tmp_file:
        tmp_file.write(response.content)
        tmp_file.flush()
        tmp_file_path = tmp_file.name

    # Use R's readRDS
    ro.r['readRDS'](tmp_file_path)
    match_results = ro.r['readRDS'](tmp_file_path)

    # Convert to pandas DataFrame (if needed)
    import pandas as pd
    df = pd.DataFrame(match_results).T
    return df

def fb_advanced_match_stats(season_from ="2021-2022", season_to="2022-2023", country="ENG", tier="1st"):
    all_player_data=None
    for stat_type in tqdm(STAT_TYPES):
        df = get_data_from_r("load_fb_advanced_match_stats",
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






def get_match_shooting_data(country="ENG",tier="1st"):
    assert country in COUNTRY_TO_LEAGUE
    league = COUNTRY_TO_LEAGUE[country]


    # Understat shooting
    understat =get_data_from_r("load_understat_league_shots", league=f"'{league}'")
    understat["season"] = understat["season"].astype(str) + "-" + (understat["season"]+1).astype(str)
    understat["Event_Type"] = "Shot_udst"
    matches = [(x[1], x[2], x[3], x[4]) for x in
               understat[["match_id", "home_team", "away_team", "season"]].drop_duplicates().to_records()]
    mapping = understat_and_fbref_match_mapping(matches)
    understat = understat.merge(mapping[["ID", "MatchURL"]], left_on=["match_id"], right_on=["ID"])


    # Fbref shooting
    fbref = get_data_from_r("load_fb_match_shooting",
                            country=f"'{country}'",
                            gender="'M'",
                            tier=f"'{tier}'"
                            )
    fbref["season"] = (fbref["Season_End_Year"] - 1).astype(str) + "-" + fbref["Season_End_Year"].astype(str)
    fbref["Event_Type"] = "Shot_fbref"


    # merge both shooting stats crudely
    return  pd.concat([fbref,understat])

def get_match_big_events_data(country="ENG",tier="1st"):
    df = get_data_from_r("load_fb_match_summary",
                         country=f"'{country}'",
                         gender="'M'",
                         tier=f"'{tier}'"
                         )
    df["season"] = (df["Season_End_Year"] - 1).astype(str) + "-" + df["Season_End_Year"].astype(str)
    return df

def get_player_match_data(season_from = "2021-2022",season_to="2022-2023",country="ENG",tier="1st"):
   df =  _get_player_match_data(season_from=season_from, season_to=season_to, country=country, tier=tier)
   lineups = get_data_from_r(" fb_match_lineups", match_url=match_url)
   return df.iloc[:, 21:]



def get_match_event_data_for_seasons_between_inc(season_from = "2021-2022",season_to="2022-2023",country="ENG",tier="1st"):
    shooting_events = get_match_shooting_data(country=country,tier=tier)
    big_events = get_match_big_events_data(country=country, tier=tier)
    df = pd.concat([shooting_events,big_events])
    df["Season_End_Year"] = df["season"].str[:4].astype(int)
    return df[(df["Season_End_Year"] <= int(season_to[-4:])) & (
                df["Season_End_Year"] >= int(season_from[-4:]))]

def get_player_data_for_seasons_between_inc(season_from = "2021-2022",season_to="2022-2023"):
    """Loading preloaded data"""
    all_player_data=None

    for stat_type in tqdm(STAT_TYPES):

        df = get_data_from_r("load_fb_big5_advanced_season_stats", stat_type=f"'{stat_type}'", team_or_player="'player'")
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


def understat_and_fbref_match_mapping(matches):
    """

    :param matches: list of tuples in form (id,home_team,away_team,season)
    :return:
    """
    join = ["Home", "Away", "Season"]

    season_ends = [int(x[-1][-4:]) for x in matches]
    season_from = f"{min(season_ends) - 1}-{min(season_ends)}"
    season_to = f"{max(season_ends) - 1}-{max(season_ends)}"
    team_names = get_team_names_master_list(
        season_from=season_from,
        season_to=season_to,
    )

    matches = pd.DataFrame(matches, columns=["ID"]+join)

    for col in ["Home","Away"]:
        matches[col] =  matches[col].apply(lambda x: TEAMS_MAPPING.get(x,x))



    fbref_matches = get_match_data_for_seasons_between_inc(season_from=season_from,season_to=season_to,country="ENG",tier="1st")
    df = matches.merge(fbref_matches[join+["MatchURL"]], on=join,how="left")

    unmapped_teams = [x for x in df["Home"].unique() if x not in team_names]
    assert len(unmapped_teams)==0,f"Teams not mapped{','.join(unmapped_teams)}"
    assert df.shape[0]==len(matches)
    assert df[df["MatchURL"].isna()].shape[0]==0

    return df




def get_team_names_master_list(season_from = "2021-2022",season_to="2022-2023",country="ENG",tier="1st"):
    return list(get_match_data_for_seasons_between_inc(season_from=season_from, season_to=season_to, country=country, tier=tier)["Home"].unique())


def get_match_summary(match_url): # NO NEW DATA
    return get_data_from_match_url_method("fb_match_summary", match_url)




if __name__ == '__main__':
    print(os.environ['R_HOME'])
    #f"https://github.com/JaseZiv/worldfootballR_data/releases/download/match_results/{country}_match_results.rds", f"https://github.com/JaseZiv/worldfootballR_data/raw/master/data/match_results/{country}_match_results.rds"
    # df = get_match_summary(  match_url="https://fbref.com/en/matches/2df9a3a1/Aston-Villa-Brighton-and-Hove-Albion-September-30-2023-Premier-League")
    match_data = get_match_data_for_seasons_between_inc()

    player_match_data = get_player_match_data() # Player match summaries

    match_event_data = get_match_event_data_for_seasons_between_inc() # all events that occur in each match

    player_data =get_player_data_for_seasons_between_inc()

    teams_by_season = get_teams_for_seasons_between_inc()


    print("HEre")


    # teams
    # players
    # matches
