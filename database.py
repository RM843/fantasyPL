from run_r import run_r_method

import os
import pandas as pd


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

    method = lambda: run_r_method("fb_teams_urls",league_url=f"'{league_url}'", time_pause=3)
    df = get_and_save_df(create_df_function=method, file_path=fr"data/worldfootballR/teams/{season}.csv",
                    update=False)
    df["team"] = df["x"].apply(lambda x: x.split("/")[-1][:-6].replace("-"," "))
    df.rename(columns={"x":"team_season_url"},inplace=True)
    df = df[["team","team_season_url"]]
    return df

def get_seasons_between_inc(season_from,season_to):
    return [f"{x-1}-{x}" for x in range(int(season_from[-4:]),int(season_to[-4:])+1)]

if __name__ == '__main__':
    # league_urls_by_season = get_league_urls_fbref(country = "ENG",   season_from = "2021-2022",season_to="2022-2023")

    teams_by_season = {}
    for season in get_seasons_between_inc( season_from = "2021-2022",season_to="2022-2023"):
        teams_by_season[season]=  get_teams_for_league_season(country = "ENG",   season =season)



    # save_df(file_location="data/worldfootballR/")
    # df = get_data( "fb_match_results",  country="'ENG'",  # Country: England
    #     gender="'M'",  # Gender: Male
    #     season_end_year=2023,  # Season ending year
    #     tier="'1st'",  # Tier: Premier League
    #     non_dom_league_url="NA")
    #
    # df = get_data("player_dictionary_mapping",**{})
    print("HEre")


    # teams
    # players
    # matches
