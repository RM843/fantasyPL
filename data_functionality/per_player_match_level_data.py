from data_functionality.database import  fb_advanced_match_stats

def get_player_match_data(season_from="2021-2022", season_to="2022-2023", country="ENG", tier="1st"):
    """
    Fetches player match data for a given range of seasons, country, and league tier.

    Data from load_fb_advanced_match_stats:https://jaseziv.github.io/worldfootballR/reference/load_fb_advanced_match_stats.html

    Args:
        season_from (str, optional): The starting season in "YYYY-YYYY" format. Defaults to "2021-2022".
        season_to (str, optional): The ending season in "YYYY-YYYY" format. Defaults to "2022-2023".
        country (str, optional): The country code for the league (e.g., "ENG" for England). Defaults to "ENG".
        tier (str, optional): The tier of the league (e.g., "1st" for the top division). Defaults to "1st".

    Returns:
        pandas.DataFrame: A DataFrame containing player match data with the first column and all columns from the 20th onward.
    """
    df = fb_advanced_match_stats(
        season_from=season_from, season_to=season_to, country=country, tier=tier
    )
    return df[[df.columns[0]] + list(df.columns)[19:]]


if __name__ == '__main__':
    df = get_player_match_data()

    print("here")