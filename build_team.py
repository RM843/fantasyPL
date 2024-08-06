def get_top_players(df,season,current_gw,gws=5):
    return df[(df["season_x"] == season) & (df["GW"] > current_gw) &
                    (df["GW"] > current_gw + gws)].groupby("element")[
        'prediction'].sum().reset_index().sort_values(by="prediction",ascending=False)