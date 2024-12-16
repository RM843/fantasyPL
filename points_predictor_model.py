import pandas as pd

from xgboost_trainer import get_xgboost_model, predict

if __name__ == '__main__':

    df = pd.read_csv("fantasyPL/data/cleaned_merged_seasons.csv")

    #         master_df = pd.concat([master_df,df])
    cat_features = ["position"]
    df[cat_features] = df[cat_features].astype('category')

    features = cat_features
    target = "total_points"

    model = get_xgboost_model(df, features, target,objective='reg:squarederror')