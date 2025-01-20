import shutil
import subprocess
import os
import pandas as pd


def run_r_script(script_path, method_name, **kwargs):
    """
    Runs an R script with the method name and keyword arguments, returns success or failure.
    :param script_path: Path to the R script
    :param method_name: Method name to call in the R script
    :param kwargs: Keyword arguments to pass to the R script
    :return: Success message or Failure message
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"The R script '{script_path}' does not exist.")
        # Check if Rscript is available
    rscript_path = shutil.which("Rscript")
    if rscript_path is None:
        raise EnvironmentError("Rscript executable not found. Ensure R is installed and added to PATH.")
    try:
        # Construct the command to run the R script with the method name and keyword arguments
        command = ["Rscript", script_path, method_name]

        # Add keyword arguments as --key value pairs
        for key, value in kwargs.items():
            command.append(key)
            command.append(str(value))  # Ensure the value is a string

        # Run the R script and capture the output
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Return the output (Success or Failure)
        return result.stdout.strip()

    except FileNotFoundError:
        raise FileNotFoundError("Rscript executable not found. Ensure R is installed and added to PATH.")



def run_method_and_load_csv(r_script, method_name, **kwargs):
    """
    Run the R script with the method name and keyword arguments, load the resulting CSV into a DataFrame, and clean up.
    :param r_script: Path to the R script
    :param method_name: Name of the method to invoke
    :param kwargs: Keyword arguments to pass to the R script
    :return: DataFrame or None
    """
    # Step 1: Run the R script
    result = run_r_script(r_script, method_name, **kwargs)

    if result[-7:] == "Success":
        print("R script ran successfully. Loading CSV...")

        # Load the resulting CSV into a pandas DataFrame
        df = pd.read_csv("output.csv")

        # Step 2: Delete the temporary CSV file
        os.remove("output.csv")

        # Return the DataFrame
        return df

    else:
        print(f"R script failed: {result}")
        return None



# Scrape player statistics for the Premier League (2022/2023 season)
# player_stats <- fb_match_results(
#     country = "ENG",         # Country: England
#     gender = "M",            # Gender: Male
#     season_end_year = 2023,  # Season ending year
#     tier = "1st",            # Tier: Premier League
#     non_dom_league_url=NA
# )

def run_r_method(method,**kwargs):
    r_script_path = "run_method.r"
    return run_method_and_load_csv(r_script_path, method, **kwargs)
# Example usage
if __name__ == "__main__":
    r_script_path = "run_method.r"  # Path to your R script
    method = "fb_match_results"  # Example method name
    kwargs = dict(
            country = "'ENG'",         # Country: England
            gender = "'M'",            # Gender: Male
            season_end_year = 2023,  # Season ending year
            tier = "'1st'",            # Tier: Premier League
            non_dom_league_url="NA"
        )
    df = run_r_method(method,**kwargs)
    # method="fb_match_lineups"
    # kwargs = dict(match_url = "https://fbref.com/en/matches/47880eb7/Liverpool-Manchester-City-November-10-2019-Premier-League")

    # method = "example_method"  # Example method name
    # # Pass keyword arguments
    # kwargs = dict(
    # name = "Alice",
    # number = 10)

    df = run_method_and_load_csv(r_script_path, method, **kwargs)

    if df is not None:
        print("Dataframe from R script:")
        print(df)
    else:
        print("Failed to retrieve data.")
