import os

import rpy2
import rpy2.robjects as ro
import sqlite3
import pandas as pd

# Set a user-writable library path (if needed)
os.environ["R_LIBS_USER"] = "C:/Users/mcevo/R/win-library/4.4"  # Adjust for your username
os.environ['R_HOME'] = "C:/Program Files/R/R-4.4.2"

# Step 1: Install R packages script
install_r_packages_script = """
# Ensure 'devtools' is installed first if not already
if (!requireNamespace("devtools", quietly = TRUE)) {
    install.packages("devtools", lib = Sys.getenv("R_LIBS_USER"))
}

# Install other required packages
required_packages <- c( "worldfootballR")
for (pkg in required_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        if (pkg == "worldfootballR") {
            devtools::install_github("JaseZiv/worldfootballR", lib = Sys.getenv("R_LIBS_USER"))
        } else {
            install.packages(pkg, lib = Sys.getenv("R_LIBS_USER"))
        }
    }
}
"""


# Function to ensure R packages are installed
def install_r_packages():
    try:
        print("Ensuring required R packages are installed...")
        ro.r(install_r_packages_script)
        print("R packages installation complete.")
    except Exception as e:
        print(f"Error installing R packages: {e}")
        raise


# Step 2: Run the data scraping script
scrape_r_script = """
library(worldfootballR)

# Scrape player statistics for the Premier League (2022/2023 season)
player_stats <- fb_player_season_stats(
    country = "ENG",         # Country: England
    gender = "M",            # Gender: Male
    season_end_year = 2023,  # Season ending year
    tier = "1st",            # Tier: Premier League
    stat_type = "standard"   # Type of statistics: Standard
)

# Return the DataFrame
player_stats
"""


# Function to scrape player stats
def scrape_player_stats():
    try:
        print("Running R script to scrape player statistics...")
        ro.r(scrape_r_script)
        r_df = ro.r("player_stats")
        print("Data scraping complete.")
        return r_df
    except Exception as e:
        print(f"Error running R script: {e}")
        return None


# Function to save data to SQLite
def save_to_sqlite(r_df, db_name="premier_league.db", table_name="player_stats"):
    try:
        # Convert R DataFrame to pandas DataFrame
        df = rpy2.robjects.pandas2ri.rpy2py_dataframe(r_df)

        print(f"Saving data to SQLite database: {db_name}")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Dynamically create the table
        columns = ", ".join([f"{col} TEXT" for col in df.columns])
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});")

        # Save DataFrame to SQLite
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Data saved to SQLite table '{table_name}' in database '{db_name}'.")
    except Exception as e:
        print(f"Error saving to SQLite: {e}")
    finally:
        conn.close()


# Main execution
if __name__ == "__main__":
    try:
        # Step 1: Install required R packages
        install_r_packages()

        # Step 2: Scrape data
        data = scrape_player_stats()

        if data is not None:
            print("Scraped data preview:")
            print(data.head())  # Display top rows for verification

            # Step 3: Save data to SQLite
            save_to_sqlite(data)
        else:
            print("Failed to scrape data.")
    except Exception as main_error:
        print(f"An error occurred: {main_error}")
