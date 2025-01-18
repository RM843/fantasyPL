# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages("devtools")
devtools::install_github("JaseZiv/worldfootballR")
library(worldfootballR)

# Scrape player statistics for the Premier League (2022/2023 season)
# player_stats <- fb_match_results(
#     country = "ENG",         # Country: England
#     gender = "M",            # Gender: Male
#     season_end_year = 2023,  # Season ending year
#     tier = "1st",            # Tier: Premier League
#     non_dom_league_url=NA
# )

args <- commandArgs(trailingOnly = TRUE)

# Extract method name
method_name <- args[1]

# Function to handle keyword arguments
parse_keywords <- function(args) {
  kw_args <- list()
  for (i in seq(2, length(args), by = 2)) {
    key <- args[i]
    value <- args[i + 1]
    kw_args[[key]] <- value
  }
  return(kw_args)
}

# Extract keyword arguments
kw_args <- parse_keywords(args)

# Define a function to handle different method calls dynamically
run_method <- function(method_name, kw_args) {
  if (method_name == "example_method") {
    result <- example_method(kw_args)
  } else {
    stop("Method not found")
  }
  return(result)
}

# Example method to demonstrate functionality with keyword arguments
example_method <- function(kw_args) {
  # Extract keyword arguments
  name <- kw_args[["name"]]
  n <- as.numeric(kw_args[["number"]])

  df <- data.frame(
    Name = c(name, paste0(name, "_Bob")),
    Age = c(n, n + 5),
    Score = c(n * 10, (n + 5) * 10)
  )
  return(df)
}

# Run the specified method
tryCatch({
  result_df <- run_method(method_name, kw_args)
  write.csv(result_df, "output.csv", row.names = FALSE)
  cat("Success")  # Output just "Success" without newlines or extra text
}, error = function(e) {
  cat("Failure")  # Output just "Failure"
})
