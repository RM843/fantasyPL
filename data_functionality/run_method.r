# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages("devtools")
devtools::install_github("JaseZiv/worldfootballR")
library(worldfootballR)

# Function to handle keyword arguments
parse_keywords <- function(args) {
  if (length(args) <= 1) {
    # If only the method name is passed, return an empty list
    return(list())
  }

  kw_args <- list()
  for (i in seq(2, length(args), by = 2)) {
    key <- args[i]
    value <- args[i + 1]

    # Convert string "NA" to R's NA
    if (value == "NA") {
      value <- NA
    }
    kw_args[[key]] <- value
  }
  return(kw_args)
}
# Extract command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Extract method name
method_name <- args[1]

# Extract keyword arguments
kw_args <- parse_keywords(args)

# Dynamically call the method
tryCatch({
  # Prepare the function call expression
  func_call <- paste0(method_name, "(", paste(names(kw_args), kw_args, sep = " = ", collapse = ", "), ")")

  # Evaluate the method call dynamically
  result <- eval(parse(text = func_call))

  # Save the result to CSV
  write.csv(result, "output.csv", row.names = FALSE, fileEncoding = "UTF-8")

  # Print success message
  cat("Success")
}, error = function(e) {
  # In case of error, print Failure
  cat("Failure: ", e$message)
})
