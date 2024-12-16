install.packages("devtools")
devtools::install_github("JaseZiv/worldfootballR")
install.packages("dplyr")

# Then load the package
library(worldfootballR)


library(dplyr)

mapped_players <- player_dictionary_mapping()
dplyr::glimpse(mapped_players)

View(mapped_players)
