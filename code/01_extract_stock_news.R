# This is an R script to pull information from the Stock News API
# with dates from January 1, 2022 to July 15, 2024
# The data was pulled on July 15 at 10:32 AM Eastern Time

# The base URL for the API and we got all 100 pages of data
# The number of pages was obtained by observing the total_pages value at
# https://stocknewsapi.com/api/v1/category?section=alltickers&items=100&date=01012022-07152024&token=grdnjugttmiwoeclidaphlunyv0yawn42sczkhjz&page=1
base_url = "https://stocknewsapi.com/api/v1/category?section=alltickers&items=100&date=01012022-07152024&token=&page="
all_urls = paste0(base_url, 1:100)

# Pulling the data from the API
historical_data = purrr::map_df(all_urls, ~jsonlite::fromJSON(.x)$data)

# Writing the data frame as a RDS file
readr::write_rds(historical_data, "data/stock_news_data.rds")

