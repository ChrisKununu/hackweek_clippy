# hackweek_clippy
Make semantic search service based on reviews
## Setup instructions for data preparation
Review text data is (obviously too large to save locally - in order to download and prepare reviews, first ensure that you have the folders `./models` and `./data` set up (these are ignored by git).
- download the pre-trained model for language detection from facebook from [here](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin)
- run the `load_reviews.py` script to download and preprocess the reviews (saved to to data folder as parquet files
