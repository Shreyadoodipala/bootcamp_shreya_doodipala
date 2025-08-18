# Data Storage
+ `.env` file is used to store the paths to the data (raw and processed)
+ `data/raw`: stores CSV files
+ `data/processed`: stores parquet files
+ Validation is done to ensure that:
    - the dataframe sizes are the same
    - the columns' datatype is the same