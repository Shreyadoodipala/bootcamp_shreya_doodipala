# Data Preprocessing
### Drop missing values
- The `drop_missing` function takes the pandas DataFrame as an input, and optionally columns and threshold.  
- If `columns` are specified, drop rows only if any of the given columns contain missing values. Otherwise, drop rows that have any column with a missing value.
- If `threshold` is specified, drop a fraction of rows based on how many non-missing values they contain.
- Default behavior: drops any row that has at least one missing value in any column  

+ The code applies this function for all columns using `threshold=0.5`. This removes rows with more that half of its values missing.
+ **Pros:** remove data points with too much missing information  
**Cons:** may remove information that is important to analysis

### Fill missing values with median
- The `fill_missing_median` function takes the pandas DataFrame as an input, and optionally columns.
- It fills the missing values of specified columns or all numerical columns of the dataframe with the median value.
+ **Pros:** not influenced by outliers  
**Cons:** doesn't take into account the relationship with other columns

### Normalize data
- The `normalize_data` function takes the pandas DataFrame as an input and optionally columns and method.
- If `columns` are specified, it only normalizes those columns, else it selects all numeric columns to normalize.
- If `method` isn't specified, the `MinMaxScaler` from `scikit-learn` is selected by default. If anything else is specified, it uses `scikit-learn`'s `StandardScaler`.
+ The code uses the default MinMaxScaler to scale all the numeric columns.