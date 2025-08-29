# Preprocessing
## Train-test split
- The data was split into 80% training data and 20% testing data, with stratification to ensure that both splits have an equal proportion of loans defaulted and not defaulted.

## Encoding categorical variables
- Frequency encoding was used to convert the `purpose` into numeric values.
- No further encoding was required for `credit_policy` and `default` as they're already numeric.

## Scaling numeric variables
