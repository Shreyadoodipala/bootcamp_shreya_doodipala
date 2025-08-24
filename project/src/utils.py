import pandas as pd

def change_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change column names of the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame whose columns need to be renamed.

    Returns:
    pd.DataFrame: A DataFrame with updated column names.
    """
    cols_mapping = {
        'credit.policy': 'credit_policy',
        'int.rate': 'interest_rate',
        'installment': 'installment',
        'log.annual.inc': 'log_annual_income',
        'dti': 'debt_income_ratio',
        'days.with.cr.line': 'days_with_credit_line',
        'revol.bal': 'revolve_balance',
        'revol.util': 'revolve_utilized',
        'inq.last.6mths': 'inquiries_last_6_mon',
        'delinq.2yrs': 'delinquent_2_yrs',
        'pub.rec': 'public_recs',
        'not.fully.paid': 'default'
    }
    return df.rename(columns=cols_mapping)