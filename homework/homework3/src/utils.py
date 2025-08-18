import pandas as pd

def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    df_grouped_category = df.groupby('category').mean(numeric_only=True).reset_index()
    return df_grouped_category