def combine_tables(df1, df2):
    
    import pandas as pd

    common_columns = list(set(df1.columns) & set(df2.columns)) 
    df_combined = pd.merge(df1, df2, how='left', on=common_columns)
    return df_combined