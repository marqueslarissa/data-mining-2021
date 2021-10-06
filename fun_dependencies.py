import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import janitor

def load_data(filepath, sep):
    df = pd.read_csv(filepath, sep, low_memory=False).clean_names()
    return df

def missing_treatment(df):
    rows_before = df.shape[0]
    before = pd.concat([df.isna().sum(), df.isna().sum()/len(df)*100], axis=1)
    print(f'Before Missing Values\n{before}')
    
    #drop data is missing - cols where there are at least the 30% of not null values        
    df.dropna(thresh=0.3*len(df), axis=1, inplace=True)
    
    #Numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    #print(numeric_columns)
    
    if numeric_columns:
        # fill the NaN values of numeric columns with the average value
        df[numeric_columns] = df[numeric_columns].fillna(df.mean())
    
    #Categorial columns
    categorial_columns = df.select_dtypes(include=np.object).columns.tolist()
    #print(categorial_columns)
    
    if categorial_columns:
        #drop categorical data is missing - rows where there are null values
        df.dropna(subset=categorial_columns, inplace=True)
        
        # object to string
        df[categorial_columns] = df[categorial_columns].astype('string')

        # to lower
        df.columns = map(str.lower, df.columns)

    rows_after = df.shape[0]
    after = pd.concat([df.isna().sum(), df.isna().sum()/len(df)*100], axis=1)
    print(f'After Missing Values\n{after}')
    print("\nPercent missing value removed: {:.2%}\n".format((rows_before-rows_after)/rows_before))
    return df
    