import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import janitor
import datetime as dt
import re


def load_data(filepath, sep):
    df = pd.read_csv(filepath, sep, low_memory=False).clean_names()
    
    for col in df.columns:
        if col in "service_comp_wbs_aff_":
            df = df.rename(columns = {'service_comp_wbs_aff_':'service_component_wbs_aff_'})
    
    return df




def numeric(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    #print(numeric_columns)
    
    return numeric_columns





def categorial(df):
    categorial_columns = df.select_dtypes(exclude=np.number).columns.tolist()
    #print(categorial_columns)
    
    return categorial_columns



def missing_treatment(df, drop, fill):
    rows_before = df.shape[0]
    before = pd.concat([df.isna().sum(), df.isna().sum()/len(df)*100], axis=1)
    print(f'Before Missing Values\n{before}')
    
    if drop:
        #drop data is missing - cols where there are at least the 30% of not null values        
        df.dropna(thresh=0.3*len(df), axis=1, inplace=True)
    
    elif fill:
        #Numeric columns
        numeric_columns = numeric(df)

        if numeric_columns:
            # fill the NaN values of numeric columns with the average value
            df[numeric_columns] = df[numeric_columns].fillna(df.mean())

        #Categorial columns
        categorial_columns = categorial(df)

        if categorial_columns:
            #drop categorical data is missing - rows where there are null values
            df.dropna(subset=categorial_columns, inplace=True)
            
    else:
        #drop data is missing - cols where there are at least the 30% of not null values        
        df.dropna(thresh=0.3*len(df), axis=1, inplace=True)
        
        #Numeric columns
        numeric_columns = numeric(df)

        if numeric_columns:
            # fill the NaN values of numeric columns with the average value
            df[numeric_columns] = df[numeric_columns].fillna(df.mean())

        #Categorial columns
        categorial_columns = categorial(df)

        if categorial_columns:
            #drop categorical data is missing - rows where there are null values
            df.dropna(subset=categorial_columns, inplace=True)

    rows_after = df.shape[0]
    after = pd.concat([df.isna().sum(), df.isna().sum()/len(df)*100], axis=1)
    print(f'After Missing Values\n{after}')
    print('\nPercent missing value removed: {:.2%}\n'.format((rows_before-rows_after)/rows_before))
    
    return df





def data_formatting(df):
    
    #Numeric columns
    numeric_columns = numeric(df)
    
    if numeric_columns:
        # 2 decimals points
        df[numeric_columns] = df[numeric_columns].apply(lambda x: x.round(decimals=2), axis=1).astype('int64')
        
        # Round down
        #df[numeric_columns] = df[numeric_columns].apply(np.floor, axis=1)
        
        
        for col in df[numeric_columns].columns:
            print(f"Numeric Column ' {col} ' Unique Values:\n {df[col].unique()}\n")
    
    #Categorial columns
    categorial_columns = categorial(df)
    
    if categorial_columns:
        # object to string
        df[categorial_columns] = df[categorial_columns].astype('string')
        
        # to lower
        df[categorial_columns] = df[categorial_columns].apply(lambda x: x.str.lower(), axis=1)
        
        # remove white space at beginning and end string
        df[categorial_columns] = df[categorial_columns].apply(lambda x: x.str.lstrip(), axis=1)
        df[categorial_columns] = df[categorial_columns].apply(lambda x: x.str.rstrip(), axis=1)
        
        date_columns = df.filter(regex='time|date|planned_start|planned_end|actual_start|actual_end').columns.tolist()
        except_date_columns =  ["handle_time_hours_","handle_time_secs_"]
        
        
        format1 ="%d/%m/%Y %H:%M:%S"
        format2 = "%Y/%m/%d %H:%M:%S"
        
        for col in date_columns:
            if col not in except_date_columns: #dados que nao são datetime e sim int
                #print(f"col:{col} ")
                df[col] = pd.to_datetime(df[col], infer_datetime_format=format1).dt.strftime(format2)    
                df[col] = df[col].astype('datetime64[s]')
                #data_frame = pd.DataFrame(columns=l_header, parse_dates=['date'], infer_datetime_format='%Y-%m-%d %H:%M:%S')      
     
        for col in df[categorial_columns].columns:
            # treatment of exception            
            if col == 'urgency':
                df[col] = df[col].astype(str).str.extract('(\d)').astype(float).astype('int64')
            if col in except_date_columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float).astype('int64')
                
            print(f"Categorial Column ' {col} ' Unique Values:\n {df[col].unique()}\n")
              
    return df