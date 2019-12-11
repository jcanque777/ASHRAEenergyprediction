import pandas as pd
import numpy as np

def convert_btu(df,building_df, energy_col='meter_reading', site_col='site_id',has_col=False):
    '''
    df.groupby(by='site_id')['meter_reading'].mean() -mean meter reading is 549 without converion, 414 with conversion
    converts site 0, meter_type 0 from BTU into kWh
    mean meter reading is 549 without converion, 414 with conversion
    df:dataset with the meter_reading col
    building_df: dataset with the site_id col
    energy_col: meter_reading column name
    site_col: site column name
    has_col: if True, does not append site_col column to main df
    '''
    
    if has_col:
        df.loc[(df['site_id'] == 0) & (df['meter'] == 0), energy_col] = df[(df['site_id'] == 0) & (df['meter'] == 0)][energy_col] * 0.2931
        return df
    else:
        df=df.join(building_df[site_col],how='left',on='building_id')
        df.loc[(df['site_id'] == 0) & (df['meter'] == 0), energy_col] = df[(df['site_id'] == 0) & (df['meter'] == 0)]['meter_reading'] * 0.2931
        return df

def convert_back(df,building_df, energy_col='meter_reading', site_col='site_id',has_col=False):
    '''
    converts site 0, meter_type 0 from kWh into BTU
    df:dataset with the meter_reading col
    building_df: dataset with the site_id col
    energy_col: meter_reading column name
    site_col: site column name
    has_col: if True, does not append site_col column to main df
    '''
    
    if has_col:
        df.loc[(df['site_id'] == 0) & (df['meter'] == 0), energy_col] = df[(df['site_id'] == 0) & (df['meter'] == 0)][energy_col] * 3.4118
        return df
    else:
        df=df.join(building_df[site_col],how='left',on='building_id')
        df.loc[(df['site_id'] == 0) & (df['meter'] == 0), energy_col] = df[(df['site_id'] == 0) & (df['meter'] == 0)]['meter_reading'] * 3.4118
        return df
