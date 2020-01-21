import pandas as pd
import requests
import json
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from datetime import date
import holidays

train = pd.read_csv('/Users/kylebaranko/ashrae-energy-prediction/train.csv')
building = pd.read_csv('/Users/kylebaranko/ashrae-energy-prediction/building_metadata.csv')
weather_data=pd.read_csv('/Users/kylebaranko/ashrae-energy-prediction/weather_train.csv')


# Site 0 Meter Conversion

def convert_btu(df, building_df, energy_col='meter_reading', site_col='site_id', has_col=False):
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
        df.loc[(df['site_id'] == 0) & (df['meter'] == 0), energy_col] = df[(df['site_id'] == 0) & (df['meter'] == 0)][
                                                                            energy_col] * 0.2931
        return df
    else:
        df = df.join(building_df[site_col], how='left', on='building_id')
        df.loc[(df['site_id'] == 0) & (df['meter'] == 0), energy_col] = df[(df['site_id'] == 0) & (df['meter'] == 0)][
                                                                            'meter_reading'] * 0.2931
        return df


def convert_back(df, building_df, energy_col='meter_reading', site_col='site_id', has_col=False):
    '''
    converts site 0, meter_type 0 from kWh into BTU
    df:dataset with the meter_reading col
    building_df: dataset with the site_id col
    energy_col: meter_reading column name
    site_col: site column name
    has_col: if True, does not append site_col column to main df
    '''

    if has_col:
        df.loc[(df['site_id'] == 0) & (df['meter'] == 0), energy_col] = df[(df['site_id'] == 0) & (df['meter'] == 0)][
                                                                            energy_col] * 3.4118
        return df
    else:
        df = df.join(building_df[site_col], how='left', on='building_id')
        df.loc[(df['site_id'] == 0) & (df['meter'] == 0), energy_col] = df[(df['site_id'] == 0) & (df['meter'] == 0)][
                                                                            'meter_reading'] * 3.4118
        return df

df_converted = convert_btu(train, building)


# Holiday and Date Time Conversions

def is_holiday(day):
    if day in us_holidays:
        return True
    else:
        return False


us_holidays = holidays.UnitedStates()


def holiday(timestamp):
    timestamp = pd.Timestamp(timestamp)
    holi = is_holiday(timestamp)
    return holi


def create_datetime_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M')
    df['month'] = pd.DatetimeIndex(df['timestamp']).month
    df['hour'] = pd.DatetimeIndex(df['timestamp']).hour
    df['weekday'] = pd.DatetimeIndex(df['timestamp']).weekday
    df['holiday'] = np.vectorize(holiday)(df['timestamp'])
    return df

df_converted_dates = create_datetime_features(df_converted)


def train_meter_columns(train_meter_copy, weather_data, dummies=True, time_index=False):
    train_meter = train_meter_copy.copy(deep=True)
    wet_train = weather_data.copy(deep=True)

    # Convert Timestamp column to date time index
    train_meter['timestamp'] = pd.to_datetime(train_meter['timestamp'])
    train_meter.set_index('timestamp', inplace=True)

    # {0: electricity, 1: chilledwater, 2: steam, 3: hotwater}
    # Takes value in meter_reading column based on the meter column value and places it in the correct column
    if dummies == True:
        train_meter = pd.concat([train_meter, pd.get_dummies(train_meter['meter'])], axis=1)

        wet_train['timestamp'] = pd.to_datetime(wet_train['timestamp'])
        df = pd.merge(train_meter,
                      wet_train[['air_temperature', 'dew_temperature', 'wind_speed', 'site_id', 'timestamp']],
                      left_on=['site_id', 'timestamp'],
                      right_on=['site_id', 'timestamp'],
                      how='left')
        for col in ['air_temperature', 'dew_temperature', 'wind_speed']:
            df[col] = df[col].interpolate()


    else:
        train_meter['0_electricity'] = np.where(train_meter['meter'] == 0, train_meter['meter_reading'], 0)
        train_meter['1_chilledwater'] = np.where(train_meter['meter'] == 1, train_meter['meter_reading'], 0)
        train_meter['2_steam'] = np.where(train_meter['meter'] == 2, train_meter['meter_reading'], 0)
        train_meter['3_hotwater'] = np.where(train_meter['meter'] == 3, train_meter['meter_reading'], 0)

        wet_train['timestamp'] = pd.to_datetime(wet_train['timestamp'])
        df = pd.merge(train_meter,
                      wet_train[['air_temperature', 'dew_temperature', 'wind_speed', 'site_id', 'timestamp']],
                      left_on=['site_id', 'timestamp'],
                      right_on=['site_id', 'timestamp'],
                      how='left')
        for col in ['air_temperature', 'dew_temperature', 'wind_speed']:
            df[col] = df[col].interpolate()

    if time_index == True:
        # Convert Timestamp column to date time index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    return df

dummies = train_meter_columns(df_converted_dates, weather_data, time_index=True)
dummies.head()

meter_values = train_meter_columns(df_converted_dates, weather_data, dummies=False)
meter_values.head()