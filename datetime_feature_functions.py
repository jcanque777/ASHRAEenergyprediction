import pandas as pd
import requests
import json
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import calendar
from datetime import date
import holidays

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