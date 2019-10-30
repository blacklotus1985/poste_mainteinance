import configparser
import os
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns

base_directory = str(Path(__file__).parent)
conf = configparser.ConfigParser()
conf.read(base_directory+'/utility/configuration.ini')
df  = pd.read_csv(base_directory+conf.get("PATH","ticket"),error_bad_lines=False,sep =";")

df_ten = df.iloc[0:10,:]

gp = df.groupby('Area')
df.columns = map(str.lower, df.columns)

df['opendate'] = pd.to_datetime(df['opendate'])
df['resolvedate'] = pd.to_datetime(df['resolvedate'])
df['resolvedate'] = df['resolvedate'].ffill().bfill()
df.set_index('resolvedate',inplace=True)
df.resample('4H').sum()
df['timesolve'] = df['resolvedate']-df['opendate']

df['timesolve'] = df['timesolve'].dt.total_seconds()
df['timesolve'] = pd.to_numeric(df['timesolve'],errors='coerce')


print(1)