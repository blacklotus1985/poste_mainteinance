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
df['resolvedate'] = df['resolvedate'].bfill()
df.set_index('resolvedate',inplace=True)

# attenzione il tempo viene calcolato dal tempo +freq al tempo quindi, ad esempio, se ci sta la riga ... 00:00:00 e 1 ora di frequenza raggruppa da 00 00 a 01:00

df_new = df.resample(conf.get("TIME","resample_time"))

# attenzione il tempo viene calcolato dal tempo -freq al tempo quindi, ad esempio, se ci sta la riga ... 01:30:00 e 1 ora di frequenza raggruppa da 00 30 a 01:30
group_1h = df.groupby([pd.Grouper(freq='60Min', base=30, label='right'),df.pbarea1]).size().reset_index(name='count_pbarea1')

df['timesolve'] = df['resolvedate']-df['opendate']

df['timesolve'] = df['timesolve'].dt.total_seconds()
df['timesolve'] = pd.to_numeric(df['timesolve'],errors='coerce')




print(1)