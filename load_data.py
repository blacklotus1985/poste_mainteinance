import configparser
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from utility import filter_functions
from utility import timeFunctions
from pathlib import Path


base_directory = str(Path(__file__).parent)
conf = configparser.ConfigParser()
main_path = os.getcwd()
conf.read(main_path+'/utility/configuration.ini')

df  = pd.read_csv(base_directory+conf.get("PATH","ticket"),error_bad_lines=False,sep =";")

# filter data only for self channel client
self_channel_filter = filter_functions.self_channel_filter(df,balance=False)

# filter data only for telepthone channel
telephone_channel_filter = filter_functions.telephone_channel_filter(df)

#concat two dataframes
df = pd.concat([self_channel_filter,telephone_channel_filter])

df.columns = map(str.lower, df.columns)

df['opendate'] = pd.to_datetime(df['opendate'])
df['resolvedate'] = pd.to_datetime(df['resolvedate'])
df['timesolve'] = df['resolvedate']-df['opendate']
df['timesolve'] = df['timesolve'].dt.total_seconds()/3600
df['timesolve'] = pd.to_numeric(df['timesolve'],errors='coerce')

#test comment
key_pbarea = ['pbarea1','pbarea2','pbarea3']
save = True

df['regione'] = df['regione'].str.lower()
risolv = df.groupby(key_pbarea).timesolve.mean()
if save:
    folder_path = os.getcwd()
    risolv.to_csv(folder_path + conf.get("OUTPUT", "folder_path") + conf.get("OUTPUT", "pbarea_all_mean_time"),sep=";")

risolv = risolv.drop('4')
risolv = risolv.drop('5')
risolv = risolv.drop('9')
risolv = risolv.drop('20')
risolv = risolv.drop('rm')
risolv = risolv.rename(index={"friuli venezia giulia":"friuli"})
risolv = risolv.rename(index={"trentino alto adige":"trentino"})
risolv = risolv.rename(index={"emilia romagna":"emilia"})
ft = risolv/sum(risolv)
plt.pie(risolv,labels=risolv.index,autopct='%.2f')
plt.title("Percentage mean time per region")
plt.show()
sns.heatmap(risolv.values.reshape(1,-1))
plt.show()
# plt.scatter(risolv.index, risolv)
# plt.xlabel("Regions")
# plt.ylabel("Time")
# plt.title("Regions mean time ticket solving")

#plt.show()

print(1)