import configparser
import os
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from utility import filter_functions
from utility import timeFunctions





base_directory = str(Path(__file__).parent)
conf = configparser.ConfigParser()
conf.read(base_directory+'/utility/configuration.ini')
df  = pd.read_csv(base_directory+conf.get("PATH","ticket"),error_bad_lines=False,sep =";")

# filter data only for self channel client
self_channel_filter = filter_functions.self_channel_filter(df,balance=False)

# filter data only for telepthone channel
telephone_channel_filter = filter_functions.telephone_channel_filter(df)

#concat two dataframes
df = pd.concat([self_channel_filter,telephone_channel_filter])


# create a columns with resolution time of ticket
df = timeFunctions.calculate_time_difference(df,column1='resolvedate',column2='opendate',new_col_name='time_solved_ticket',total_seconds=True)


# filters df based on time of column given
timeFunctions.time_filter_rows(df,column_name='resolvedate',conf=conf,bigger=True,set_index=True)

df['opendate'] = pd.to_datetime(df['opendate'])
df['opendate'] = pd.to_datetime(df['opendate'])
df['opendate'] = df['opendate'].bfill()
# if len(conf.get("TIME","filter")):
#     df = df[df['resolvedate']<conf.get("TIME","filter")]
df.set_index('opendate',inplace=True)

# attenzione il tempo viene calcolato dal tempo +freq al tempo quindi, ad esempio, se ci sta la riga ... 00:00:00 e 1 ora di frequenza raggruppa da 00 00 a 01:00

#df_new = df.resample(conf.get("TIME","resample_time"))

# attenzione il tempo viene calcolato dal tempo -freq al tempo quindi, ad esempio, se ci sta la riga ... 01:30:00 e 1 ora di frequenza raggruppa da 00 30 a 01:30
min15 = df.groupby(pd.Grouper(freq='15Min')).size()

#save to csv in output folder
folder_path = os.getcwd()
min15.to_csv(folder_path + conf.get("OUTPUT","folder_path")+conf.get("OUTPUT","filename_time_interval"),header=['n° tickets_open'],sep=';')


'''
some small plot if necessary

#variable_to_plot = group_1h[group_1h[conf.get('PLOT','column')]==conf.get('PLOT','filter')]

# ax = plt.subplot(111)
# ax.bar(variable_to_plot['resolvedate'],variable_to_plot['count_pbarea2'], width=10)
# ax.xaxis_date()
#
# plt.plot_date(variable_to_plot['resolvedate'],variable_to_plot['count_pbarea2'],kind='bar')
# plt.xlabel('interval with week freq')
# plt.ylabel('pbarea2 ATM')
# plt.show()

# df['timesolve'] = df['resolvedate']-df['opendate']
#
# df['timesolve'] = df['timesolve'].dt.total_seconds()
# df['timesolve'] = pd.to_numeric(df['timesolve'],errors='coerce')

'''





print(1)