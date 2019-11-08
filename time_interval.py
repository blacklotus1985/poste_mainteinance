import configparser
import os
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from utility import filter_functions
from utility import time_functions





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
df = time_functions.calculate_time_difference(df, column1='resolvedate', column2='opendate', new_col_name='time_solved_ticket', total_seconds=True)


# filters df based on time of column given
time_functions.time_filter_rows(df, column_name='resolvedate', conf=conf, bigger=True, set_index=True)

df['opendate'] = pd.to_datetime(df['opendate'])
df['opendate'] = pd.to_datetime(df['opendate'])
df['opendate'] = df['opendate'].bfill()
# if len(conf.get("TIME","filter")):
#     df = df[df['resolvedate']<conf.get("TIME","filter")]

# calculate day of the week of opendate, 0 is Monday 6 is Sunday
weeday_df = df['opendate'].dt.weekday
weeday_df.replace(to_replace=[0,1,2,3,4,5,6],value=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],inplace=True)

#weeday_df.rename(index={0: "Monday", 1: "Tuesday", 2: "Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"},inplace=True)


# list of keys to group
keylist = ['pbarea1','pbarea2','pbarea3','opendate_y']


df_opendate = df.merge(weeday_df, left_index=True, right_index=True)

df_opendate['opendate_y'].rename(index={0: "Monday", 1: "Tuesday", 2: "Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"},inplace=True)


# groups of pbarea and weekday
groups = df_opendate.groupby(keylist).size()

#dataframe of groups

groups = groups.to_frame()


# area frazionario gropuby
key_area_frazionario = ['pbarea1','frazionario']
area_frazionario = df.groupby(key_area_frazionario).size()

#key all pbarea
keypbarea = ['pbarea1','pbarea2','pbarea3']
pbarea_all = df.groupby(keypbarea).size()




df.set_index('opendate',inplace=True)

# attenzione il tempo viene calcolato dal tempo +freq al tempo quindi, ad esempio, se ci sta la riga ... 00:00:00 e 1 ora di frequenza raggruppa da 00 00 a 01:00


# attenzione il tempo viene calcolato dal tempo -freq al tempo quindi, ad esempio, se ci sta la riga ... 01:30:00 e 1 ora di frequenza raggruppa da 00 30 a 01:30
min15 = df.groupby(pd.Grouper(freq='24H')).size()

# to have week day groups, to be edited
min15_new = min15.reset_index()
week = min15_new['opendate'].dt.weekday
frame_fin = min15_new.merge(week, left_index=True, right_index=True)

frame_fin = frame_fin.rename(columns={"opendate_y": "weekday", 0:"n_tickets", "opendate_x": "ticket_interval_time"})

#risolv = df.groupby('regione').timesolve.mean()
group_week = frame_fin.groupby('weekday').n_tickets.sum()
group_week_frame = group_week.to_frame()

#save to csv in output folder
folder_path = os.getcwd()
min15.to_csv(folder_path + conf.get("OUTPUT","folder_path")+conf.get("OUTPUT","filename_time_interval"),header=['n° tickets_open'],sep=';')
group_week_frame.rename(index={0: "Monday", 1: "Tuesday", 2: "Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"},inplace=True)
group_week_frame.to_csv(folder_path + conf.get("OUTPUT","folder_path")+conf.get("OUTPUT","filename_weekday_ticket_count"),sep=";")
groups.to_csv(folder_path + conf.get("OUTPUT","folder_path")+conf.get("OUTPUT","groups_and_pbarea_all"),sep=";")
print(1)
pbarea = False

if pbarea:
    pbarea1_group = df.groupby('pbarea1').size()
    pbarea1_group = pbarea1_group.to_frame()
    pbarea1_group.to_csv(folder_path + conf.get("OUTPUT", "folder_path") + conf.get("OUTPUT", "pbarea1_general"), sep=";")
    pbarea2_group = df.groupby('pbarea2').size()
    pbarea2_group = pbarea2_group.to_frame()
    pbarea2_group.to_csv(folder_path + conf.get("OUTPUT", "folder_path") + conf.get("OUTPUT", "pbarea2_general"),sep=";")

    area_frazionario.to_csv(folder_path + conf.get("OUTPUT", "folder_path") + conf.get("OUTPUT", "area_frazionario"),sep=";")
    area_frazionario.to_csv(folder_path + conf.get("OUTPUT", "folder_path") + conf.get("OUTPUT", "area_frazionario"),sep=";")
    pbarea_all.to_csv(folder_path + conf.get("OUTPUT", "folder_path") + conf.get("OUTPUT", "pbarea_all"),sep=";")

    print(1)



print(1)
print(2)
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