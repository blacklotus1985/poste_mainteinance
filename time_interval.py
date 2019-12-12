import configparser
import os
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from utility import filter_functions
from utility import time_functions

from itertools import product
import numpy as np
from scipy.stats import percentileofscore
from matplotlib import pyplot as plt




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
min15 = df.groupby(pd.Grouper(freq='15min')).size()

# to have week day groups, to be edited
min15_new = min15.reset_index()
week = min15_new['opendate'].dt.weekday
frame_fin = min15_new.merge(week, left_index=True, right_index=True)

frame_fin = frame_fin.rename(columns={"opendate_y": "weekday", 0:"n_tickets", "opendate_x": "ticket_interval_time"})

#risolv = df.groupby('regione').timesolve.mean()
group_week = frame_fin.groupby('weekday').n_tickets.sum()
group_week_frame = group_week.to_frame()

hour_min_list = []
year_month_day_list = []

pd_time = pd.to_datetime(frame_fin['ticket_interval_time'])

for elem in pd_time:
    hour_min_list.append(elem.strftime("%H:%M"))
    year_month_day_list.append(elem.strftime('%y-%m-%d'))


frame_fin['hour_min'] = hour_min_list
frame_fin['date'] = year_month_day_list
time_unique = frame_fin['hour_min'].unique()
week_day_unique = frame_fin['weekday'].unique()

def calc_percentile(df, weekday, value_to_compare, time_start='08:00', box_plot = False):
    """
    calculates percentile of value of number of tickets based on time and weekday
    :param df: dataframe with n_tickets, date,weekday and hour_min
    :param weekday: weekday to choose: 0 is Monday 6 is Sunday
    :param value_to_compare: value to look for percentile
    :param time_start: time start to analyze (usually beginning of 15 minute interval)
    :return: percentile of score 0-100 relative to value to compare
    """
    df = df[df['weekday']==weekday]
    df = df[df['hour_min']==time_start]
    #df.hour_min = df.hour_min.astype('datetime64[ns]')
    array_tickets = df['n_tickets'].values
    array_tickets = np.sort(array_tickets)
    result = percentileofscore(array_tickets,value_to_compare)
    if box_plot:
        plt.boxplot(array_tickets)
        plt.title("Boxplot of tickets")
        plt.show()
    return np.round(result,1)


def cumulative_tickets(df,weekday,time_end, value_to_compare=3, box_plot=False):
    """
    calculates percentile of value of number of cumulative tickets based on time and weekday
    :param df: dataframe with n_tickets, date,weekday and hour_min
    :param weekday: weekday to choose: 0 is Monday 6 is Sunday
    :param time_end: time end to analyze (usually end of 15 minute interval starting at midnight)
    :param value_to_compare: value to look for percentile
    :param box_plot: if true plots boxplot of cumulative tickets per day
    :return: percentile of score 0-100 relative to value to compare
    """
    df = df[df['weekday']==weekday]
    df = df[df['hour_min']<time_end]
    df_series= df.groupby(df.date).sum()
    array_tickets = df_series.n_tickets.values
    array_tickets = np.sort(array_tickets)
    result = percentileofscore(array_tickets, value_to_compare)
    if box_plot:
        plt.boxplot(array_tickets)
        plt.title("Boxplot of cumulative tickets")
        plt.show()
    return np.round(result, 1)


resulte_cum = cumulative_tickets(df = frame_fin, weekday=2,time_end='09:45',value_to_compare=50,box_plot=True)



result_single = calc_percentile(df = frame_fin, weekday=2,value_to_compare=3,time_start='09:45',box_plot=True)

prod = list(product(week_day_unique,time_unique))
df_week_time = pd.DataFrame()
dict_percentile_list = []
for elem in prod:
    time_week_single_df = frame_fin[(frame_fin['weekday']==elem[0])&(frame_fin['hour_min']==elem[1])]
    df_week_time = df_week_time.append(time_week_single_df, ignore_index=True)
    threshold_value = np.percentile(time_week_single_df["n_tickets"],conf.getint("PARAMETERS","percentile_week_time"))
    dict = {"weekday":time_week_single_df.iloc[1,2], "min_15": time_week_single_df.iloc[1,3], "percentile_considered":conf.getint("PARAMETERS","percentile_week_time"),"threshold_percentile_value":np.round(threshold_value,0)}
    dict_percentile_list.append(dict)
df_percentile = pd.DataFrame(dict_percentile_list)
df_percentile['weekday'].replace(to_replace=[0,1,2,3,4,5,6], value = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday","Sunday"],inplace=True)




result = calc_percentile(frame_fin,4,3)


#save to csv in output folder
folder_path = os.getcwd()
min15.to_csv(folder_path + conf.get("OUTPUT","folder_path")+conf.get("OUTPUT","filename_time_interval"),header=['n° tickets_open'],sep=';')
group_week_frame.rename(index={0: "Monday", 1: "Tuesday", 2: "Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"},inplace=True)
group_week_frame.to_csv(folder_path + conf.get("OUTPUT","folder_path")+conf.get("OUTPUT","filename_weekday_ticket_count"),sep=";")
groups.to_csv(folder_path + conf.get("OUTPUT","folder_path")+conf.get("OUTPUT","groups_and_pbarea_all"),sep=";")
pbarea = True

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
    df_percentile.to_csv(folder_path + conf.get("OUTPUT", "folder_path") + conf.get("OUTPUT", "percentile_week_time"),sep=";")










print(1)