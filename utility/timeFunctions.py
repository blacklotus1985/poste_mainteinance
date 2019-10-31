import pandas as pd
def getActualDate(printTime=False, exclude_ms=True):
    """
    get actual time
    :param printTime: if true prints the time
    :param exclude_ms: if true excludes microseconds
    :return: actual time
    """
    import datetime
    now = datetime.datetime.now()
    if exclude_ms:
        now = now.replace(microsecond=0)
    if printTime:
        print ("actual time is " + str(now))
    return now

def getTime():
    """
    Return the current time in seconds since the Epoch.
    Fractions of a second may be present if the system clock provides them
    :return: time
    """
    from time import time
    now = time()
    return now

def seconds_to_time(sec, show = False):
    sec = int(sec)
    d = sec // (24 * 60 * 60)
    sec %= (24 * 60 * 60)
    h = sec // (60 * 60)
    sec %= (60 * 60)
    m = sec // 60
    s = sec % 60
    dict = {"days":d, "hours":h, "minutes":m, "seconds":s}
    if show:
        print('{:n} days, {:n} hours, {:n} minutes, {:n} seconds'.format(d, h, m, s))

    return dict



def calculate_time_difference(df,column1,column2, total_seconds = False,new_col_name='default'):
    """
    get pandas dataframe calculate difference in time between columns 1 and column 2 and transform column into total seconds
    :param df: df to analyze
    :param column1: column name 1
    :param column2: column name 2
    :param new_col_name: name of new column
    :param total_seconds: if true return column in total seconds
    :return: dataframe with new columns
    """

    df[column1] = pd.to_datetime(df[column1])
    df[column2] = pd.to_datetime(df[column2])
    df[new_col_name] = df[column1] - df[column2]
    if total_seconds:
        df[new_col_name] = df[new_col_name].dt.total_seconds()
        df[new_col_name] = pd.to_numeric(df[new_col_name], errors='coerce')
    return df



def time_filter_rows(df,column_name,conf,bigger=False,set_index=False):
    """
    check in config file if time has value and filter dataframe with that filter
    :param df: df to filter
    :param conf: config file to get value of time
    :param column_name: name of column to filter
    :param bigger: if True bigger than, if False smaller than
    :param set_index: if True set index to column name
    :return: filtered df
    """
    if len(conf.get("TIME", "filter")):
        if bigger:
            df = df[df[column_name] > conf.get("TIME", "filter")]
        else:
            df = df[df[column_name] < conf.get("TIME", "filter")]
    if set_index:
        df.set_index(column_name, inplace=True)
    return df
