import configparser
import os
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns


def intern_channel_filter(df,bilancio=False):
    """
    filter dataframe with major filters, first entity, first group, autore to retrieve only intern channel
    :param df:df to filter
    :param bilancio: if true filter pbarea2 for squadratura di bilancio
    :return:filtered df
    """
    df.columns = map(str.lower, df.columns)
    df = df[(df['firstentity'] == 'Polo Tecnologico') | (df['firstentity'] == 'SERVICESUPPORT')]
    df = df [(df['firstgroup'] == 'SERVICE DESK_TD') | (df['firstgroup'] == 'ServiceDesk_CO')|(df['firstgroup'] == 'SERVICESUPPORT')]
    df = df[df['autore']=='AMSD Automation']
    if bilancio:
        df[df['pbarea2'] == 'SQUADRATURE_DI_BILANCIO ']
    return df



base_directory = str(Path(__file__).parent)
conf = configparser.ConfigParser()
conf.read(base_directory+'/utility/configuration.ini')
df  = pd.read_csv(base_directory+conf.get("PATH","ticket"),error_bad_lines=False,sep =";")
df = intern_channel_filter(df=df,bilancio=False)



df['opendate'] = pd.to_datetime(df['opendate'])
df['resolvedate'] = pd.to_datetime(df['resolvedate'])
df['resolvedate'] = df['resolvedate'].bfill()
if len(conf.get("TIME","filter")):
    df = df[df['resolvedate']<conf.get("TIME","filter")]
df.set_index('resolvedate',inplace=True)

# attenzione il tempo viene calcolato dal tempo +freq al tempo quindi, ad esempio, se ci sta la riga ... 00:00:00 e 1 ora di frequenza raggruppa da 00 00 a 01:00

df_new = df.resample(conf.get("TIME","resample_time"))

# attenzione il tempo viene calcolato dal tempo -freq al tempo quindi, ad esempio, se ci sta la riga ... 01:30:00 e 1 ora di frequenza raggruppa da 00 30 a 01:30
group_1h = df.groupby([pd.Grouper(freq='12H', base=00, label='right'),df.pbarea2]).size().reset_index(name='count_pbarea2')

variable_to_plot = group_1h[group_1h[conf.get('PLOT','column')]==conf.get('PLOT','filter')]

# ax = plt.subplot(111)
# ax.bar(variable_to_plot['resolvedate'],variable_to_plot['count_pbarea2'], width=10)
# ax.xaxis_date()

plt.plot_date(variable_to_plot['resolvedate'],variable_to_plot['count_pbarea2'],kind='bar')
plt.xlabel('interval with week freq')
plt.ylabel('pbarea2 ATM')
plt.show()

# df['timesolve'] = df['resolvedate']-df['opendate']
#
# df['timesolve'] = df['timesolve'].dt.total_seconds()
# df['timesolve'] = pd.to_numeric(df['timesolve'],errors='coerce')


def intern_channel_filter(df,bilancio=False):
    """
    filter dataframe with major filters, first entity, first group, autore to retrieve only intern channel
    :param df:df to filter
    :param bilancio: if true filter pbarea2 for squadratura di bilancio
    :return:filtered df
    """

    df = df[(df['firstentity'] == 'Polo Tecnologico') | (df['firstentity'] == 'SERVICESUPPORT')]
    df = df [(df['firstgroup'] == 'SERVICE DESK_TD') | (df['firstgroup'] == 'ServiceDesk_CO')|(df['firstgroup'] == 'SERVICESUPPORT')]
    df = df[df['autore']=='AMSD Automation']
    if bilancio:
        df[df['pbarea2'] == 'SQUADRATURE_DI_BILANCIO ']
    return df




print(1)