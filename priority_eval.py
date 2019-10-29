import configparser
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
conf = configparser.ConfigParser()
main_path = os.getcwd()
conf.read(main_path+'/utility/configuration.ini')
df  = pd.read_csv(main_path+conf.get("PATH","ticket"),error_bad_lines=False,sep =";")


df.columns = map(str.lower, df.columns)

df['opendate'] = pd.to_datetime(df['opendate'])
df['resolvedate'] = pd.to_datetime(df['resolvedate'])
df['timesolve'] = df['resolvedate']-df['opendate']
df['timesolve'] = df['timesolve'].dt.total_seconds()
df['timesolve'] = pd.to_numeric(df['timesolve'],errors='coerce')
#sum_pnl_bnp_curr = dettaglio_pns_s2[['id_trade', 'pnl_bnp_curr']].groupby(['id_trade']).sum().reset_index()

#test comment
df['priority'] = df['priority'].str.lower()
risolv = df.groupby('priority').timesolve.mean()

# objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
# y_pos = np.arange(len(objects))
# performance = [10,8,6,4,2,1]
#
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Usage')
# plt.title('Programming language usage')

y_pos = np.arange(len(risolv))
plt.bar(y_pos, risolv, align='center', alpha=0.5)
plt.xticks(y_pos, risolv.index)
plt.scatter(risolv.index, risolv)
plt.xlabel("Priority")
plt.ylabel("Resolution time in seconds")
plt.title("Mean  time ticket solving")

plt.show()

print(1)