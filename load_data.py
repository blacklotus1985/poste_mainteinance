import configparser
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

conf = configparser.ConfigParser()
main_path = os.getcwd()
conf.read(main_path+'/utility/configuration.ini')
df  = pd.read_csv(main_path+conf.get("PATH","ticket"),error_bad_lines=False,sep =";")

df_ten = df.iloc[0:10,:]

gp = df.groupby('Area')
df.columns = map(str.lower, df.columns)

df['opendate'] = pd.to_datetime(df['opendate'])
df['resolvedate'] = pd.to_datetime(df['resolvedate'])
df['timesolve'] = df['resolvedate']-df['opendate']
df['timesolve'] = df['timesolve'].dt.total_seconds()
df['timesolve'] = pd.to_numeric(df['timesolve'],errors='coerce')

#test comment
df['regione'] = df['regione'].str.lower()
risolv = df.groupby('regione').timesolve.mean()
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