
import configparser
import os
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utility import filter_functions
from utility.language_functions import clean_stop_words
from utility.language_functions import find_best_words
from utility.language_functions import threshold_descriptions
from utility.language_functions import top_desciptions

stopwords = nltk.corpus.stopwords.words('italian')
print(len(stopwords))
newStopWords = ['sff','ae','ipp','dc','st','ml','ae','mx','tp','ipp','imz','bs','agp','st','dm','ae','nr','tft','dn','rr','mp','ra','bm','gi','nd','rr','rc','pm','mz']
stopwords.extend(newStopWords)
print(len(stopwords))
import datetime
start = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')



# read input files, create configuration instance and make a copy of df
conf = configparser.ConfigParser()
main_path = os.getcwd()
base_directory = str(Path(__file__).parent)
conf = configparser.ConfigParser()
conf.read(base_directory+'/utility/configuration.ini')
df  = pd.read_csv(base_directory+conf.get("PATH","ticket"),error_bad_lines=False,sep =";")
df.columns = map(str.lower,df.columns)
# filter data only for self channel client
self_channel_filter = filter_functions.self_channel_filter(df,balance=False)

# filter data only for telepthone channel
telephone_channel_filter = filter_functions.telephone_channel_filter(df)

#concat two dataframes
df = pd.concat([self_channel_filter,telephone_channel_filter])
df = df[df['tipo']!='R'].reset_index()
old_df = df.copy()

df = df[['numero',conf.get("PARAMETERS","column_name")]]
df = df.loc[0:conf.getint("PARAMETERS","desc_rows"),:]
print("df shape before filter for lenght of descrpition = " +str(df.shape[0]))

df = df[(df[conf.get("PARAMETERS","column_name")].str.len() > conf.getint("PARAMETERS","min_len_desc"))]
print("df shape after filter for lenght of descrpition = " +str(df.shape[0]))


#df = df[df[len(df['description'])>=10]]
# clean descriptions from italian stopwords
df = clean_stop_words(df=df, column=conf.get("PARAMETERS","column_name"), lang = "italian",stem=True)

# clean descriptions
df[conf.get("PARAMETERS","column_name")] = df[conf.get("PARAMETERS","column_name")].fillna('')

# create tfidf model instance
tfidf = TfidfVectorizer(stop_words='english')

# apply tfidf model to description column and create tf idf matrix
tfidf_matrix = tfidf.fit_transform(df[conf.get("PARAMETERS","column_name")])

# swamp key value of vocabulary name-index
word_indexes = tfidf.get_feature_names()
dict_vocab = tfidf.vocabulary_
swap_vocab = {v:k for k,v in dict_vocab.items()}


# calculate cosine similarity for the embedded vectors of the job positions
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# put diagonal values default = -1 to prevent counting them in clusters afterwards
np.fill_diagonal(cosine_sim,-1)

cosine_sim_pd = pd.DataFrame(cosine_sim,columns=df['numero'],index=df['numero'])

# find the most 5 representative words for each job position and save it into csv file
final_dict_list, data_frame_id_words = find_best_words(df=df,column_index_name='numero',matrix=tfidf_matrix,conf=conf,start = start,word_dict=swap_vocab,n=conf.getint("PARAMETERS","num_parole_rilevanti"),filename="id_words")

# creates a list for each description and the index of the best 5 descriptions
description_index_list = top_desciptions(cosine_sim)

# loops all the description and gets indexes of all the descriptions that are within a threshold of similarity.
threshhold_list,df_threshold = threshold_descriptions(df=df,matrix=cosine_sim,data_frame_id_words=data_frame_id_words,conf=conf,start=start,threshold=0.5,filename="threshold_poste_cluster_descriptions",drop_duplicates=True)


# plt.scatter(cosine_sim[y_kmeans == 0, 0], cosine_sim[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
# plt.scatter(cosine_sim[y_kmeans == 1, 0], cosine_sim[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
# plt.scatter(cosine_sim[y_kmeans == 2, 0], cosine_sim[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
# plt.title('Semantic Clusters')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show()
print(1)
print(1)

# drop duplicates from column
#indices = pd.Series(df.index, index=df['title']).drop_duplicates()
