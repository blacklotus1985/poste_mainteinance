
from utility.language_functions import clean_stop_words
from utility.language_functions import find_best_words
from utility.language_functions import top_desciptions
from utility.language_functions import threshold_descriptions

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import configparser
from nltk.corpus import stopwords
from nltk.stem.snowball import ItalianStemmer
import numpy as np
import os
import re

# read input files, create configuration instance and make a copy of df
conf = configparser.ConfigParser()
main_path = os.getcwd()
path = os.path.dirname(os.getcwd())
conf.read(os.path.dirname(os.getcwd())+'/configurations/configurations.ini')
filename = conf.get("INPUT_FILES",conf.get("INPUT_FILES","input"))
df = pd.read_csv(os.path.dirname(os.getcwd())+filename,delimiter=';' )
old_df = df.copy()

# clean descriptions from italian stopwords
df = clean_stop_words(df=df, column="description", lang = "italian",stem=True)

# clean descriptions
df['description'] = df['description'].fillna('')

# create tfidf model instance
tfidf = TfidfVectorizer(stop_words='english')

# apply tfidf model to description column and create tf idf matrix
tfidf_matrix = tfidf.fit_transform(df['description'])

# swamp key value of vocabulary name-index
word_indexes = tfidf.get_feature_names()
dict_vocab = tfidf.vocabulary_
swap_vocab = {v:k for k,v in dict_vocab.items()}


# calculate cosine similarity for the embedded vectors of the job positions
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# find the most 5 representative words for each job position and save it into csv file
final_dict_list, data_frame_id_words = find_best_words(df=df,matrix=tfidf_matrix,conf=conf,word_dict=swap_vocab,n=5,filename="id_words.csv")

# creates a list for each description and the index of the best 5 descriptions
description_index_list = top_desciptions(cosine_sim)

# loops all the description and gets indexes of all the descriptions that are within a threshold of similarity.
threshhold_list,df_threshold = threshold_descriptions(df=df,matrix=cosine_sim,conf=conf,threshold=0.3,filename="threshold_descriptions.csv")


# drop duplicates from column
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
