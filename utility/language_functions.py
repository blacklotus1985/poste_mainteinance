from nltk.corpus import stopwords
from nltk.stem.snowball import ItalianStemmer
import re
import os
import pandas as pd
import numpy as np
from utility.filter_functions import n_top_items


def clean_stop_words(df,column,lang,stem=True):
    """
    (df,str,str) -> df
    cleans column of dataframe from stopwords with the given language
    :param df: dataframe to clean
    :param column: column of dataframe to clean
    :param lang: language of stopwords
    :param stem: if stemming activated
    :return: cleaned dataframe
    """
    for i in range(df.shape[0]):
        try:
            df.loc[i,column]=re.sub('[^a-zA-Z]+',' ', df[column][i])
        except:
            print ("problems with row =" + str(i))
            df.drop(labels=i,axis=0,errors= 'ignore',inplace=True)
            continue
    document = df[column].str.lower().str.split()
    sentence_stem = []
    document_stem = []

    nltk_stop = stopwords.words(lang)
    clean_document = document.apply(lambda x: [item for item in x if item not in nltk_stop])
    stemmer = ItalianStemmer()
    if stem:
        for sentence in clean_document:
            for word in sentence:
                word = stemmer.stem(word)
                sentence_stem.append(word)
            document_stem.append(sentence_stem)
            sentence_stem = []
        sentences = [' '.join(i) for i in document_stem]
        cleaned_series = pd.Series((v for v in sentences))
        df[column] = cleaned_series
    else:
        sentences = [' '.join(i) for i in clean_document]
        cleaned_series = pd.Series((v for v in sentences))
        df[column] = cleaned_series
    return df

def find_best_words(df,column_index_name,matrix,word_dict,n,conf,filename="default",save=True):
    """
    retrieves most significant words from tfidf matrix
    :param df: df to analyze
    :param column_index_name: column of index to be used
    :param matrix: sparse matrix
    :param word_dict: dictionary of index and words of tfidf
    :param n: number of words retrieved
    :return: dataframe with job id -  top words
    """

    list=[]
    final_terms=[]
    final_dict_list=[]
    for i in range(matrix.shape[0]):
        arr = matrix[i].toarray()[0]
        top_items = n_top_items(arr,n_items=n,max=True)
        list.append(top_items)
        for elem in list[0]:
            final_terms.append(word_dict[elem])
            try:
                dict={"numero":df[column_index_name][i], "top_words":final_terms}
                final_dict_list.append(dict)
            except:
                print ("cannot create dictionary top words of index = "+ str(i))
        list=[]
        final_terms=[]
    data_frame_id_words = pd.DataFrame.from_records(final_dict_list,coerce_float=True)
    data_frame_id_words.drop_duplicates(subset='numero',inplace=True)
    if save:
        data_frame_id_words.to_csv(os.getcwd() + conf.get("OUTPUT", "folder_path") + conf.get("OUTPUT", "id_words"), sep=";", index=False)
    return final_dict_list, data_frame_id_words


def top_desciptions(matrix, n=5):
    """
    gets the top similarities for each description
    :param matrix: matrix of similarity
    :param n: number of top similarities
    :return: list of similarities
    """
    description_index_list =[]
    for i in range(len(matrix[0])):
        array_res = n_top_items(matrix[i], n_items=n, max=True)
        description_index_list.append(array_res)
    return description_index_list

def threshold_descriptions(df,matrix, conf, threshold=0.5,filename="default",save=True):
    """
    gets all the similarities for each description that are bigger of a certain threshold
    :param matrix: matrix of similarties
    :param threshold: fixed threshold
    :return: data frame of dictionary list of Job ID - similarityID - Similarity value
    """
    threshhold_list=[]
    for i in range(len(matrix[0])):
        cosine_desc = matrix[i]
        dict = {"job_ID":df["ID"][i],"similar_ID":df["ID"][np.where(matrix[i]>threshold)[0]].values, "similarity_value":np.asarray(cosine_desc[np.where(cosine_desc>threshold)[0]])}
        threshhold_list.append(dict)
    df_threshold = pd.DataFrame.from_records(threshhold_list,coerce_float=True)

    if save:
        wd = os.getcwd()
        os.chdir(wd)
        df_threshold.to_csv(os.path.dirname(os.getcwd())+conf.get("OUTPUT_FILES","folder")+filename,sep=";", index = False)
    return threshhold_list,df_threshold