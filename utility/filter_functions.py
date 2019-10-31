import pandas as pd


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

def cleanDataWithSigma(X, sigma=1E-02):
    """
    removes columns with sigma less than sigma
    :param X:Matrix to clean
    :param sigma: sigma threshold to use to exclude Data
    :return: cleaned Matrix
    """
    import numpy
    listIndex = []
    for k in range(len(X[1])):
        if numpy.std((X[:, k])) < sigma:
            listIndex.append(k)
    X = numpy.delete(X, listIndex, axis=1)
    return X




def filterCombinations(list,operator="=",index_of_comb=0, value=0):
    """
    filters combinations from list with condition
    :param list: list to filter
    :param operator: equal or different
    :param index_of_comb: index of element of combinations to check
    :param value: value to compare each element of list
    :return:list filtered
    """
    if operator=="=":
        list = [x for x in list if x[index_of_comb] == value]
    if operator=="!=":
        list = [x for x in list if x[index_of_comb] != value]
    if operator==">":
        list = [x for x in list if x[index_of_comb] > value]
    if operator=="<":
        list = [x for x in list if x[index_of_comb] < value]
    return list

    return best_candidates,winner,dist_candidates_list,winner_distance,time_distance_edges_from_first,index_min_list,interval_appl




def checkCompatibility(value_to_check, value_threshold, looseness = 0.3):
    """
    checks if value is similar to value threshold with tolerance = looseness
    :param value_to_check:
    :param value_threshold:
    :param looseness:
    :return: Boolean
    """
    import numpy as np
    if np.abs(value_to_check-value_threshold)<np.abs(looseness*value_threshold):
        return True
    else:
        return False

def readListFromConfFile(list, type=str):
    list = list.split(",")
    list = map(type, list)
    return list

def normalizeData(array, positionIndex="median"):
    """
    normalizes data subtracting position index from array
    :param array: array to normalize
    :param positionIndex: position index to use
    :return: array normalized
    """

    import numpy as np
    if positionIndex == "median":
        array = array - np.median(array)

    elif positionIndex == "mean":
        array = array - np.mean(array)

    else:
        positionIndex = "median"
        return normalizeData(array=array,positionIndex=positionIndex)

    return array

def checkCentroidDimension(centroid):
    """
    checks shape of centroid to get only first dimension
    :param centroid: centroid to check
    :return: first dimension of centroid
    """
    import numpy as np

    if centroid.ndim ==1:
        centroid = np.squeeze(centroid)
    elif centroid.ndim > 1:
        centroid = centroid[:,0]
        centroid = np.squeeze(centroid)
    else:
        print ("dimension of centroid does not make sense")
        pass
    return centroid

def hot_encode(df,cols):
    df[cols] =df[cols].apply(lambda x: pd.factorize(x)[0]+1)
    df = pd.get_dummies(df,columns=cols)
    return df
