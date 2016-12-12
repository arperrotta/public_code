
# coding: utf-8

# In[1]:

#This code was last modified on 121216
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import pandas as pd





# In[2]:

def select_samples_via_metadata(my_selection, metadata):
    '''
    This function was written by Isaac Rockafellow
    FUNCTION: Takes a metadata dataframe with columns as categories
                and rows as samples and outputs the sample names
                that meet your defined criteria
    INPUT: my_selection: a dictionary of key-value pairs from metadata
            'metdata: a pandas dataframe containing all metadata
                Ex: {'Tissue': 'Kidney', 'Prognosis': 'Bad'}
    OUTPUT: A list of the samples that meet the selected criteria
                Ex: ['patient 1', patient 12', 'patient 3'] all
                had bad kidneys
    '''
    selected_metadata = metadata # Not sure if this assignment is necessary
    for key, val in my_selection.items():
        selected_metadata = selected_metadata[ (selected_metadata[key]
            == val)]

    selected_samples = selected_metadata.index
    return list(selected_samples)


# In[2]:

def makeFracs(data):
    '''This function normailizes count data to generate relative fractions,
    has been tested by looking at the count of OTU/sumRead of a sample
    and making sure it is correct
    data = otu table of read counts with columns= OTUs and index= samples 
    THIS NEEDS TO BE RUN BEFORE ANY OTUS ARE REMOVED WITH dropOTU_SampOccur()'''
    #make fractions 
    data['sumRead']=data.sum(axis=1)
    fracs = data.loc[:,:].div(data["sumRead"], axis=0)
    fracs=fracs.drop('sumRead',axis='columns')
    return(fracs)

def filtSamps_lowCount(data,num):
    '''This function will filter out samples with low read counts
    data = otu table with columns = OTUs and index = samples 
    num = count cut off you want your samples to be above to be included'''
    data['sumRead']=data.sum(axis=1)
    dataF = data[data['sumRead'] >= num]
    dataF=dataF.drop('sumRead',axis='columns')
    samp=list(data.index)
    sampF=list(dataF.index)
    #look at which samples were removed for low read count
    filtOut=[]
    for i in samp:
        if i not in sampF:
            filtOut.append(i)
    print(len(filtOut),' samples out of ',len(samp),'were removed')
    return(dataF)

def filtMeta4filtData(dataF,meta):
    '''This function will filter your meta data DF so it only has samples in your filtered data DF
    dataF = otu table filtered for low read count samples, columns = OTUs and index = samples 
    meta = meta data DF'''
    sampF=list(dataF.index.values)
    metaF=meta.loc[sampF]
    return(metaF)


# In[1]:


def dropOTU_SampOccur(dataF,num):
    '''This function filters out OTUs that have non-zero abundance in at leats num samples
    dataF = OTU table (columns= OTUs and index= samples) that 
    has already been filtered for samples of low read count
    num = number of samples you require an OTU to be > 0 in to keep it in the table'''
    #drop OTUs that are present in less than 3 subjects 
    ## first make all zeros into NANs
    sampF=list(dataF.index.values)
    dNaN=dataF.replace(0, NaN)
    dNaN=dNaN.T
    #count coccurances of NaN
    x=dNaN.count(axis=1)
    dNaN['Occur']=x
    d=dNaN.replace(NaN,0)
    dM3=d[d['Occur'] >= num]
    dM3=dM3.drop('Occur',axis='columns')
    return(dM3.T)





