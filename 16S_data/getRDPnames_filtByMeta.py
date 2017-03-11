#This script contains functions for:
## loading a table of RDP taxonomies
## getting the highest resolution taxonomy leve form and RDP for OTUs
## filter DFs by metadata


import pandas as pd
import numpy as np
from numpy import NaN

#load up the RDP file that you will use for your taxonomies
def loadRDP(file):
	columns_names=['empty1','empty2','empty3','empty4','Kingdom','Kingdom1','Kingdom_score'
               	,'Phylum','Phylum1','Phylum_score','Class','Class1','Class_score','Order','Order1'
               	,'Order_score','Family','Family1','Family_score','Genera','Genera1','Genera_score']

	rdp=pd.read_table(file, sep='\t',index_col=0,names=columns_names)
	return(rdp)

def get_phylo_from_RDP(rdp, otu, cutoff):
    '''
    FUNTION - Grabs the most specific phylogenetic info that matches your cutoff
    INPUT - rdp: a pandas dataframe, otu: a string, and cutoff: a float
    OUTPUT - a string stating the phylogenetic assignment
    '''
    phylo_assignment = ''
    phylo_scoring = ['Kingdom_score', 'Phylum_score', 'Class_score', 'Order_score', 'Family_score','Genera_score']
    for phylo in phylo_scoring:
        rdp_score =  rdp.loc[otu,phylo]
        #print 'rdp_score', rdp_score
        if rdp_score >= cutoff:
            # Yay, you found a thing!
            phylo_name = phylo.replace('_score','')
            phylo_assignment = rdp.loc[otu].loc[phylo_name]
            x=phylo_name
            #print 'yaaaaay', rdp_score, phylo_assignment
    return(phylo_assignment)

def select_samples_via_metadata(my_selection, metadata):
    '''
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

