#This code contains functions for:
## plotting linear regressions of OTU abundance vs continuous meta data
## plotting boxplots of OTU abundance vs discrete meta data 
## you will also need a dataframe of RDP taxonomies generated using getRDPname_filtbymeta.py
## You will also need q-value dataframes generated using getQvals.py 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm
import scipy.stats as st
import statsmodels.sandbox.stats.multicomp as stest
from decimal import Decimal

#make lists of positive and negatiove correlations --> how many pos? or neg?
def linReg_plotIntOTUs(dataF,metaF,otuList,xaxis,xlabel,plotName,pVals,rdp):
    '''purpose: plot spearman regressions of otus against continuous data
    inputs: dataF = dataframe of OTU abundances sorted by metadata
    metaF= dataframe of metadata filtered for the samples in the DF
    otuList = a txt file of interesting otus that you want to plot
    xaxis = column of continuous data in metadatafile (string)
    xlabel = label you want on the x axis (string)
    plotName = name you want the output file to be saved under (string)'''
    with open(otuList,'r') as infile:
        feats=infile.read().replace("\"","").split()
    for i in feats:
        otu=i
        B=pd.DataFrame(data={xaxis:metaF[xaxis],'Abundance':dataF[i]})
        x=B[xaxis]
        y=B['Abundance']
        tName=get_phylo_from_RDP(rdp,i, 0.5)
        otus=list(pVals.index)
        if i in otus:
            p=pVals.loc[i,'Pval']
            q=pVals.loc[i,'Qval']
            t=st.pearsonr(x,y)
            tS=t[0].item()
            if p <= 0.05 and np.absolute(tS) > 0.3:
                sns_plot=sns.lmplot(xaxis,"Abundance",data=B, y_jitter=True)
                #add a jitter so you can see 
                sns.plt.xlabel(xlabel)
                sns.plt.ylabel(i+' Relative Abundance (log)')
                sns.plt.ylim(-20,6)
                sns.plt.xlim(0.5,4.5)
                plt.tight_layout(pad=3)
                plt.title(tName+'\n'+'R: %s, p-val: %s, q-val: %s' %
                  (('%.2E' % Decimal(tS)),('%.2E' % Decimal(p)),('%.2E' % Decimal(q))))
                sns.plt.savefig(outP+otu+"_"+plotName+'_linRegLogY.pdf',format='pdf')
                sns.plt.close()

def linRegKt_plotIntOTUs(dataF,metaF,otuList,xaxis,xlabel,plotName,pVals, rdp):
    '''purpose: plot spearman regressions of otus against continuous data
    inputs with kendall Tau coefficients: 
    dataF = dataframe of OTU abundances sorted by metadata
    metaF= dataframe of metadata filtered for the samples in the DF
    otuList = a txt file of interesting otus that you want to plot
    xaxis = column of continuous data in metadatafile (string)
    xlabel = label you want on the x axis (string)
    plotName = name you want the output file to be saved under (string)'''
    with open(otuList,'r') as infile:
        feats=infile.read().replace("\"","").split()
    for i in feats:
        otu=i
        B=pd.DataFrame(data={xaxis:metaF[xaxis],'Abundance':dataF[i]})
        x=B[xaxis]
        y=B['Abundance']
        tName=get_phylo_from_RDP(rdp,i, 0.5)
        otus=list(pVals.index)
        if i in otus:
            p=pVals.loc[i,'Pval']
            q=pVals.loc[i,'Qval']
            tS=pVals.loc[i,'kTau']
            if p <= 0.05 and np.absolute(tS) >= 0.3:
                sns_plot=sns.lmplot(xaxis,"Abundance",data=B, y_jitter=True)
                #add a jitter so you can see 
                sns.plt.xlabel(xlabel)
                sns.plt.ylabel(i+' Relative Abundance (log)')
                sns.plt.ylim(-20,6)
                sns.plt.xlim(0.5,4.5)
                plt.tight_layout(pad=3)
                plt.title(tName+'\n'+'Tau: %s, p-val: %s, q-val: %s' %
                  (('%.2E' % Decimal(tS)),('%.2E' % Decimal(p)),('%.2E' % Decimal(q))))
                sns.plt.savefig(outP+otu+"_"+plotName+'_linRegKT.pdf',format='pdf')
                sns.plt.close()

def otu_boxPlot(data1,data2,otuList,title1, title2,pVals,plotLabel, rdp):
    '''Purpose: make box plots of interesting OTUs and compare them between two categories, 
    Input: 
    data1 = dataframe of the abundance of an OTU across smaples that have been metadata sliced (High grading);
    data2 = dataframe of the abundance of an OTU across smaples that have been metadata sliced (Low grading)
    title1 = STRING of the label for data1 (high)
    title2 = STRING of the label for data2 (low); 
    plotLabel = STRING of the identifier you would like in the saved plot
    rdp = taxonomy dataframe generated using '''
    with open(otuList,'r') as infile:
        feats=infile.read().replace("\"","").split()
    for i in feats:
        tName=get_phylo_from_RDP(rdp,i, 0.5)
        C=pd.DataFrame(data={title1:data1[i],title2:data2[i]})
        otus=list(pVals.index)
        if i in otus:
            p=pVals.loc[i,'Pval']
            q=pVals.loc[i,'Qval']
            if p <=0.05:
                cx=sns.boxplot(data=C,saturation=0.5)
                cx=sns.stripplot(data=C, size=2, jitter=True, color="r")
                sns.plt.ylabel('Relative Abudnace (Log)')
                sns.plt.ylim(-14,0)
                plt.title(tName+'\n'+'p-Value: %s, q-Value: %s' % (('%.2E' % Decimal(p)),('%.2E' % Decimal(q))))
                sns.plt.savefig(outP+i+'_'+plotLabel+'_BoxPlt.pdf',format='pdf')
                sns.plt.close()
    



