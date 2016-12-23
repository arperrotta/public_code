#This script contains functions for generating q-values form a statistical test
## function include those for: 
### abundance filtering OTUs
### kendall test qValues
### getting the correlations form the kendall (filtered to be >=|3| in value)
### mann-whitney test qValues
### filtering qValues for a significance level 
### filtering qValues for specific OTUs and a significance level   

import scipy.stats as st
import statsmodels.sandbox.stats.multicomp as stest
import numpy as np
import pandas as pd

def dropOTU_lowAbund(data, maxread, cutoff):
	''' 
	This function wil remove OTUs that are below an abundance cut off from your OTU table
	data = OTU table (columns= OTUs and index= samples) for all sampes
	maxread = maximum read count from the counts table
	cutoff = abundance cut off (fraction)'''
	z = np.log(cutoff+(1/float(2))*(1/float(maxread)))
	x=data.T
	x['avFrac']=x.mean(axis='columns')
	y=x[x['avFrac'] >= z]
	y=y.drop('avFrac',axis='columns')
	return(y.T)


## when generating q-val lists also grap correlation so know if good or bad? think on
def qVal_kendall(otus, data, metaCol):
	'''otus = list of ALL of the OTUs in your abundance filtered DF
	data = dataframe of samples (rows) vs OTUs (columns)
	metaCol = meta data DF and column of the continuous data you want to correlate with your OTU data'''
	pVal=[]
	kTau=[]
	for i in otus:
		a=data[i].tolist()
		b=metaCol.tolist()
		try:
			p1=st.kendalltau(a,b)[1]
			pVal.append(p1)
			k=st.kendalltau(a,b)[0]
			kTau.append(k)
		except ValueError:
			p1=1
			pVal.append(p1)
			k=0
			kTau.append(k)
	qVal=stest.multipletests(pVal,alpha=0.05,method='fdr_bh')
	B=pd.DataFrame(data={'Qval':qVal[1],'Pval':pVal, 'kTau':kTau},index=otus)
	return(B)

def corr_kendall(otus, data, metaCol, out):
	'''otus = list of ALL of the OTUs in your abundance filtered DF
	data = dataframe of samples (rows) vs OTUs (columns)
	metaCol = meta data DF and column of the continuous data you want to correlate with your OTU data'''
	''''out = a string, where do you want the text files to go'''
	outPos = open(out+'posKendallList.txt', 'w')
	outNeg = open(out+'negKendallList.txt', 'w')
	corrPos=[]
	kPos=[]
	corrNeg=[]
	kNeg=[]
	for i in otus:
		a=data[i].tolist()
		b=metaCol.tolist()
		try:
			k=st.kendalltau(a,b)[0]
		except ValueError:
			k=0
		if k >= 0.3:
				corrPos.append(i)
				kPos.append(k)
		if k <= -0.3:
				corrNeg.append(i)
				kNeg.append(k)
	B=pd.DataFrame(data={'kTau_Pos':kPos},index=corrPos)
	C=pd.DataFrame(data={'kTau_Neg':kNeg},index=corrNeg)
	D=B.append(C)
	D.to_csv(out+'kendallCorr.txt',na_rep='NaN',sep='\t',header=True)
	for i in corrPos:
		outPos.write(i+'\n')
	for e in corrNeg:
		outNeg.write(e+'\n')
	outPos.close()
	outNeg.close()
	return(D)
	
def qVal_mannwhit(otus, data1, data2):
	'''otus = list of ALL of the OTUs in your abundance filtered DF
	data1 = dataframe of samples (rows) vs OTUs (columns) of one class
	data2 = dataframe of samples (rows) vs OTUs (columns) of another class'''
	pVal=[]
	for i in otus:
		a=data1[i].tolist()
		b=data2[i].tolist()
		try:
			p1=st.mannwhitneyu(a,b)[1]
			pVal.append(p1)
		except ValueError:
			p1=1
			pVal.append(p1)
	qVal=stest.multipletests(pVal,alpha=0.05,method='fdr_bh')
	B=pd.DataFrame(data={'Qval':qVal[1],'Pval':pVal},index=otus)
	return(B)
	
def getOTUs_QvalCutOff(qVal, cutOff):
	y=qVal[qVal['Qval'] <= cutOff]
	return(y)
	
def getSpecOTUs_QvalCutOff(qVal, otus, cutOff):
	x=[]
	for i in otus:
		try:
			if qVal.loc[i,'Qval'] <= cutOff:
				x.append(i)
		except KeyError:
			continue
	y=qVal.loc[x]
	return(y)
	
	