#This script will take a reference fasta database and trim it fo a given length

import sys
#This script was made on 03/26/17 
##Used to take a fasta downloaded form SILVA and get the genus and species names of each sequence 
## so you can use those names on an ITOL tree
IN = sys.argv[1]
begTrim=sys.argv[2]
endTrim=sys.argv[3]
OUT = sys.argv[4]
#open file
fileIN = open(IN,'r')
#make empty array
aRR = []
#make new file
fileOUT=open(OUT,'w')
for line in fileIN: 
    aRR.append(line)
for item in aRR:
    if (">" in item and 'denovo18' not in item and 'unidentified' not in item):
        x=item.split(' Bacteria')
        y=x[0].split('>')
        z=x[1].split(';')
        fileOUT.write(y[1]+'\t')
        fileOUT.write(z[len(z)-1])
    if (">" in item and 'unidentified' in item):
    	x=item.split(' Bacteria')
        y=x[0].split('>')
        z=x[1].split(';')
        fileOUT.write(y[1]+'\t')
        fileOUT.write(z[len(z)-2]+'__'+y[1]+'\n')
    if '>denovo18' in item:
    	y=item.split('>')
    	fileOUT.write(y[1]+'\t')
    	fileOUT.write('Predictive Anaerococcus OTU'+'\n')
    else:
    	continue
		
fileIN.close()
fileOUT.close()

