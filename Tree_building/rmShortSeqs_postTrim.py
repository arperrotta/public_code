#This script will take a reference fasta that has been trimmed and filter out any hat are too short 
## leng = minimum length that ref sequence must be


import sys
IN = sys.argv[1]
leng=sys.argv[2]
OUT = sys.argv[3]
#open file
fileIN = open(IN,'r')
#make empty array
aRR = []
#make new file
fileOUT=open(OUT,'w')
for line in fileIN: 
    aRR.append(line)
for item in aRR:
    if ">" in item:
    	a=aRR.index(item)
    	if len(aRR[a+1]) >= leng:
    		fileOUT.write(item + aRR[a+1])
    else:
        continue
fileIN.close()
fileOUT.close()

