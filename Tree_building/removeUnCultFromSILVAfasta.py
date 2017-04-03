#This script will take a reference fasta database and trim it fo a given length
import sys
IN = sys.argv[1]
OUT = sys.argv[2]
#open file
fileIN = open(IN,'r')
#make empty array
aRR = []
#make new file
fileOUT=open(OUT,'w')
for line in fileIN: 
    aRR.append(line)
for item in aRR:
    if (">" in item and "uncultured" not in item):
        fileOUT.write(item)
        x=aRR.index(item)
        fileOUT.write(aRR[x+1])
    else:
        continue
fileIN.close()
fileOUT.close()

