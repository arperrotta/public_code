#This script will take a fasta file with split line sequences and merge them into one line
#The first line of the output file will be empty though so you must remove it before further processing
import sys
IN = sys.argv[1]
OUT = sys.argv[2]

#open file
fileIN = open(IN,'r')
#make empty array
aRR = []
#make new file
fileOut=open(OUT,'w')
for line in fileIN: 
    aRR.append(line)
for item in aRR:
    if ">" in item:
        fileOut.write('\n' + item)
    else:
        fileOut.write(item.strip())
fileIN.close()
fileOut.close()
    
