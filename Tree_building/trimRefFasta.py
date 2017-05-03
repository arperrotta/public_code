#This script will take a reference fasta database and trim it for a given region
## begTrim = the nt position you want your reference to start at
## endTrim = the nt position you want your reference to end at 
## trim positions are determined by mapping you query to the reference and seeing the most common alignment positions

import sys
IN = sys.argv[1]
begTrim=sys.argv[2]
endTrim=sys.argv[3]
OUT = sys.argv[4]
#open file
fileIN = open(IN,'r')
#make empty array
aRR = []
aRR2=[]
#make new file
fileOUT=open(OUT,'w')
for line in fileIN: 
    aRR.append(line)
for item in aRR:
    if ">" in item:
        aRR2.append(item)

    else:
        x=item[:int(endTrim)+1]
        y=x[int(begTrim):]
        aRR2.append(y)

for item in aRR2:
	if ">" in item:
		a=aRR2.index(item)
		if len(aRR2[a+1]) > 0:
			fileOUT.write(item)
			fileOUT.write(aRR2[a+1]+'\n')
		else:
			continue
		
fileIN.close()
fileOUT.close()

