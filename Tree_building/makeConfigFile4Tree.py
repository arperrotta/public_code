#this script takes a text file that has denovoOTU ID and corresponding RDP taxonomy (k__Kingdom;p__Phyla; ect.)
## and creates a tab delim file with is the OTUID and corresponding highest RDP classification, format is similar to a fasta file
##NOTE: This should only be used if the tree is for visualization purposes and not if true phylogenetic inferences will be made from it
###In the later inference case avoid using a config file, get your self an out group and root your tree instead!
import sys
IN = sys.argv[1]
OUT = sys.argv[2]
#open file
fileIN = open(IN,'r')
#make empty array
aRR = []
for line in fileIN: 
    x=line.strip('\n').split('\t')
    aRR.append(x)
tax=[]
id=[]
for i in aRR:
	y=i[1].split(";")
	tax.append(y)
	id.append(i[0])
#now have a list with the OTUids and a different one with the taxonomies
##want to find last and longest in each item
##all good above, below wind up missing some IDs... 
#Now have a list of IDs and a list of lists of taxonomies
## Make lists of different levels
king=[]
phyla=[]
class1=[]
order=[]
family=[]
genera=[]
#the length requirement is to make up for empty taxa levels 
##this might be an issue though as now the binary string for each one will be different lengths 
for a in tax:
	if a[0] not in king and a[1]=='p__':
		king.append(a[0])
	if a[1] not in phyla and len(a[1])>3:
		phyla.append(a[1])
	if a[2] not in class1 and len(a[2])>3:
		class1.append(a[2])
	if a[3] not in order and len(a[3])>3:
		order.append(a[3])
	if a[4] not in family and len(a[4])>3:
		family.append(a[4])
	if a[5] not in genera and len(a[5])>3:
		genera.append(a[5])
#now need to make a list of each binary identifier	
fileOut=open(OUT,'w')
test=[]
posNum=[]
for x in aRR:
	kL=[]
	pL=[]
	cL=[]
	oL=[]
	fL=[]
	gL=[]
	pos=aRR.index(x)
	test.append(id[pos])
	posNum.append(pos)
	for t in king:
		if t in x[1]:
			k='1'
		kL.append(k)
	for b in phyla:
		if b in x[1]:
			p="1"
		if b not in x[1]:
			p="0"
		pL.append(p)
	for z in class1:
		if z in x[1]:
			c="1"
		if z not in x[1]:
			c="0"
		cL.append(c)
	for d in order:
		if d in x[1]:
			o="1"
		if d not in x[1]:
			o="0"
		oL.append(o)
	for e in family:
		if e in x[1]:
			f="1"
		if e not in x[1]:
			f="0"
		fL.append(f)
	for f in genera:
		if f in x[1]:
			g="1"
		if f not in x[1]:
			g="0"
		gL.append(g)
	tL=[kL,pL,cL,oL,fL,gL]
	tBins=[]
	for w in tL:
		bins=''.join(w)
		tBins.append(bins)
	fileOut.write('>'+id[pos]+'\n'+''.join(tBins)+'\n')

fileIN.close()
fileOut.close()