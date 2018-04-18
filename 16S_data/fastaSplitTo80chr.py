#!/usr/bin/env python              
## written by APerrotta on 02/28/18                                                                       
import sys
import argparse


def main():
	# Usage statement
	parseStr = 'This script takes a FASTA file and reformats it into 80 character line breaks (common FASTA format)'
	parser = argparse.ArgumentParser(description=parseStr)
	parser.add_argument('-i','--inputfile',help='Input FASTA without sequence linebreaks',required=True,type=argparse.FileType('r'))
	parser.add_argument('-o','--outputfile',help='Output FASTA with 80 character sequence linebreaks',required=True,type=argparse.FileType('w'))
	args = parser.parse_args()
	
#open file   
	#print(args.inputfile)
	fileIN = args.inputfile
#make empty array                                                                                         
	aRR = []

#make new file                                                                                            
	fileOut=args.outputfile

#Fill in lines of new file                                                                                
	for line in fileIN:
        	line = line.strip()
        	aRR.append(line)

    	
    	

	for item in aRR:
		i=79
		if ">" in item:
			fileOut.write('\n'+item)
		if ">" not in item:
			z=list(item)
			while i<len(z):
				z.insert(i,'\n')
				#w=''.join(z)
				i+=80
			w=''.join(z)
			fileOut.write('\n'+w)


	fileIN.close()
	fileOut.close()
if __name__ == '__main__':       
    main() 
