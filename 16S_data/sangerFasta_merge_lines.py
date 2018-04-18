#!/usr/bin/env python
#Written by APerrotta on 4/18/18

import sys
import argparse

def main():
	# Usage statement
	parseStr = 'This script takes a fasta file with split line sequences and merges them into one line'
	parser = argparse.ArgumentParser(description=parseStr)
	parser.add_argument('-i','--inputfile',help='Input FASTA with sequence linebreaks',required=True,type=argparse.FileType('r'))
	parser.add_argument('-o','--outputfile',help='Output FASTA without sequence linebreaks',required=True,type=argparse.FileType('w'))
	args = parser.parse_args()
	
	#open file
	fileIN = args.inputfile
	#make empty array
	aRR = []
	#make new file
	fileOut=args.outputfile
	for line in fileIN: 
    	aRR.append(line)
	for item in aRR:
    	if ">" in item:
        	fileOut.write('\n' + item)
    	else:
        	fileOut.write(item.strip())
	fileIN.close()
	fileOut.close()
	
if __name__ == '__main__':       
    main() 
    
