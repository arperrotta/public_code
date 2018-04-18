#!/usr/bin/env bash

#This script was written by APerrotta on 03/01/18
## The purpose of this script is to take a file with a list of OTUs that you want to grep the sequences of from a reference FASTA

#A file with the list of each ASV/OTU that you want the sequence of
##NOTE: each ASV/OTU must be on a different line and each line must start with >
FILE=$1

#The FASTA file you wish to reference
DB=$2

#The name of the output FASTA
OUT=$3

#Create a tmp file of the database that you want to reference and merge the fasta onto one line for easy grepping
python sangerFasta_merge_lines.py -i $DB -o ./refDB_tmp.fa

#Now grep it 
grep -A 1 -wFf $FILE refDB_tmp.fa >> ./outSeq_tmp.fa

#Grep adds "--" when it skips lines in the ref file
sed '/--/d' outSeq_tmp.fa > ./outSeq_tmp2.fa 

#Fix the line breaks so they are in 80 character break format (common FASTA format)
python fastaSplitTo80chr.py -i ./outSeq_tmp2.fa -o $OUT

#remove the empty line at the top of out 
sed -i '/^$/d' $OUT

#Remove the tmp files
rm ./refDB_tmp.fa
rm ./outSeq_tmp.fa
rm ./outSeq_tmp2.fa
