#!/bin/bash

#################################
# Author: Patrick Shober        #
# Date: 17/06/22                #
#################################
#          Description          #
#  Runs Pokorny '13 impact      #
#  probability code on csv of   #
#  a,e,i,w values.              #
#################################

# Collect inputs
read -p "Enter the csv location (ex. /home/patrick/Downloads/example.csv): " filename
read -p "Target semi-major axis (au): " sma
read -p "Target eccentricity: " ecc
read -p "Drive location: " drive

# create folder for all the output files
rmdir $drive/output_files_pokorny
mkdir $drive/output_files_pokorny

N=1

csvcut -c 2,3,4,5 $filename | while read line; do

# create input file
echo $line > IN
echo $sma >> IN
echo $ecc >> IN

# run code
./CODE < IN

# rename and mv file to storage folder
mv output.txt $drive/output_files_pokorny/output$N.txt

N=$(( $N + 1 ))

done


# # append output files and generate one csv file
# python pokorny_output_to_usable.py -f ./output_files_pokorny
