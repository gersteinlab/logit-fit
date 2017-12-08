#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -l mem=15GB
#PBS -m abe -M lucas.lochovsky@yale.edu
#PBS -N glm_nb_qsub
#PBS -r n
#PBS -j oe
#PBS -q gerstein
#PBS -V		

cd /net/gerstein/ll426/code/nb-fit
time ./glm.nb test/y.txt test/x.txt 0.000001 > test/out.txt
