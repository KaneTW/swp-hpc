#!/usr/bin/env zsh
 
### Job name
#BSUB -J SERENAJOB
 
### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
#BSUB -o SERENAJOB.%J.%I
 
### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute,
### that means for 80 minutes you could also use this: 1:20
#BSUB -W 10

#BSUB -P lect0005
 
### Request memory you need for your job in TOTAL in MB
#BSUB -M 8192
 
### Request the number of compute slots you want to use
#BSUB -n 64
 
### Use esub for OpenMP/shared memeory jobs
#BSUB -a openmp
 
### Change to the work directory
cd /home/dk406646/swp-hpc/src
 
### Execute your application

CG_MAX_ITER=6000 OMP_PLACES=cores OMP_PROC_BIND=spread  ./cg.exe /home/lect0005/matrix/Serena.mtx

