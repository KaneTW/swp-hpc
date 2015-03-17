#!/usr/bin/env zsh
  
### Job name
#BSUB -J AMPLIFIER_LOWMEM
  
### File / path where STDOUT will be written, the %J is the job id
#BSUB -o AMPLIFIER_LOWMEM.%J
  
### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute,
### that means for 80 minutes you could also use this: 1:20
#BSUB -W 60

#BSUB -P lect0005
 
### Request a job with X11-Forwarding
#BSUB -XF
 
### Request memory you need for your job in MB
#BSUB -M 2048
  
### Request the number of compute slots you want to use
#BSUB -n 6
  
### Use esub for OpenMP/shared memory jobs
#BSUB -a openmp

#BSUB -R "select[model==Beckton]"
  
### Change to the work directory
cd $HOME
  
### load modules and execute
module load intelvtune
uname -a > $HOME/info
uptime >> $HOME/info
free >> $HOME/info  
amplxe-gui
