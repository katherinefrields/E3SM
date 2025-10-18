#!/bin/bash
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH --ntasks-per-node 32
#SBATCH --cpus-per-task 1
#SBATCH --mem=128G 
#SBATCH --mail-user=frieldskatherine@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err

free -h
ulimit -a

#I moved this file so you might have to cd to the top in order to run it correctly
cd ..

podman-hpc run  --rm -it --volume="/pscratch/sd/k/kfrields/climsim-online-data/inputdata:/storage/inputdata" \
    --volume "/pscratch/sd/k/kfrields/climsim-online-data/shared_e3sm:/storage/shared_e3sm" \
    --volume "/pscratch/sd/k/kfrields/climsim-online-data/scratch:/scratch" \
    --volume "/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models:/hugging" \
    --volume "/dev/shm:/dev/shm" \
    climsim:podman E3SM/climsim_scripts/example_job_submit_nnwrapper_v2_old.py

