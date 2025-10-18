#!/bin/bash
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH --ntasks-per-node 32
#SBATCH --cpus-per-task 1
#SBATCH --mem=128G 
#SBATCH --mail-user=frieldskatherine@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err

free -h
ulimit -a

cd ..

podman-hpc run  --rm -it --volume="/pscratch/sd/k/kfrields/climsim-online-data/inputdata:/storage/inputdata" \
    --volume "/pscratch/sd/k/kfrields/climsim-online-data/shared_e3sm:/storage/shared_e3sm" \
    --volume "/pscratch/sd/k/kfrields/climsim-online-data/scratch:/scratch" \
    --volume "/dev/shm:/dev/shm" \
    climsim:podman example_job_submit_nnwrapper_v4_constrained.py
