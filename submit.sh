#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
##SBATCH --cpus-per-task=4
#SBATCH -o 'logs/%A.log'
 
if [[ "$HOSTNAME" == *"tiger"* ]]
then
    echo "It's tiger"
    module load anaconda
    conda activate 247-main
elif [[ "$HOSTNAME" == *"della"* ]]
then
    echo "It's della-gpu"
    module purge
    module load anaconda3/2021.11
    conda activate 247-main
else
    module purge
    module load anacondapy
    source activate srm
fi

echo 'Requester:' $USER
echo 'Node:' $HOSTNAME
echo 'Start time:' `date`
echo "$@"
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    python "$@" --conversation-id $SLURM_ARRAY_TASK_ID
else
    python "$@"
fi
echo 'End time:' `date`
