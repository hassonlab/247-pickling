#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --mem=192GB
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH -o './logs/%A.out'
#SBATCH -e './logs/%A.err'
#SBATCH --mail-type=fail
#SBATCH --mail-user=hvgazula@umich.edu

if [[ "$HOSTNAME" == *"tiger"* ]]
then
    echo "It's tiger"
    module load anaconda
    source activate torch-env
else
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
