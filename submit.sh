#!/bin/bash
#SBATCH --time=02:10:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
##SBATCH --cpus-per-task=4
#SBATCH --open-mode=truncate
#SBATCH -o './logs/%x.out'
#SBATCH -e './logs/%x.err'
 
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

echo 'Requester:' $USER 'Node:' $HOSTNAME
echo "$@"
echo 'Start time:' `date`
start=$(date +%s)

if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    python "$@" --conversation-id $SLURM_ARRAY_TASK_ID
else
    python "$@"
fi

end=$(date +%s)
echo 'End time:' `date`
echo "Elapsed Time: $(($end-$start)) seconds"
