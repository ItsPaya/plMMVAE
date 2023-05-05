#!/usr/bin/env bash

#SBATCH --job-name=mmnist
#SBATCH --output=outputs/mmnist.output_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=07:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1

set -eo pipefail
shopt -s nullglob globstar

# define TMPDIR, if it's empty
if [[ -z "$TMPDIR" ]]; then
    TMPDIR="/tmp"
fi
echo "TMPDIR: $TMPDIR"

# activate conda env
eval "$(conda shell.bash hook)"
conda activate gpu_env
echo "CONDA_PREFIX: $CONDA_PREFIX"

# METHOD="joint_elbo"  # NOTE: valid options are "joint_elbo", "poe", and "moe"
# LIKELIHOOD_M1="laplace"
# LIKELIHOOD_M2="laplace"
# LIKELIHOOD_M3="categorical"
DIR_DATA="$PWD/data"
DIR_CLF="$PWD/trained_classifiers/trained_clfs_mst"
# DIR_EXPERIMENT="$PWD/runs/MNIST_SVHN_Text/${METHOD}/non_factorized/${LIKELIHOOD_M1}_${LIKELIHOOD_M2}_${LIKELIHOOD_M3}"
PATH_INC_V3="$PWD/pt_inception-2015-12-05-6726825d.pth"
DIR_FID="$TMPDIR/MNIST_SVHN_text"

# copy data to $TMPDIR
cp -r "${DIR_DATA}/MNIST" "${TMPDIR}/"
cp -r "${DIR_DATA}/SVHN" "${TMPDIR}/"
cp -r "${DIR_DATA}/MNIST_SVHN" "${TMPDIR}/"

python setConfig.py --config='./configs/mmnist.yaml' --data_path=$TMPDIR --clf="$DIR_CLF" --inception_state_dict="$PATH_INC_V3" --fid=$DIR_FID

python MasterTrainer.py --config='./configs/mmnist.yaml'
