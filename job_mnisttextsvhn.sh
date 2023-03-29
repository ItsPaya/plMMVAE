#!/usr/bin/env bash

#SBATCH --job-name=mnisttextsvhn
#SBATCH --output=mnisttextsvhn.output_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=07:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

set -eo pipefail
shopt -s nullglob globstar

# define TMPDIR, if it's empty
if [[ -z "$TMPDIR" ]]; then
    TMPDIR="/tmp"
fi
echo "TMPDIR: $TMPDIR"

# activate conda env
eval "$(conda shell.bash hook)"
conda activate mopoe
echo "CONDA_PREFIX: $CONDA_PREFIX"

METHOD="joint_elbo"  # NOTE: valid options are "joint_elbo", "poe", and "moe"
LIKELIHOOD_M1="laplace"
LIKELIHOOD_M2="laplace"
LIKELIHOOD_M3="categorical"
DIR_DATA="$PWD/data/data"
DIR_CLF="$PWD/trained_classifiers/trained_clfs_mst"
DIR_EXPERIMENT="$PWD/runs/MNIST_SVHN_Text/${METHOD}/non_factorized/${LIKELIHOOD_M1}_${LIKELIHOOD_M2}_${LIKELIHOOD_M3}"
PATH_INC_V3="$PWD/pt_inception-2015-12-05-6726825d.pth"
DIR_FID="$TMPDIR/MNIST_SVHN_text"

# copy data to $TMPDIR
cp -r "${DIR_DATA}/MNIST" "${TMPDIR}/"
cp -r "${DIR_DATA}/SVHN" "${TMPDIR}/"
cp -r "${DIR_DATA}/MNIST_SVHN" "${TMPDIR}/"

python Trainer.py --dir_data=$TMPDIR \
			--dir_clf="$DIR_CLF" \
			--dir_experiment="$DIR_EXPERIMENT" \
			--inception_state_dict="$PATH_INC_V3" \
			--dir_fid=$DIR_FID \
			--method=$METHOD \
			--style_m1_dim=0 \
			--style_m2_dim=0 \
			--style_m3_dim=0 \
			--class_dim=20 \
			--beta=2.5 \
			--likelihood_m1=$LIKELIHOOD_M1 \
			--likelihood_m2=$LIKELIHOOD_M2 \
			--likelihood_m3=$LIKELIHOOD_M3 \
			--batch_size=256 \
			--initial_learning_rate=0.001 \
			--eval_freq=1 \
			--eval_freq_fid=1 \
			--data_multiplications=20 \
			--num_hidden_layers=1 \
			--end_epoch=200 \
			--calc_nll \
			--eval_lr \
			--calc_prd \
			--use_clf
