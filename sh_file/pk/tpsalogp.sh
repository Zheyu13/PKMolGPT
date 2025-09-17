#!/bin/bash -l
#$ -l h_rt=12:00:00
#$ -l mem=4G
#$ -l tmpfs=20G
#$ -pe smp 4
#$ -l gpu=1
#$ -cwd
#$ -N ptpsalogp
#$ -o $HOME/log/p2/ptpsalogp.out
#$ -e $HOME/log/p2/ptpsalogp.err

module purge
module load default-modules
module load cuda/12.2.2/gnu-10.2.0
module load python/miniconda3/24.3.0-0
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate molgpt_env

echo "===== Step 1: Training model ====="
python ../code/train_pk.py \
  --run_name ptpsalogp2 \
  --data_name coconutpk \
  --batch_size 256 \
  --num_props 2 \
  --props tpsa logp \
  --n_layer 8 \
  --learning_rate 6e-4 \
  --max_epochs 10

echo "===== Moving vocabulary file to parent directory ====="
cp data/coconutpk_stoi.json ../data/

echo "===== Step 2: Generating molecules using the trained model ====="
echo "===== Generation run $run ====="
time python ../code/generate_pk.py \
  --model_weight ../cond_gpt/weights/ptpsalogp2.pt \
  --data_name coconutpk \
  --props tpsa logp \
  --csv_name ../g/p2/ptpsalogp \
  --gen_size 10000 \
  --n_layer 8 \
  --block_size 300 \
  --vocab_size 113
