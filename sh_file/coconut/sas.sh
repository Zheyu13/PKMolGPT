#!/bin/bash -l
#$ -l h_rt=24:00:00
#$ -l mem=4G
#$ -l tmpfs=20G
#$ -pe smp 4
#$ -l gpu=1
#$ -cwd
#$ -N cssas2
#$ -o $HOME/log/cs2/cssas2.out
#$ -e $HOME/log/cs2/cssas2.err

module purge
module load default-modules
module load cuda/12.2.2/gnu-10.2.0
module load python/miniconda3/24.3.0-0
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate molgpt_env

echo "===== Step 1: Training model ====="
python ../code/train_c.py \
  --run_name cssas2 \
  --scaffold \
  --data_name coconut \
  --batch_size 256 \
  --num_props 1 \
  --props sas \
  --n_layer 8 \
  --learning_rate 6e-4 \
  --max_epochs 10

echo "===== Moving vocabulary file to parent directory ====="
cp data/coconut_stoi.json ../data/

echo "===== Step 2: Generating molecules using the trained model ====="
echo "===== Generation run $run ====="
time python ../code/generate_c2.py \
  --model_weight ../cond_gpt/weights/cssas2.pt \
  --data_name coconut \
  --batch_size 256 \
  --scaffold \
  --props sas \
  --csv_name ../g/cs2/cssas \
  --gen_size 10000 \
  --n_layer 8 \
  --block_size 200 \
  --vocab_size 252
