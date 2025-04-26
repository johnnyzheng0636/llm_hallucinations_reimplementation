#!/bin/bash
#SBATCH --job-name=hook_chunk    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=1                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=09:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal       # partition(large/normal/cpu) where you submit
#SBATCH --account=mscbdt2024     # only require for multiple projects
#SBATCH --output=hook_main.out       #output file

echo "start"
date
python3 main.py --model open_llama_7b --dataset capitals --train_exist --run_baseline
python3 main.py --model open_llama_7b --dataset trivia_qa --train_exist --run_baseline
python3 main.py --model open_llama_7b --dataset place_of_birth --train_exist --run_baseline
python3 main.py --model open_llama_7b --dataset founders --train_exist --run_baseline
date
echo "end"