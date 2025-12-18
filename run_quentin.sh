#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=coop_dqn

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=0-05:00:00
# SBATCH --gres=gpu:1


python -m src.run seed=$1 \
  batch_size=16384 \
  tau=0.01 \
  optim.lr=0.004 \
  agent.hidden_size=32 \
  do_self_play=True \
  pretrain_iters=600 \
  agent.decay_eps=True \
  agent.eps_end=0.01 \
  agent.eps_start=0.5 \
  agent.eps_decay=600
