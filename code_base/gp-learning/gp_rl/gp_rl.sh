#!/bin/bash
# Check gp_rl_main.py for more options
#
# run training progress with 5 trials, each trial for 20 iterations, with 1 observable state (position), using NAdam optimizer,
# with non-deterministic initial position initialization (create tensor with <n_trajs> random initial positions), and retrain GP model
python gp_rl_main.py --verbose DEBUG --trials 5 --trial-max-iter 20 --num-states 1 --train-data ../data/boom_trial_6_10hz.pkl \
  --test-data ../data/boom_trial_1_10hz.pkl \
  --optimizer NAdam --nondet-init --force-train-gp \
  --verbose INFO --train-single-controller
