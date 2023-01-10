#!/bin/bash
# Check gp_rl_main.py for more options
#
# run training progress with 5 trials, each trial for 20 iterations, with 1 observable state (position), using NAdam optimizer,
# with non-deterministic initial position initialization (create tensor with <n_trajs> random initial positions), and retrain GP model
# For boom
python gp_rl_main.py --verbose DEBUG --trials 5 --trial-max-iter 20 \
  --num-states 1 --train-data ../data/boom_trial_6_10hz.pkl \
  --test-data ../data/boom_trial_1_10hz.pkl --tf 10 --dt 0.1 \
  --optimizer NAdam --nondet-init --force-train-gp \
  --verbose INFO --joint boom --input-lim 0.2\
  --gpmodel ./results/gp/GPmodel_boom.pkl --plot-mc
# For bucket
python gp_rl_main.py --verbose DEBUG --trials 5 --trial-max-iter 20 \
  --num-states 1 --train-data ../data/bucket_trial_2_10hz.pkl \
  --test-data ../data/bucket_trial_2_10hz.pkl --tf 10 --dt 0.1  \
  --optimizer NAdam --nondet-init --force-train-gp \
  --verbose INFO --joint bucket --input-lim 0.5\
  --gpmodel ./results/gp/GPmodel_bucket.pkl --plot-mc
# For telescope
python gp_rl_main.py --verbose DEBUG --trials 5 --trial-max-iter 20 \
  --num-states 1 --train-data ../data/telescope_trial_1_10hz.pkl \
  --test-data ../data/telescope_trial_1_10hz.pkl --tf 10 --dt 0.1  \
  --optimizer NAdam --nondet-init --force-train-gp \
  --verbose INFO --joint telescope --input-lim 0.5\
  --gpmodel ./results/gp/GPmodel_telescope.pkl --plot-mc
