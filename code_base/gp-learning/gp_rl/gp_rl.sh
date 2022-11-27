#!/bin/bash
python gp_rl_main.py --verbose DEBUG --trials 5 --trial-max-iter 20 --num-states 1  --train-data ../data/boom_xu_trial6_10hz.pkl --test-data ../data/boom_xu_trial1_10hz.pkl --normalize --state-dim 1
