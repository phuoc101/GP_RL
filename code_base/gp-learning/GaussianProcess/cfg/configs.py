import numpy as np


def get_gp_train_config():
    return {
        "model_name": "GPModel-v1",  # simulation time step - seconds
        "gp_training_iter": 500,
        "verbose": 1,
        "target_state": np.array([0, 0]),
        "force_cpu": False,
        "force_train": False,
    }


def get_controller_config():
    return {
        "state_dim": 2,  # number of observations
        "control_dim": 1,  # number of actions
        "controller_type": "NN",
        "verbose": 1,
        "Tf": 100,  # T final - seconds
        "dt": 0.1,  # in seconds, so frequency is 10Hz
        "NNlayers": [64, 64],
    }


def get_optimizer_config():
    return {
        "path_train_data": "../../data/boom_trial6_10hz.pkl",
        "path_test_data": "../../data/boom_trial1_10hz.pkl",
        "gp_training_iter": 500,
        "verbose": 1,
        "controller_type": "NN",
        "gpmodel": "results/GPmodel.pkl",
        "traindata": "../../data/boom_trial6_10hz.pkl",
        "force_cpu": False,
        "force_train_gp": False,
        "optimizer_opts": {
            "optimizer": "Adam",
            "trials": 5,  # number of restarts
            "max_iter": 20,
        },
        "optimizer_log_dir": "results/",
        "n_trajectories": 100,
        "is_deterministic_init": True,
        # full: non-colliding uniform random R,s , 'constrained': only between certain
        # areas, 'grid' (only MC analysis)
        "initial_distr": "grid",
        # 'train_for_collision': False, # keep collision constraints and penalties in
        # trainings
        "is_deterministic_goal": True,
        # full: non-colliding uniform random R,s , 'constrained': only between certain
        # areas
        "goal_distr": "constrained",
        "Rinit": 40,
        "Sinit": 0 * np.pi / 180,
    }
