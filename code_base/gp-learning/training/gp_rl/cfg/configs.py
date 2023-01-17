import numpy as np


def get_gp_train_config():
    """Generate GP model configuration

    Returns:
        Dict containing configuration for GP models
    """
    return {
        "gp_training_iter": 500,
        "target_state": np.array([0, 0]),
        "torch_output": True,
        "normalize_train": False,
        "force_cpu": False,
        "force_train": False,
        "num_tasks": 1,
        "joint": "boom",
    }


def get_controller_config():
    """Generate NN controller configuration

    Returns:
        Dict containing configuration for NN controller
    """
    return {
        "state_dim": 2,  # number of observations
        "control_dim": 1,  # number of actions
        "controller_type": "NN",
        "NNlayers": [64, 64],
        "input_lim": 1,
    }


def get_optimizer_config():
    """Generate policy optimizer configuration

    Returns:
        Dict containing configuration for policy optimizer
    """
    return {
        "state_dim": 1,  # number of observations
        "force_train_gp": False,
        "control_dim": 1,  # number of actions
        "path_train_data": None,
        "path_test_data": None,
        "gp_training_iter": 500,
        "controller_type": "NN",
        "gpmodel": None,
        "force_cpu": False,
        "optimizer_opts": {
            "optimizer": "Adam",
            "trials": 5,  # number of restarts
            "max_iter": 20,
        },
        "optimizer_log_dir": "./results/",
        "n_trajectories": 100,
        "is_deterministic_init": True,
        # full: non-colliding uniform random R,s , 'constrained': only between certain
        # areas, 'grid' (only MC analysis)
        "initial_distr": "full",
        # 'train_for_collision': False, # keep collision constraints and penalties in
        # trainings
        "is_deterministic_goal": True,
        # full: non-colliding uniform random R,s , 'constrained': only between certain
        # areas
        "goal_distr": "constrained",
        "Rinit": 40,
        "Sinit": 0 * np.pi / 180,
        # initial and goal state
        "init_state": np.array([-0.5, 0]),
        "target_state": np.array([0, 0]),
        "Tf": 30,  # T final - seconds
        "dt": 0.1,  # in seconds, so frequency is 10Hz
        "W_R": np.diag([0.5, 0.2]),
        "normalize_target": True,
        "learning_rate": 0.01,
        "train_single_controller": False,
    }
