import numpy as np


def get_train_config():
    return {
        "modelName": "GPModel-v1",  # simulation time step - seconds
        "GP_training_iter": 500,
        "GPModel_datafields": get_model_fields(),
        "verbose": 1,
        "target_state": np.array([0, 0]),
        "Force_CPU": False,
        "force_train": False,
    }


def get_model_fields():
    return [
        "Tf",  # T final - seconds
        "dt",  # simulation time step - seconds
        "train_for_collision",  # keep collision constraints and penalties in trainings
        "obsScale",
        "safeDist",  # (self.obsScale + self.safeDist) defines the collision Upperbound\
        "isDeterministic_init",
        "isDeterministic_goal",
        "initialDistr",
        "goalDistr",
        "Rinit",
        "Sinit",
        "R_goal",
        "s_goal",
        "obstacleType",
        "obstacle_dual",
        "obstacle_list",
        "uncertainties",
        "acceptable_ss_error",
        "simulation_type",
        "observations",
        "Wrew_swing",
        "Wrew_pos",
        "Wrew_slew",
        "obsNormalizingW",
        "bonusOnGoalReach",
        "doneOnGoalReached",
        "doneOnCollision",
        "rewardType",
        "verbose",
        "saveOnBestMC",
        "path_train_data",
        "path_test_data",
        "W_R",
        "GP_training_iter",
        "init_states",
        "target_state",
        "controllerType",
        "optimLogPath",
        "optimizerType",
        "optimopts",
        "N_trajs",
        "NNlayers",
        "Force_CPU",
        "force_train",
    ]
