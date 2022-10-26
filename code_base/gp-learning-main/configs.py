import numpy as np


def train_configs():
    deg2rad = np.pi / 180
    return {
        # model simplest_1 with:
        # 1st order noSwing dynamics
        # settling on dt = 0.5, Tf = 10000
        # done conditions: OOB, high swing, goalReached
        # settling on BS= 1024
        # no CA
        #
        "modelName": "GPModel-v1",  # simulation time step - seconds
        "test_": False,  # Test of train function
        "RLalgorithm": "PPO",
        "autoNormalizeObs": False,
        "autoNormalizeRew": False,
        "n_envs": 0,  # number of vectorized envs, if using trainVec
        # full: non-colliding uniform random R,s , 'constrained': only between certain areas
        "initialDistr": "full",
        # full: non-colliding uniform random R,s , 'constrained': only between certain areas
        "goalDistr": "constrained-safe",
        # initial and goal distributions only work when deterministic == False
        "isDeterministic_init": True,
        "isDeterministic_goal": True,
        "init_states": np.array([-1, 0]),
        "Tf": 100,  # T final - seconds
        "dt": 0.1,
        "rewardType": "positive",
        "obsScale": 8,
        # (self.obsScale + self.safeDist) defines the collision Upperbound
        "safeDist": 3,
        "obstacleType": "list",  # 'dynamic-constrained-dual' or 'list' or 'dynamic-full'
        # np.array([[20, 0],[36, 0]]),
        "obstacle_list": np.array([[40, 40 * np.pi / 180]]),
        "loadModel": 0,  # keep collision constraints and penalties in trainings
        "saveModel": 1,
        "runModel": 1,
        "controllerType": "Linear",
        "NNlayers": [64, 64],  # [64, 64],
        "NNact_fun": "relu",  # 'tanh' or 'relu' or whatever from tf.nn.*
        "log_dir": "./results/logs/",
        "uncertainties": {
            "type": "nominal",  # nominal (=0), fixed or grid or random
            "uncertU0": 0,  # for 'fixed' type
            "uncertU1": 0,  # for 'fixed' type
            # (for non- nominal) - magnitude of uncertainties (%)
            "uncert_mag": 0.5,
            "uncert_step": 0.1,
        },  # (for grid) - spacing between uncertainties
        "acceptable_ss_error": np.array(
            [
                0.02 * deg2rad,
                0.01 * deg2rad,
                0.02 * deg2rad,
                0.01 * deg2rad,
                1e-3,
                1e-3,
                0.02 * deg2rad,
                0.01 * deg2rad,
            ]
        ),  # (positive) acceptable tolerance of steady state error for early termination
        "Wrew_swing": 2e-4,
        "Wrew_pos": 1.0 / 100,
        # self.Wrew_pos*(self.maxSlew /self.R)**2# if we want to equalize
        "Wrew_slew": 2.5 / 100,
        "bonusOnGoalReach": False,  # whether or not give rew if goal reached
        # whether or not terminate an episode (done=True) if goal is reached
        "doneOnGoalReached": False,
        # 'doneOnCollision': True, #whether or not terminate if collision occured, reward positive automatically overwrites this to True
        "obsNormalizingW": [40, 40, 10 / 46.0, 2, 2, 50],
        # TRPO
        "totalTimesteps": 300000,
        "logging_freq": 50000,
        # 'TRPOparams':{
        #             'gamma': 0.99,
        #             'timesteps_per_batch': 1024*8,
        #             'max_kl': 0.000193,
        #             'cg_iters': 10,
        #             'lam': 0.9,
        #             'entcoeff': 0.01118,
        #             'cg_damping': 2.35e-5,
        #             'vf_stepsize': 0.00428,
        #             'vf_iters': 10,
        #             'verbose': 0,
        #             'tensorboard_log': './tmp/logs/graph',
        #             '_init_setup_model': True,
        #             # 'policy_kwargs': None,
        #             'full_tensorboard_log': False,
        #             'seed': None, #CHECK! SEED IS FIXED!
        #             'n_cpu_tf_sess': 1
        #         },
        "PPOparams": getPPOparams(),
        "saveOnBestMC": False,  # Monitor RL policy performance on the test_set
        "path_train_data": "data/avant_Trainingdata.pkl",
        "path_test_data": "data/avant_Testdata.pkl",
        "optimopts": {
            "optimizer": "SGD",
            "trials": 5,  # number of restarts
            "max_iter": 20,
        },
        "N_trajs": 100,
        "optimLogPath": "results/",
        "W_R": np.diag([0.5, 0.2]),
        "GP_training_iter": 500,
        "GPModel_datafields": getModelFields(),
        "RLalg_data_fields": get_RLalg_data_fields(),
        "verbose": 1,
        "target_state": np.array([0, 0]),
        "Force_CPU": False,
    }


# also add to fields


def getModelFields():
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
    ]


def get_RLalg_data_fields():
    return [
        "test_",  # T final - seconds
        "modelName",  # simulation time step - seconds
        "loadModel",  # keep collision constraints and penalties in trainings
        "saveModel",
        "totalTimesteps",
        "runModel",
        "logging_freq",
        "log_dir",
        "NNlayers",
        "controllerType",
        "gamma",
        "TRPOparams",
        "autoNormalizeObs",
        "autoNormalizeRew",
    ]


def getTRPOparams():
    return {  # TRPO default
        "gamma": 0.99,
        "timesteps_per_batch": 1024 * 16,
        "max_kl": 0.01,
        "cg_iters": 10,
        "lam": 0.98,
        "entcoeff": 0.0,
        "cg_damping": 0.01,
        "vf_stepsize": 0.0003,
        "vf_iters": 10,
        "verbose": 0,
        "tensorboard_log": None,
        "_init_setup_model": True,
        "full_tensorboard_log": False,
        "seed": None,
        "n_cpu_tf_sess": 1,
    }


def getPPOparams():
    return {
        "learning_rate": 0.0003,
        "n_steps": 2048,  # 4096*2,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_kl": None,
        "tensorboard_log": None,
        "create_eval_env": False,
        "verbose": 1,
        "seed": None,
        "device": "auto",
        "_init_setup_model": True,
    }


def test_configs():  # for testing, monte carlo, etc

    return {
        **train_configs(),
        "Tf": 300,  # T final - seconds
        "dt": 0.1,  # for simulations
        "N_MCruns": 25,
        "MCtest": False,  # test MonteCarlo
        "runRL": True,
        "runMPC": False,
        "uncert_mag": 0.5,  # magnitude of uncertainties (%)
        "uncert_step": 0.1,  # spacing between uncertainties
        "MC_dir": "./tmp/MonteCarlo/",
        "whichmodelMC": "best_model.zip",  # which model to load
        "isDeterministic_init": True,
        # full: non-colliding uniform random R,s , 'constrained': only between certain areas, 'grid' (only MC analysis)
        "initialDistr": "grid",
        # 'train_for_collision': False, # keep collision constraints and penalties in trainings
        "isDeterministic_goal": True,
        # full: non-colliding uniform random R,s , 'constrained': only between certain areas
        "goalDistr": "constrained",
        "Rinit": 40,
        "Sinit": 0 * np.pi / 180,
        "R_goal": 25,
        "s_goal": 90 * np.pi / 180,
        "uncertainties": {
            "type": "nominal",  # nominal (=0), fixed or grid or random
            "uncertU0": 0,  # for 'fixed' type
            "uncertU1": 0,  # for 'fixed' type
            # (for non- nominal) - magnitude of uncertainties (%)
            "uncert_mag": 0.5,
            "uncert_step": 0.1,
        },  # (for grid) - spacing between uncertainties
        # overwrite obstacle type to avoid mistakes:
        "obstacleType": "list",  # 'static'-'dynamic-constrained-dual' or 'list'
        # 'obstacle_list': np.array([[36, 0],[20, 0]]),
        "MPC_setup": {
            "n_horizon": 30,
            "t_step": 0.1,  # 'n_robust': 1,
            "store_full_solution": False,  # True,
        },
        # speed up do-mpc by changing IPOPT solver from 'mumps' to 'MA27'
        "MPC_solverIPOPT": "MA27",
        "MPC_suppress_output": True,
    }
