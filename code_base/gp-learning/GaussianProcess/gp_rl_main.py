import sys
import argparse
from loguru import logger
from cfg import configs
from models.policy_optimizer import PolicyOptimizer


def main(opts):
    # set gp config
    gp_config = configs.get_gp_train_config()
    gp_config = {
        **gp_config,
        "normalize_train": opts.normalize,
        "GP_training_iter": opts.gp_training_iter,
        "force_train": opts.force_train_gp,
        "force_cpu": opts.force_cpu,
    }
    # set controller config
    controller_config = configs.get_controller_config()
    controller_config = {
        **controller_config,
        "force_cpu": opts.force_cpu,
        "state_dim": opts.state_dim,
        "control_dim": opts.control_dim,
    }
    # set optimizer config
    config = configs.get_optimizer_config()
    config = {
        **config,
        "verbose": 1,
        "path_train_data": opts.train_data,
        "path_test_data": opts.test_data,
        "optimizer_opts": {
            "optimizer": opts.optimizer,
            "trials": opts.trials,  # number of restarts
            "max_iter": opts.trial_max_iter,
        },
        "force_cpu": opts.force_cpu,
        "normalize_target": opts.normalize,
        "gp_config": gp_config,
        "controller_config": controller_config,
    }
    policy_optimizer = PolicyOptimizer(**config)
    policy_optimizer.optimize_policy()
    # policy_optimizer.plot_policy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--train-data", default="../data/boom_trial6_10hz.pkl", help="Path to training dataset")  # noqa
    parser.add_argument("--test-data", default="../data/boom_trial3_10hz.pkl", help="Path to test dataset")  # noqa
    parser.add_argument("--force_cpu", action="store_true", help="Force using CPU")  # noqa
    parser.add_argument("--normalize", action="store_true", help="Normalize training data") # noqa
    parser.add_argument("--force-train-gp", action="store_true", help="Force training GP model") # noqa
    parser.add_argument("--trials", type=int, default=5, help="Number of trials") # noqa
    parser.add_argument("--gp-training-iter", type=int, default=500, help="Max number of iterations to train GP model") # noqa
    parser.add_argument("--trial-max-iter", type=int, default=20, help="Max number of iterations per trials") # noqa
    parser.add_argument("--state-dim", type=int, default=2, help="Observation space dimension") # noqa
    parser.add_argument("--control-dim", type=int, default=1, help="Action space dimension") # noqa
    parser.add_argument("--optimizer", default="Adam", help="Optimizer type") # noqa
    parser.add_argument("--verbose", default="DEBUG", help="logging verbosity (DEBUG, INFO, WARNING, ERROR)") # noqa
    # fmt: on
    opts = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level=opts.verbose)
    main(opts)