import argparse
from cfg import configs
from models.policy_optimizer import PolicyOptimizer


def main(opts):
    config = configs.get_optimizer_config()
    config = {
        **config,
        "verbose": 1,
        "path_train_data": opts.train_data,
        "path_test_data": opts.test_data,
    }
    policy_optimizer = PolicyOptimizer(**config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        default="../data/boom_trial6_10hz.pkl",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--test-data",
        default="../data/boom_trial3_10hz.pkl",
        help="Path to training dataset",
    )
    opts = parser.parse_args()
    main()
