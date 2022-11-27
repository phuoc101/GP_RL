import numpy as np
import torch
from loguru import logger
import sys
import argparse

from cfg import configs
from models.GPModel import GPModel
from utils.data_loading import load_test_data
import os
from pathlib import Path

# * make the paths absolute instead of relative, to avoid headaches
dir_path = Path(os.path.dirname(os.path.abspath(__file__)))

def main(opts):
    print("abc")
    config = configs.get_train_config()
    config = {**config, "GP_training_iter": opts.gp_train_iter, "verbose": opts.verbose}
    model = GPModel(**config)
    model.initialize_model(
        path_model=opts.gpmodel,
        path_train_data=opts.train,
    )


    X_test,y_test = load_test_data(data_path=opts.test,output_torch=True, normalize=True)
    # convert to a single state/control input from batch of states/inputs
    X_test_single_input,y_test_single_input = torch.reshape(X_test[0],(1,3)), torch.reshape(y_test[0],(1,2))

    # custom input what we will need on gazebo
    # pos,vel,input = 0.0,0.0,0.5
    # X_input = torch.tensor(pos,vel,input]).reshape(1,3)


    X_test_single_input = X_test_single_input.to(model.device, model.dtype)
    y_test_single_input = y_test_single_input.to(model.device, model.dtype)

    logger.debug(f"input data shape {X_test_single_input.shape} ")
    logger.info(f"input data field [0] (current position) {X_test_single_input[0][0]} ")
    logger.info(f"input data field [1] (current velocity) {X_test_single_input[0][1]} ")
    logger.info(f"input data field [2] (current input) {X_test_single_input[0][2]} ")

    y_pred = model.predict(X_test_single_input)

    logger.debug(f"prediction data type {type(y_pred.mean)} ")
    logger.debug(f"prediction data field shape {y_pred.mean[0].shape} ")
    logger.info(f"prediction data field [0] (change in position) {y_pred.mean[0][0]} ") # to be integrated to get next state
    logger.info(f"prediction data field [1] (change in velocity) {y_pred.mean[0][1]} ")

    logger.debug(f"ground truth data type {type(y_test_single_input)} ")
    logger.debug(f"ground truth data shape {y_test_single_input.shape} ")
    logger.info(f"ground truth data field [0] (change in position) {y_test_single_input[0][0]} ")
    logger.info(f"ground truth data field [1] (change in velocity) {y_test_single_input[0][1]} ")



if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-gp",
        "--gpmodel",
        type=str,
        default=dir_path / "./results/GPmodel.pkl",
        help="Path to training data",
    )

    parser.add_argument(
        "--gp-train-iter",
        type=int,
        default=500,
        help="Maximum train iterations for GP model",
    )
    parser.add_argument(
        "-v,",
        "--verbose",
        type=int,
        default=1,
        help="Path to test data",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=dir_path / "../data/avant_TrainingData.pkl",
        help="Path to training data",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=dir_path / "../data/avant_TestData.pkl",
        help="Path to test data",
    )
    opts = parser.parse_args()
    main(opts)