import argparse
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from cfg.configs import get_gp_train_config
from models.GPModel import GPModel
from utils.data_loading import load_data
from utils.plot import plot_gp
from utils.torch_utils import to_gpu
from utils.rl_utils import calc_realization_mean


def get_best_controller(controller_data):
    losses = []
    for controller in controller_data["all_optimizer_data"]:
        losses.append(controller["optimInfo"]["loss"][-1])
    best_controller_data = controller_data["all_optimizer_data"][np.argmin(losses)]
    return best_controller_data["controller_final"]


def plot_response(M_mean, dt):
    time = np.arange(0, dt * (M_mean.shape[2]), dt)
    fig, ax = fig, ax = plt.subplots(2)
    ax[0].plot(time, M_mean[0, 0, :], label="positional error")
    ax[0].plot(
        time,
        np.ones(M_mean.shape[2]) * opts.target_state,
        "r--",
        label="goal",
    )

    ax[1].plot(time, M_mean[0, 1, :], label="control input")
    for a in ax:
        a.legend()

    plt.show()


def main(opts):
    # Define params for simulation
    Tf = opts.tf
    dt = opts.dt
    state_dim = opts.state_dim
    control_dim = opts.control_dim
    init_state = np.array(opts.init_state)
    target_state = np.array(opts.target_state)
    horizon = round(Tf / dt)
    # Init GP model
    gpmodel = GPModel(**get_gp_train_config())
    gpmodel.initialize_model(
        path_model="./results/gp/GPmodel.pkl",
        path_train_data="../data/boom_trial_6_10hz.pkl",
        force_train=opts.force_train_gp,
    )
    # Function to test GP Model
    if opts.visualize_gp:
        plot_gp(gpmodel, test_data="../data/boom_trial_1_10hz.pkl", num_states=2)
    logger.info("GP model loaded")

    controller_data = load_data("./results/controller/_all.pkl")
    controller = get_best_controller(controller_data)
    to_gpu(controller)
    logger.info("Model loaded")
    M_mean = (
        calc_realization_mean(
            gp_model=gpmodel,
            controller=controller,
            state_dim=state_dim,
            control_dim=control_dim,
            horizon=horizon,
            init_state=init_state,
            target_state=target_state,
        )
        .cpu()
        .numpy()
    )
    plot_response(M_mean, dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--tf", default=25, type=float, help="Time to run simulation")  # noqa
    parser.add_argument("--dt", default=0.1, type=float, help="Sampling time")  # noqa
    parser.add_argument("--state-dim", default=1, type=int, help="Number of observed states")  # noqa
    parser.add_argument("--control-dim", default=1, type=int, help="Number of controlled states")  # noqa
    parser.add_argument("--init-state", nargs="+", default=[-1], help="Define initial state (based on state-dim)")  # noqa
    parser.add_argument("--target-state", nargs="+", default=[0], help="Define goal state (based on state-dim)")  # noqa
    parser.add_argument("--visualize-gp", action="store_true", help="Visualize GP")  # noqa
    parser.add_argument("--force-train-gp", action="store_true", help="Force train GP Model again")  # noqa
    # fmt: on
    opts = parser.parse_args()
    main(opts)
