import sys
import argparse
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from gp_rl.cfg.configs import get_gp_train_config
from gp_rl.models.GPModel import GPModel
from gp_rl.utils.data_loading import load_data
from gp_rl.utils.plot import plot_gp
from gp_rl.utils.torch_utils import to_gpu, get_tensor
from gp_rl.utils.rl_utils import calc_realization_mean


def get_best_controller(controller_data):
    """Choose controller with the lowest loss (highest reward)

    Args:
        controller_data: Controller data loaded from pkl file generated
        from training

    Returns:
        The controller from the trial that results in the lowest loss
    """
    losses = []
    for controller in controller_data["all_optimizer_data"]:
        losses.append(controller["optimInfo"]["loss"][-1])
    best_controller_data = controller_data["all_optimizer_data"][np.argmin(losses)]
    return best_controller_data["controller_final"]


def plot_response(M_mean, dt):
    """Plot the system response over a period tf

    Args:
        M_mean: The realized trajectory, calculated from calc_realization_mean
        dt: Sampling time
    """
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
    # Parameters for model + controller
    state_dim = opts.state_dim
    control_dim = opts.control_dim
    # ========LOAD GP_MODEL========== #
    gpmodel = GPModel(**get_gp_train_config())
    gpmodel.initialize_model(
        path_model="./results/gp/GPmodel_{}.pkl".format(opts.joint),
        # uncomment the lines below for retraining
        # path_train_data=opts.train_gp_data,
        # force_train=opts.force_train_gp,
    )
    # =====RUN GP MODEL FOR 1 INSTANCE========#
    logger.info("=====SINGLE INSTANCE GP PREDICTION====")
    X_sample = [0.5, -1]
    print(get_tensor(X_sample))
    X_sample_tensor = get_tensor(X_sample).reshape(1, 2)
    pred = gpmodel.predict(X_sample_tensor)
    logger.info(f"Sample initial position: {X_sample}")
    logger.info(f"Single Instance GP model prediction (mean dx): {pred.mean.item()}")
    logger.info(
        "Next position with GP prediction: {}".format(
            (X_sample_tensor[0, 0] + pred.mean).item()
        )
    )
    logger.info("=====END SINGLE INSTANCE GP PREDICTION====\n\n")

    # ========TEST GP_MODEL OVER TEST SET====== #
    if opts.visualize_gp:
        plot_gp(
            gpmodel,
            joint=opts.joint,
            test_data=opts.test_gp_data,
            num_states=2,
        )
    logger.info("GP model loaded")

    # ========LOAD CONTROLLER========== #
    controller_data = load_data(
        "./results/controller/{}/_all_{}.pkl".format(opts.joint, opts.joint)
    )
    controller = get_best_controller(controller_data)
    to_gpu(controller)
    logger.info("Model loaded")

    # Parameters for running simulation
    tf = opts.tf
    dt = opts.dt
    init_state = np.array(opts.init_state)
    target_state = get_tensor(np.array(opts.target_state))
    # ======SINGLE INSTANCE CONTROLLER EXAMPLE========#
    # M_instace is tensor with size
    # [1 (1 trajectory), input_size (states+control), 1+1 (first one is init )]
    # Set tf=None (default) for single instance, dt as sampling time
    # calc_realization_mean retunrs mean prediction, for predictions that also includes
    # uncertainty, use calc_realization
    logger.info("=====SINGLE INSTANCE CONTROLLER PREDICTION====")
    M_instance = (
        calc_realization_mean(
            gp_model=gpmodel,
            controller=controller,
            state_dim=state_dim,
            control_dim=control_dim,
            dt=dt,
            init_state=init_state,
            target_state=target_state,
        )
        .cpu()
        .numpy()
    )
    logger.info(f"Single instance input: {M_instance[:, -1, 0]}")
    logger.info(f"Single instance state (positional error): {M_instance[:, 0, 1]}")
    logger.info("=====END SINGLE INSTANCE CONTROLLER PREDICTION====\n\n")

    # =======RUN MODEL+CONTROLLER OVER TF=======#
    # M_mean is tensor with size
    # [1 (1 trajectory), input_size (states+control), 1+1 (first one is init )]
    # Set dt as sampling time, tf as time to run simulation
    logger.info("=====CONTROLLER PREDICTION OVER TF====")
    M_mean = (
        calc_realization_mean(
            gp_model=gpmodel,
            controller=controller,
            state_dim=state_dim,
            control_dim=control_dim,
            tf=tf,
            dt=dt,
            init_state=init_state,
            target_state=target_state,
        )
        .cpu()
        .numpy()
    )
    plot_response(M_mean, dt)
    logger.info("=====END CONTROLLER PREDICTION OVER TF====\n\n")


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
    parser.add_argument("--joint", default="boom", type=str, help="Joint to control (boom, bucket, telescope)")  # noqa
    parser.add_argument("--train-gp-data", default="../data/boom_trial_6_10hz.pkl", type=str, help="Joint to control (boom, bucket, telescope)")  # noqa
    parser.add_argument("--test-gp-data", default="../data/boom_trial_1_10hz.pkl", type=str, help="Joint to control (boom, bucket, telescope)")  # noqa
    parser.add_argument("--verbose", default="DEBUG", type=str, help="Verbosity level (INFO, DEBUG, WARNING, ERROR)")  # noqa
    # fmt: on
    opts = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level=opts.verbose)
    main(opts)
