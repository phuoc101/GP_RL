import os
import torch
from loguru import logger
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as line
from utils.miscellaneous import get_tensor


def make_2D_normalized_grid(xlb, xub, n_x=30, n_y=30):
    def d2grid(x, v):
        X = np.zeros((x.shape[0], v.shape[0]))
        V = np.zeros((x.shape[0], v.shape[0]))
        for ix in range(x.shape[0]):
            for iv in range(v.shape[0]):
                X[ix, iv] = x[ix]
                V[ix, iv] = v[iv]
        return X, V

    """ Create 3x(ndgrids) """
    x = np.linspace(xlb[0], xub[0], n_x)
    v = np.linspace(xlb[-2], xub[-2], n_y)
    # normalize\
    # x = np.divide(x - mean_states[0], std_states[0])
    # if len(mean_states.shape) > 1:
    # v = np.divide(v - mean_states[-1], std_states[-1])

    # X,V = np.meshgrid(x,v)   #order is wrong with this method (X<->V swap)
    X1, V1 = d2grid(x, v)
    return X1, V1


def plot_reward(x_lb, x_ub, file_path, get_reward):
    # calc normalized 2Dmap
    Xgd_2Dnormalized, Vgd_2Dnormalized = make_2D_normalized_grid(x_lb, x_ub, n_x=30)
    rewards = get_reward(Xgd_2Dnormalized.ravel())
    rewards = rewards.reshape(30, 30)
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.set_xlabel("X")
    ax.set_ylabel("V")
    ax.set_title("Reward function")
    # ax.contour(xg,vg, rew)
    pc = ax.pcolormesh(Xgd_2Dnormalized, Vgd_2Dnormalized, rewards, cmap=cm.jet)
    f.colorbar(pc)
    os.makedirs("./results/reward_fun", exist_ok=True)
    plt.savefig(f"{file_path}", dpi=f.dpi)
    logger.info("reward plot saved to {}".format(file_path))


def plot_policy(controller, x_lb, x_ub, policy_log_dir, trial=1):
    n_x = 100
    # calc normalized 2Dmap
    # (
    #     stacked_inputs,
    #     Xgd_2Dnormalized,
    #     Vgd_2Dnormalized,
    # ) = self.get_grid_stacked_inputs(n_x, n_x)
    num_states = len(x_lb) - 1
    inputs = np.linspace(x_lb[0:num_states], x_ub[0:num_states], n_x)
    # actions = self.linearModel(stacked_inputs) #debugging
    actions = controller(get_tensor(inputs.reshape(-1, num_states)))
    # actions = actions.reshape(n_x, n_x).cpu().detach().numpy()
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"Policy Plot trial {trial}")
    ax.plot(inputs, actions.cpu().detach().numpy())
    os.makedirs(policy_log_dir, exist_ok=True)
    plt.savefig(os.path.join(policy_log_dir, f"policy_plot_{trial}.png"), dpi=100)
    logger.info("policy plot saved...")


def get_grid_stacked_inputs(
    x_lb, x_ub, n_x=100, n_y=20, device=torch.device("cuda:0"), dtype=torch.float32
):
    Xgd_2Dnormalized, Vgd_2Dnormalized = make_2D_normalized_grid(
        x_lb, x_ub, n_x=n_x, n_y=n_y
    )
    stacked_inputs = np.stack((Xgd_2Dnormalized.ravel(), Vgd_2Dnormalized.ravel())).T
    stacked_inputs = get_tensor(data=stacked_inputs)
    return stacked_inputs, Xgd_2Dnormalized, Vgd_2Dnormalized


def plot_MC(Tf, dt, target, x_lb, x_ub, M_mean, M_trajs, save_dir=None):
    plt.rc("font", family="serif")

    fig_pos, ax_pos = plt.subplots(1, 1)
    # fig2D, ax2D = plt.subplots(1, 1)
    fig_u, ax_u = plt.subplots(1, 1)
    fig_pos.set_size_inches((7, 6))

    time_vec = np.arange(0, dt * (M_trajs.shape[2]), dt)
    time_vec_horizon = np.arange(0, Tf, dt)
    # one-time plots
    targetlineopt = "g--"
    targetlinewidth = 1.6
    targetlinealpha = 0.9
    zo1 = 20
    # target R
    ax_pos.plot(
        time_vec_horizon,
        np.repeat(target[0], len(time_vec_horizon)),
        targetlineopt,
        linewidth=targetlinewidth,
        alpha=targetlinealpha,
        zorder=zo1,
        label="Goal",
    )

    # Monte Carlo trajectories
    total_realizations = M_trajs.shape[0] + 1  # + mean trajectory
    for k in range(total_realizations):
        if k == total_realizations - 1:  # last entry
            lineopt = "r-"  # red line (nominal)
            zo = 20  # zorder (top plot layer)
            default_alpha = 0.9
            x_data = M_mean[0, 0, :]
            u_data = M_mean[0, -1, :]
            ax_pos.plot(
                time_vec,
                x_data,
                lineopt,
                alpha=default_alpha,
                zorder=zo,
                label="Mean trajectory",
            )
            ax_u.plot(
                time_vec,
                u_data,
                lineopt,
                alpha=default_alpha,
                zorder=zo,
                label="Mean input",
            )
        else:
            x_data = M_trajs[k, 0, :]
            u_data = M_trajs[k, -1, :]
            lineopt = "b-"  # blue line (Monte Carlo M_trajs)
            zo = 0  # zorder (lowest plot layer)
            default_alpha = 0.01
            # label_x = "Monte Carlo trajectory"
            # label_u = "Monte Carlo input"
            if k == total_realizations - 2:
                ax_pos.plot(
                    time_vec,
                    x_data,
                    lineopt,
                    alpha=default_alpha,
                    zorder=zo,
                    label="Monte Carlo trajectories",
                )
                ax_u.plot(
                    time_vec,
                    u_data,
                    lineopt,
                    alpha=default_alpha,
                    zorder=zo,
                    label="Monte Carlo inputs",
                )
            else:
                ax_pos.plot(time_vec, x_data, lineopt, alpha=default_alpha, zorder=zo)
                ax_u.plot(time_vec, u_data, lineopt, alpha=default_alpha, zorder=zo)

        # rewards
        # ax5[3].plot(timeVec, self.reward, lineopt, zorder = zo)
    plot_max_x = 1.2 * x_ub[0]
    ax_pos.set_ylabel("Position")
    ax_pos.set_ylim((-plot_max_x, plot_max_x))
    ax_pos.set_xlim((0, time_vec[-1]))

    ax_u.set_ylabel("control input")

    # ax2D.grid(linestyle="--")
    for Q in [ax_pos, ax_u]:
        Q.grid(linestyle="--")
        Q.set_xlim((0, time_vec[-1]))
        # manually generate labels
        handles, labels = Q.get_legend_handles_labels()
        lines = [
            line.Line2D(
                [],
                [],
                color=handle.get_color(),
                linestyle=handle.get_linestyle(),
                label=label,
            )
            for handle, label in zip(handles, labels)
        ]
        Q.legend(
            handles=lines,
            bbox_to_anchor=(1.005, 1),
            borderaxespad=0.0,
            frameon=False,
        )
    if save_dir is not None:
        # save figures but do not show
        os.makedirs(save_dir, exist_ok=True)
        fig_pos.savefig(os.path.join(save_dir, "_fig_pos.png"))
        fig_u.savefig(os.path.join(save_dir, "_fig_u.png"))

        # save figures pdf
        fig_pos.savefig(os.path.join(save_dir, "_fig_pos.pdf"))
        fig_u.savefig(os.path.join(save_dir, "_fig_u.pdf"))

    else:
        plt.show()


def plot_MC_non_det(Tf, dt, target, x_lb, x_ub, M_trajs_non_det, save_dir=None):
    plt.rc("font", family="serif")

    fig_pos, ax_pos = plt.subplots(1, 1)
    # fig2D, ax2D = plt.subplots(1, 1)
    fig_u, ax_u = plt.subplots(1, 1)
    fig_pos.set_size_inches((7, 6))

    time_vec = np.arange(0, dt * (M_trajs_non_det.shape[2]), dt)
    time_vec_horizon = np.arange(0, Tf, dt)
    # one-time plots
    targetlineopt = "g--"
    targetlinewidth = 1.6
    targetlinealpha = 0.9
    zo1 = 20
    # target R
    ax_pos.plot(
        time_vec_horizon,
        np.repeat(target[0], len(time_vec_horizon)),
        targetlineopt,
        linewidth=targetlinewidth,
        alpha=targetlinealpha,
        zorder=zo1,
        label="Goal",
    )

    # Monte Carlo trajectories
    total_realizations = M_trajs_non_det.shape[0]  # + mean trajectory
    for k in range(total_realizations):
        x_data = M_trajs_non_det[k, 0, :]
        u_data = M_trajs_non_det[k, -1, :]
        # lineopt = "b-"  # blue line (Monte Carlo M_trajs)
        zo = 0  # zorder (lowest plot layer)
        default_alpha = 0.5
        # label_x = "Monte Carlo trajectory"
        # label_u = "Monte Carlo input"
        ax_pos.plot(time_vec, x_data, alpha=default_alpha, zorder=zo)
        ax_u.plot(time_vec, u_data, alpha=default_alpha, zorder=zo)

        # rewards
        # ax5[3].plot(timeVec, self.reward, lineopt, zorder = zo)
    plot_max_x = 1.2 * x_ub[0]
    ax_pos.set_ylabel("Position")
    ax_pos.set_ylim((-plot_max_x, plot_max_x))
    ax_pos.set_xlim((0, time_vec[-1]))

    ax_u.set_ylabel("control input")

    # ax2D.grid(linestyle="--")
    for Q in [ax_pos, ax_u]:
        # manually generate labels
        Q.grid(linestyle="--")
        Q.set_xlim((0, time_vec[-1]))
        handles, labels = Q.get_legend_handles_labels()
        lines = [
            line.Line2D(
                [],
                [],
                color=handle.get_color(),
                linestyle=handle.get_linestyle(),
                label=label,
            )
            for handle, label in zip(handles, labels)
        ]
        Q.legend(
            handles=lines,
            bbox_to_anchor=(1.005, 1),
            borderaxespad=0.0,
            frameon=False,
        )

    if save_dir is not None:
        # save figures but do not show
        os.makedirs(save_dir, exist_ok=True)
        fig_pos.savefig(os.path.join(save_dir, "_fig_pos_nondet.png"))
        fig_u.savefig(os.path.join(save_dir, "_fig_u_nondet.png"))

        # save figures pdf
        fig_pos.savefig(os.path.join(save_dir, "_fig_pos_nondet.pdf"))
        fig_u.savefig(os.path.join(save_dir, "_fig_u_nondet.pdf"))

    else:
        plt.show()
