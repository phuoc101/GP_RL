import time
import os
from copy import deepcopy
from loguru import logger
import torch
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from models.GPModel import GPModel
from models.controller import Controller
from utils.data_loading import save_data, load_training_data, load_test_data
from utils.grid import make_2D_normalized_grid

float_type = torch.float32
torch.set_default_dtype(torch.float32)


class PolicyOptimizer:
    def __init__(self, **kwargs):
        super(PolicyOptimizer, self).__init__()
        logger.debug("===== Configuring optimizer with parameters: ======")
        for key, value in kwargs.items():
            setattr(self, key, value)
            logger.debug(f"attribute {key}: {value}")
        # set device
        if not self.force_cpu:
            self.set_device()
        else:
            logger.info("Forcing CPU as processor...")
            self.set_device_cpu()
        self.gp_model = self.get_gp_model()
        self.horizon = round(self.Tf / self.dt)
        # load train data
        (
            self.X_train,
            self.y_train,
            self.mean_states,
            self.std_states,
            self.x_lb,
            self.x_ub,
        ) = load_training_data(
            data_path=self.path_train_data, output_torch=True, normalize=True
        )
        # load test data
        self.X_test, self.y_test = load_test_data(data_path=self.path_test_data)
        # angle (goal) generation offset from bounds, normalized
        self.angle_goal_offset = 5 * np.pi / 180 / self.std_states[0]
        # set device
        if not self.force_cpu:
            self.set_device()
        else:
            logger.info("Forcing CPU as processor...")
            self.set_device_cpu()
        # if self.RewType == 'exponential-single-target':
        #     self.get_reward = self.Rew_single_goal_pos
        self.get_reward = self.rew_exponential_PILCO
        if not Path("./results/reward_fun.png").exists():
            self.plot_reward(file_path="./results/reward_fun.png")
        self.reset()

    def get_gp_model(self):
        model = GPModel(**self.gp_config)
        model.initialize_model(
            path_model=self.gpmodel,
            path_train_data=self.path_train_data,
        )
        return model

    def get_controller(self):
        controller = Controller(**self.controller_config)
        model = controller.init_controller()
        return model

    def set_optimizer(self):
        optimizing_method = self.optimizer_opts["optimizer"]
        if optimizing_method == "RMSprop":
            self.optimizer = torch.optim.RMSprop(
                [
                    {"params": self.controller.parameters()},
                ],
                lr=0.05,
                alpha=0.99,
                eps=1e-08,
                weight_decay=0,
                momentum=0,
                centered=False,
            )
        elif optimizing_method == "Adam":
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.controller.parameters()},
                ],
                lr=0.05,
            )
        elif optimizing_method == "SGD":
            self.optimizer = torch.optim.SGD(self.controller.parameters(), lr=0.05)
        else:
            raise NotImplementedError

    # for SGD directly on policy parameters
    def optimize_policy(self):
        """
        Optimize controller's parameter's
        """
        maxiter = self.optimizer_opts["max_iter"]
        trials = self.optimizer_opts["trials"]
        all_optim_data = {"all_optimizer_data": []}
        os.makedirs(self.optimizer_log_dir, exist_ok=True)
        for tl in range(trials):
            # optimization algorithm:  self.configs['optimopts']['optimizer']
            # self.controller.randomize()
            logger.info(f"Starting optimization trial {tl}...")
            self.randTensor = self.tensor(
                torch.randn((self.n_trajectories, 2, self.horizon))
            )
            # Initialize a new controller
            self.controller = self.get_controller()
            controller_initial = deepcopy(self.controller)

            self.set_optimizer()
            t_start = time.perf_counter()
            # keeping track of variables for plots
            optimInfo = {"loss": [], "time": []}
            for i in range(maxiter):
                # with torch.autograd.detect_anomaly():
                self.optimizer.zero_grad()
                # reward = self.opt_predict()
                # loss = -reward
                loss = -self.optstep_loss()
                loss.backward()
                # plot_grad_flow_v2(self.controller.parameters())
                # - params: {self.controller[0].weight},
                # {self.controller[0].bias.cpu().detach.numpy()}
                logger.info(
                    "Optimize Policy: Iter {}/{} - Loss: {:.3f}".format(
                        i + 1, maxiter, loss.item()
                    )
                )
                self.optimizer.step()
                # keep track of runtime and loss
                t2 = time.perf_counter() - t_start
                optimInfo["loss"].append(loss.item())
                optimInfo["time"].append(t2)

            logger.info(
                "Controller's optimization: done in %.1f seconds with reward=%.3f."
                % (t2, loss.item())
            )
            trial_save_info = {
                "optimInfo": optimInfo,
                "controller_initial": controller_initial,
                "controller_final": deepcopy(self.controller),
                "mean_states": self.mean_states,
                "std_states": self.std_states,
            }
            save_data(f"{self.optimizer_log_dir}_trial_{tl}.pkl", trial_save_info)
            # collect trial data into one
            all_optim_data["all_optimizer_data"].append(trial_save_info)
        save_data(f"{self.optimizer_log_dir}_all.pkl", all_optim_data)

    # calculate predictions+reward without having everything in one big tensor
    def optstep_loss(self):
        n_trajectories = self.n_trajectories  # number of (parallel) trajectories
        # initial state+u and concat
        state = torch.cat(n_trajectories * [self.obs_torch[None, :]], dim=0)
        u = self.controller.forward(state)
        rew = self.rew_exp_torch_batch_all(state[:, 0])
        logger.debug("starting trajectory realization...")
        t1 = time.perf_counter()
        for k in range(self.horizon - 1):
            # get u:  state -> controller -> u
            u = self.controller.forward(state)[:, 0]
            # predict next state: s_{t+1} = GP(s,u)
            GPinput = torch.cat((state, u[:, None]), dim=1)
            predictions = self.gp_model.predict(GPinput)
            # use torch.normal(predictions.mean, predictions.stddev) or as below
            # predict next state
            next_state = (
                state
                + predictions.mean
                + torch.mul(predictions.stddev, self.randTensor[:, :, k])
            )
            # get reward
            rewards = self.rew_exp_torch_batch_all(next_state[:, 0])
            rew = rew + rewards
            state = next_state
        t_elapsed = time.perf_counter() - t1
        logger.debug(f"Predictions completed... elapsed time: {t_elapsed:.2f}")
        return rew

    def reset(self):
        self.k_step = 0  # current simulation step number
        self.done = False
        self.time_reached = 1e4  # checkpoint for payload within acceptable limits

        # generate init/goal states
        initial_state = self.generate_init_states()
        self.target_state = self.generate_goal()
        if self.normalize_target:
            self.target_state = np.divide(
                self.target_state - self.mean_states, self.std_states
            )

        self.state = torch.zeros(
            (self.horizon, self.state_dim), device=self.device, dtype=self.dtype
        )
        self.state[self.k_step] = initial_state
        self.reward = torch.zeros((self.horizon), device=self.device, dtype=self.dtype)

        self.obs_torch = initial_state
        logger.debug("reset() complete: observation:")
        logger.debug(self.obs_torch)
        logger.debug(f"observation type: {type(self.obs_torch)}")
        return self.obs_torch

    def generate_init_states(self):
        # in deterministic, default values are already in config
        if not self.is_deterministic_init:
            # initial state distribution
            if self.initial_dist == "full":  # non-colliding with obstacle
                init_states = np.random.uniform(self.x_lb[:2], self.x_ub[:2])

            elif self.initial_dist == "constrained":
                init_states = np.random.uniform(
                    self.x_lb + self.angle_goal_offset,
                    self.x_ub - self.angle_goal_offset,
                )

            elif self.initial_dist == "grid":
                return (
                    self.Rinit,
                    self.Sinit,
                )  # pass and let the MC plotter handle initial states
            else:
                raise NotImplementedError()
            return self.tensor(init_states)
        else:
            return self.tensor(self.init_states)

    def generate_goal(self):
        # in deterministic, default values are already in config
        if not self.is_deterministic_goal:
            if self.goal_distr == "full":
                goal_state = np.random.uniform(self.x_lb, self.x_ub)

            elif self.goal_distr == "constrained-safe":  # safe means far from bounds
                goal_state = np.random.uniform(
                    self.x_lb + self.angle_goal_offset,
                    self.x_ub - self.angle_goal_offset,
                )
            else:
                raise NotImplementedError()
            return goal_state
        else:
            return self.target_state

    def rew_exp_torch_batch_all(self, x):
        # returns sum of all rewards over all horizon over all trajs
        sigma2 = self.W_R[0, 0] ** 2
        state_error = x - self.target_state[0]
        reward = torch.exp(-torch.divide(torch.pow(state_error, 2), sigma2))
        return torch.sum(reward)

    def rew_exponential_PILCO(self, x):
        sigma2 = self.W_R[0, 0] ** 2
        state_error = x - self.target_state[0]
        reward = np.exp(-np.divide((state_error) ** 2, sigma2))
        return reward

    def tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=self.dtype, device=self.device)

    def set_device(self):
        self.is_cuda = torch.cuda.is_available()
        # self.Tensortype = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
        self.dtype = torch.float32
        self.device = torch.device("cuda:0") if self.is_cuda else torch.device("cpu")
        logger.info(f"using GPU: {self.is_cuda} - using processor: *({self.device})")

    def set_device_cpu(self):
        self.is_cuda = False
        # self.Tensortype = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
        self.dtype = torch.float32
        self.device = torch.device("cuda:0") if self.is_cuda else torch.device("cpu")
        logger.info(f"Forcing CPU... using processor: *({self.device})")

    def plot_reward(self, file_path):
        # calc normalized 2Dmap
        Xgd_2Dnormalized, Vgd_2Dnormalized = make_2D_normalized_grid(
            self.x_lb, self.x_ub, n_x=30
        )
        rewards = self.get_reward(Xgd_2Dnormalized.ravel())
        rewards = rewards.reshape(30, 30)
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.set_xlabel("X")
        ax.set_ylabel("V")
        ax.set_title("Reward function")
        # ax.contour(xg,vg, rew)
        pc = ax.pcolormesh(
            Xgd_2Dnormalized, Vgd_2Dnormalized, rewards, cmap=matplotlib.cm.jet
        )
        f.colorbar(pc)
        os.makedirs("./results/", exist_ok=True)
        plt.savefig(f"{file_path}")
        logger.info("reward plot saved to {}".format(file_path))

    def plot_policy(self, model=None, iter=1):  # this is a torch.nn policy
        if model is None:
            model = self.controller
        n_x = 100
        # calc normalized 2Dmap
        (
            stacked_inputs,
            Xgd_2Dnormalized,
            Vgd_2Dnormalized,
        ) = self.get_grid_stacked_inputs(n_x, n_x)
        # actions = self.linearModel(stacked_inputs) #debugging
        actions = self.gp_model.predict(stacked_inputs)
        actions = actions.reshape(n_x, n_x).cpu().detach().numpy()
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.set_xlabel("X")
        ax.set_ylabel("V")
        ax.set_title("Policy Plot")
        # ax.contour(xg,vg, rew)
        pc = ax.pcolormesh(
            Xgd_2Dnormalized, Vgd_2Dnormalized, actions, cmap=matplotlib.cm.jet
        )
        f.colorbar(pc)
        plt.savefig(f"{self.optimizer_log_dir}policy_plot_{iter}.png")
        if self.verbose > 0:
            print("policy plot saved...")

    def get_grid_stacked_inputs(self, n_x=100, n_y=20):
        Xgd_2Dnormalized, Vgd_2Dnormalized = make_2D_normalized_grid(
            self.x_lb, self.x_ub, n_x=n_x, n_y=n_y
        )
        stacked_inputs = np.stack(
            (Xgd_2Dnormalized.ravel(), Vgd_2Dnormalized.ravel())
        ).T
        stacked_inputs = self.tensor(stacked_inputs)
        return stacked_inputs, Xgd_2Dnormalized, Vgd_2Dnormalized
