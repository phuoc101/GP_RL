import time
import os
from copy import deepcopy
from loguru import logger
import torch
from pathlib import Path
import numpy as np

from models.GPModel import GPModel
from models.controller import Controller
from utils.data_loading import save_data, load_training_data, load_test_data
from utils.plot import plot_policy, plot_reward, plot_MC, plot_MC_non_det
from utils.torch_utils import get_tensor, set_device, set_device_cpu
from utils.init import generate_goal, generate_init_state

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
            set_device(self)
        else:
            logger.info("Forcing CPU as processor...")
            set_device_cpu(self)
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
            num_inputs=self.state_dim + 1,
            data_path=self.path_train_data,
            output_torch=True,
            normalize=True,
        )
        # load test data
        self.X_test, self.y_test = load_test_data(
            num_inputs=self.state_dim + 1, data_path=self.path_test_data
        )
        # angle (goal) generation offset from bounds, normalized
        self.angle_goal_offset = 5 * np.pi / 180 / self.std_states[0]
        self.get_reward = self.rew_exponential_PILCO
        if not Path("./results/reward_fun/reward_fun.png").exists():
            plot_reward(
                x_lb=self.x_lb,
                x_ub=self.x_ub,
                file_path="./results/reward_fun/reward_fun.png",
                get_reward=self.get_reward,
            )
        self.controller = self.get_controller()
        # self.reset()

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
                lr=self.learning_rate,
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
                lr=self.learning_rate,
            )
        elif optimizing_method == "NAdam":
            self.optimizer = torch.optim.NAdam(
                [
                    {"params": self.controller.parameters()},
                ],
                lr=self.learning_rate,
            )
        elif optimizing_method == "SGD":
            self.optimizer = torch.optim.SGD(
                self.controller.parameters(), lr=self.learning_rate
            )
        else:
            raise NotImplementedError

    def optimize_policy(self):
        """
        Optimize controller's parameter's
        """
        controller_log_dir = os.path.join(self.optimizer_log_dir, "controller")
        os.makedirs(controller_log_dir, exist_ok=True)
        maxiter = self.optimizer_opts["max_iter"]
        trials = self.optimizer_opts["trials"]
        all_optim_data = {"all_optimizer_data": []}
        os.makedirs(self.optimizer_log_dir, exist_ok=True)
        for tl in range(trials):
            # optimization algorithm:  self.configs['optimopts']['optimizer']
            logger.info(f"Starting optimization trial {tl+1}/{trials}...")
            self.randTensor = get_tensor(
                data=torch.randn((self.n_trajectories, self.state_dim, self.horizon)),
                device=self.device,
                dtype=self.dtype,
            )
            self.reset()
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
                loss, mean_error = self.opt_step_loss()
                loss = -loss
                loss.backward()
                logger.info(
                    "Optimize Policy: Iter {}/{} - Loss: {:.3f}".format(
                        i + 1, maxiter, loss.item()
                    )
                )
                logger.info("Mean error {:.5f}".format(mean_error))
                self.optimizer.step()
                # keep track of runtime and loss
                t2 = time.perf_counter() - t_start
                optimInfo["loss"].append(loss.item())
                optimInfo["time"].append(t2)

            plot_policy(
                controller=self.controller,
                x_lb=self.x_lb,
                x_ub=self.x_ub,
                policy_log_dir=os.path.join(self.optimizer_log_dir, "policies"),
                trial=tl + 1,
            )
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
            save_data(
                os.path.join(controller_log_dir, f"_trial_{tl}.pkl"), trial_save_info
            )
            # collect trial data into one
            all_optim_data["all_optimizer_data"].append(trial_save_info)
        save_data(os.path.join(controller_log_dir, "_all.pkl"), all_optim_data)

    # calculate predictions+reward without having everything in one big tensor
    def opt_step_loss(self):
        # n_trajectories = self.n_trajectories  # number of (parallel) trajectories
        # initial state+u and concat
        # state = torch.cat(n_trajectories * [self.obs_torch[None, :]], dim=0)
        # state = torch.cat(self.obs_torch, dim=0)
        state = deepcopy(self.obs_torch)
        u = self.controller(state)
        rew = self.rew_exp_torch_batch_all(state[:, 0])
        logger.debug("starting trajectory realization...")
        t1 = time.perf_counter()
        target_tensor = get_tensor(
            data=self.target_state, device=self.device, dtype=self.dtype
        )
        for k in range(self.horizon - 1):
            # get u:  state -> controller -> u
            u = self.controller(state - target_tensor)[:, 0]
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
            # input()
        mean_error = torch.mean(state - self.target_state[0])
        t_elapsed = time.perf_counter() - t1
        logger.debug(f"Predictions completed... elapsed time: {t_elapsed:.2f}s")
        return rew, mean_error

    def reset(self):
        # self.k_step = 0  # current simulation step number
        self.done = False
        self.time_reached = 1e4  # checkpoint for payload within acceptable limits

        # generate init/goal states
        initial_state = generate_init_state(
            self=self, is_det=self.is_deterministic_init, n_trajs=self.n_trajectories
        )
        self.target_state = generate_goal(self=self, is_det=self.is_deterministic_goal)
        # if self.normalize_target:
        #     self.target_state = np.divide(
        #         self.target_state - self.mean_states, self.std_states
        #     )

        self.state = torch.zeros(
            (self.horizon, self.state_dim), device=self.device, dtype=self.dtype
        )
        self.state = initial_state
        self.reward = torch.zeros((self.horizon), device=self.device, dtype=self.dtype)

        self.obs_torch = initial_state
        logger.info(
            "Reset complete: observation[0-10]: {}..., goal: {}".format(
                self.obs_torch[0:10], self.target_state
            )
        )
        # logger.info(f"observation type: {type(self.obs_torch)}")
        return self.obs_torch

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

    def calc_realizations(self, gp_model=None, controller=None):
        if gp_model is None:
            gp_model = self.gp_model
        if controller is None:
            controller = self.controller
        n_trajectories = self.n_trajectories  # number of (parallel) trajectories
        # initialize big tensor, keeping track of all variables
        M = get_tensor(
            torch.zeros(
                (n_trajectories, self.state_dim + self.control_dim, self.horizon)
            ),
            device=self.device,
            dtype=self.dtype,
        )
        # initial state+u and concat
        # state = deepcopy(self.obs_torch)
        state = generate_init_state(self=self, is_det=True, n_trajs=self.n_trajectories)
        self.randTensor = get_tensor(
            torch.randn((self.n_trajectories, self.state_dim, self.horizon)),
            device=self.device,
            dtype=self.dtype,
        )
        # assign data to M, memory := sys.getsizeof(M.storage())
        M[:, :-1, 0] = state  # M[:,:,k] := torch.cat( (state, u), dim = 1)
        logger.info("Starting Monte Carlo trajectory realization...")
        t1 = time.perf_counter()
        target_tensor = get_tensor(
            data=generate_goal(self=self, is_det=True),
            device=self.device,
            dtype=self.dtype,
        )

        for k in range(self.horizon - 1):
            with torch.no_grad():
                # get u:  state -> controller -> u
                M[:, -1, k] = controller(M[:, :-1, k] - target_tensor)[:, 0]
                # predict next state: s_{t+1} = GP(s,u)
                predictions = gp_model.predict(M[:, :, k])
            randtensor = (
                predictions.mean + predictions.stddev * self.randTensor[:, :, k]
            )
            # predict next state
            M[:, :-1, k + 1] = M[:, :-1, k] + randtensor

        t_elapsed = time.perf_counter() - t1
        logger.info(f"predictions completed... elapsed time: {t_elapsed:.2f}s")
        # logger.debug(f"size of randomTensor is {randtensor.shape}")
        return M

    def calc_realization_mean(
        self, gp_model=None, controller=None
    ):  # mean of GP predictions, no sampling
        if gp_model is None:
            gp_model = self.gp_model
        if controller is None:
            controller = self.controller
        n_trajectories = 1  # number of (parallel) trajectories
        # initialize big tensor, keeping track of all variables
        M = get_tensor(
            data=torch.zeros(
                (n_trajectories, self.state_dim + self.control_dim, self.horizon)
            ),
            device=self.device,
            dtype=self.dtype,
        )
        # initial state+u and concat
        # state = torch.cat(n_trajectories * [self.obs_torch[None, :]], dim=0)
        # state = torch.cat(self.obs_torch, dim=0)
        state = generate_init_state(self=self, is_det=True, n_trajs=n_trajectories)

        # assign data to M, memory := sys.getsizeof(M.storage())
        M[:, :-1, 0] = state  # M[:,:,k] := torch.cat( (state, u), dim = 1)
        target_tensor = get_tensor(
            data=generate_goal(self=self, is_det=True),
            device=self.device,
            dtype=self.dtype,
        )
        logger.info("Starting mean trajectory realization...")
        t1 = time.perf_counter()
        for k in range(self.horizon - 1):
            # get u:  state -> controller -> u
            with torch.no_grad():
                M[:, -1, k] = controller(M[:, :-1, k] - target_tensor)[:, 0]
                # predict next state: s_{t+1} = GP(s,u)
                predictions = gp_model.predict(M[:, :, k])
            px = predictions.mean
            # predict next state
            M[:, :-1, k + 1] = M[:, :-1, k] + px
        t_elapsed = time.perf_counter() - t1
        logger.info(f"predictions completed... elapsed time: {t_elapsed:.2f}s")
        return M

    def calc_realizations_non_det_init(
        self, n_trajs_sim, gp_model=None, controller=None
    ):
        if gp_model is None:
            gp_model = self.gp_model
        if controller is None:
            controller = self.controller
        # initialize big tensor, keeping track of all variables
        M = get_tensor(
            torch.zeros((n_trajs_sim, self.state_dim + self.control_dim, self.horizon)),
            device=self.device,
            dtype=self.dtype,
        )
        # initial state+u and concat
        state = generate_init_state(self=self, is_det=False, n_trajs=n_trajs_sim)
        self.randTensor = get_tensor(
            torch.randn((n_trajs_sim, self.state_dim, self.horizon)),
            device=self.device,
            dtype=self.dtype,
        )
        # assign data to M, memory := sys.getsizeof(M.storage())
        M[:, :-1, 0] = state  # M[:,:,k] := torch.cat( (state, u), dim = 1)
        logger.info(
            "Starting trajectory realization with non-deterministic initialization..."
        )
        t1 = time.perf_counter()
        for k in range(self.horizon - 1):
            with torch.no_grad():
                # get u:  state -> controller -> u
                M[:, -1, k] = controller(M[:, :-1, k])[:, 0]
                # predict next state: s_{t+1} = GP(s,u)
                predictions = gp_model.predict(M[:, :, k])
            px = predictions.mean
            # predict next state
            M[:, :-1, k + 1] = M[:, :-1, k] + px

        t_elapsed = time.perf_counter() - t1
        logger.info(f"predictions completed... elapsed time: {t_elapsed:.2f}s")
        # logger.debug(f"size of randomTensor is {randtensor.shape}")
        return M

    def MC_oneSweep(self, controller=None, gp_model=None):
        if gp_model is None:
            gp_model = self.gp_model
        if controller is None:
            controller = self.controller
        M_trajectories = (
            self.calc_realizations(gp_model, controller).cpu().detach().numpy()
        )
        M_mean = self.calc_realization_mean(gp_model, controller).cpu().detach().numpy()
        plot_MC(
            self.Tf,
            self.dt,
            self.target_state,
            self.x_lb,
            self.x_ub,
            M_mean,
            M_trajectories,
            save_dir=os.path.join(self.optimizer_log_dir, "MC_sim"),
        )
        M_trajectories_nondet = (
            self.calc_realizations_non_det_init(50, gp_model, controller)
            .cpu()
            .detach()
            .numpy()
        )
        plot_MC_non_det(
            self.Tf,
            self.dt,
            self.target_state,
            self.x_lb,
            self.x_ub,
            M_trajectories_nondet,
            save_dir=os.path.join(self.optimizer_log_dir, "MC_sim"),
        )