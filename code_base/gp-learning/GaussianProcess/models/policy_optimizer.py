import time
import os
from copy import deepcopy
from loguru import logger
import torch
import numpy as np

from models.GPModel import GPModel
from models.controller import Controller
from cfg import configs
from utils.data_loading import save_data, load_training_data, load_test_data

float_type = torch.float32
torch.set_default_dtype(torch.float32)


class PolicyOptimizer:
    def __init__(self, **kwargs):
        super(PolicyOptimizer, self).__init__()
        self.verbose = kwargs["verbose"]
        # verbose level: Trace full
        logger.debug("===== Configuring optimizer with parameters: ======")
        for key, value in kwargs.items():
            setattr(self, key, value)
            # verbose level: Trace Full
            logger.debug(f"attribute {key}: {value}")
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

    def get_gp_model(self):
        config = configs.get_gp_train_config()
        config = {
            **config,
            "GP_training_iter": self.gp_training_iter,
            "verbose": self.verbose,
            "force_train": self.force_train_gp,
            "force_cpu": self.force_cpu,
        }
        model = GPModel(**config)
        model.initialize_model(
            path_model=self.gpmodel,
            path_train_data=self.traindata,
        )
        return model

    def get_controller(self):
        config = configs.get_controller_config()
        config = {
            **config,
            "verbose": self.verbose,
            "force_cpu": self.force_cpu,
        }
        controller = Controller(**config)
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
        self.set_optimizer()
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

            # set fixed randnumber
            t_start = time.perf_counter()
            # reward = torch.zeros(1).float().cuda()
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
                        i, maxiter, loss.item()
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
            save_data(f"{self.optimizer_log_path}_trial_{tl}.pkl", trial_save_info)
            # collect trial data into one
            all_optim_data["all_optimizer_data"].append(trial_save_info)
        save_data(f"{self.optimizer_log_path}_all.pkl", all_optim_data)

    # calculate predictions+reward without having everything in one big tensor
    def optstep_loss(self):
        n_trajectories = self.n_trajectories  # number of (parallel) trajectories
        # initialize big tensor, keeping track of all variables
        # M = self.tensor(torch.zeros((N_traj, self.state_dim+self.control_dim, Horizon)))
        # initial state+u and concat
        state = torch.cat(n_trajectories * [self.obs_torch[None, :]], dim=0)
        u = self.controller.forward(state)
        rew = self.rew_exp_torchBatch_all(state[:, 0])
        # assign data to M, memory := sys.getsizeof(M.storage())
        # M[:,:-1,0] = state    #M[:,:,k] := torch.cat( (state, u), dim = 1)
        if self.verbose > 1:
            print("starting trajectory realization...")
            t1 = time.time()
        for k in range(self.Horizon - 1):
            # get u:  state -> controller -> u
            u = self.controller.forward(state)[:, 0]
            # predict next state: s_{t+1} = GP(s,u)
            GPinput = torch.cat((state, u[:, None]), dim=1)
            predictions = self.predict(GPinput)
            # px = torch.distributions.Normal(predictions.mean, predictions.stddev) ##BAD..
            # use torch.normal(predictions.mean, predictions.stddev) or as below
            # predict next state
            next_state = (
                state
                + predictions.mean
                + torch.mul(predictions.stddev, self.randTensor[:, :, k])
            )
            # get reward
            rewards = self.rew_exp_torchBatch_all(next_state[:, 0])
            rew = rew + rewards
            state = next_state
        if self.verbose > 1:
            t_elapsed = time.time() - t1
            print(f"predictions completed... elapsed time: {t_elapsed:.2f}")
            t1 = time.time()
        return rew

    def reset(self):
        self.k_step = 0  # current simulation step number
        self.done = False
        self.time_reached = 1e4  # checkpoint for payload within acceptable limits

        # generate init/goal states
        initial_state = self.generate_init_states()
        self.target_state = self.generate_goal()

        self.state = torch.zeros(
            (self.Horizon, self.state_dim), device=self.device, dtype=self.dtype
        )
        self.state[self.k_step] = initial_state
        self.reward = torch.zeros((self.Horizon), device=self.device, dtype=self.dtype)

        self.obs_torch = initial_state
        if self.verbose > 3:
            print("reset() complete: observation:")
            print(self.obs_torch)
            print(f"observation type: {type(self.obs_torch)}")
        # self.obs_numpy = self.obs_torch.cpu().numpy()
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
