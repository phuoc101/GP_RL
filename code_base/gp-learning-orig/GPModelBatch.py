"""
GP model for batch querying of inputs, with propagation and
parameterized controller
"""
from copy import deepcopy
import matplotlib
from pathlib import Path
import time
from utils import load_trainingData, load_data, save_data
from utils import make_2DnormalizedGrid
import os
from gym import spaces
import gym
import numpy as np
from matplotlib import pyplot as plt
import gpytorch
import sys
import torch

float_type = torch.float32
torch.set_default_dtype(torch.float32)


class GPModel(gym.Env):
    n_obs = 2
    n_actions = 1
    state_dim = n_obs
    control_dim = n_actions
    action_space = spaces.Box(low=-1, high=1, shape=(n_actions,), dtype="float32")
    observation_space = spaces.Box(
        low=-50.0, high=50.0, shape=(n_obs,), dtype="float32"
    )

    def __init__(self, **kwargs):
        super(GPModel, self).__init__()
        self.data_fields = kwargs["GPModel_datafields"]
        self.verbose = kwargs["verbose"]
        # Configure with keyword args
        if self.verbose > 0:  # verbose level: Basic/Critical
            print("Configuring model with parameters:")
        for key, value in kwargs.items():
            if key in self.data_fields:
                setattr(self, key, value)
                if self.verbose > 3:  # verbose level: Trace Full
                    print("attribute {}: {}".format(key, value))

        self.configs = kwargs
        # set device
        if not self.Force_CPU:
            self.set_processor()
        else:
            print("Forcing CPU as processor...")
            self.set_processor_cpu()
        # default simulation and machine-specific params
        self.Horizon = round(self.Tf / self.dt)
        self.timeVec = np.arange(0, self.Tf, self.dt)

        # load data

        (
            self.X,
            self.Y,
            self.mean_states,
            self.std_states,
            self.x_lb,
            self.x_ub,
        ) = load_trainingData(self.path_train_data, output="torch", normalize_=True)
        self.Xtest, self.Ytest, _, _, _, _ = load_trainingData(
            self.path_test_data, output="torch", normalize_=True
        )
        # convert to GPU if required
        self.X = self.X.to(self.device, self.dtype)
        self.Y = self.Y.to(self.device, self.dtype)
        self.Xtest = self.Xtest.to(self.device, self.dtype)
        self.Ytest = self.Ytest.to(self.device, self.dtype)
        # augment data
        self.augmentData()
        # controller
        self.createController()
        # optimizer
        self.optimizer = None

        # angle (goal) generation offset from bounds, normalized
        self.angle_goal_offset = 5 * np.pi / 180 / self.std_states[0]
        # Train GP model
        # if self.is_cuda:
        #     path_model = './results/GPmodel_cuda.pkl'
        # else:
        path_model = "./results/GPmodel.pkl"

        if not Path(path_model).exists():
            # initialize models, train, save
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=2
            ).to(device=self.device, dtype=self.dtype)
            self.model = BatchIndependentMultitaskGPModel(
                self.X, self.Y, self.likelihood
            ).to(device=self.device, dtype=self.dtype)
            self.trainGPModel()
            info = {
                "model": self.model,
                "likelihood": self.likelihood,
            }
            os.makedirs("./results/", exist_ok=True)
            save_data(path_model, info)

        else:  # load models
            info = load_data(path_model)
            self.model = info["model"].to(device=self.device, dtype=self.dtype)
            self.likelihood = info["likelihood"].to(
                device=self.device, dtype=self.dtype
            )

        # reward functions and goal(s)
        self.target_normalized = np.divide(
            self.target_state - self.mean_states, self.std_states
        )
        self.getReward = self.Rew_exponential_PILCO
        # if self.RewType == 'exponential-single-target':
        #     self.getReward = self.Rew_single_goal_pos
        self.max_positive_rew = (2 * self.x_ub) ** 2
        if not Path("./results/reward_fun.png").exists():
            self.plotReward(file_path="./results/reward_fun.png")
        self.reset()

    def set_processor(self):
        self.is_cuda = torch.cuda.is_available()
        # self.Tensortype = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
        self.dtype = torch.float32
        self.device = torch.device("cuda:0") if self.is_cuda else torch.device("cpu")
        if self.verbose > 0:
            print(f"using GPU: {self.is_cuda} - using processor: *({self.device})")

    def set_processor_cpu(self):
        self.is_cuda = False
        # self.Tensortype = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
        self.dtype = torch.float32
        self.device = torch.device("cuda:0") if self.is_cuda else torch.device("cpu")
        if self.verbose > 0:
            print(f"Forcing CPU... using processor: *({self.device})")

    def createController(self):
        if self.controllerType == "Linear":
            if self.verbose > 0:
                print("initializing Linear controller")
            self.linearModel = torch.nn.Linear(
                self.state_dim, self.control_dim, device=self.device, dtype=self.dtype
            )
            self.saturation = torch.nn.Hardtanh()
            self.controller = torch.nn.Sequential(self.linearModel, self.saturation)
        else:
            if self.verbose > 0:
                print(f"initializing NN controller with layers {self.NNlayers}")
            self.controller = torch.nn.Sequential(
                torch.nn.Linear(
                    self.state_dim,
                    self.NNlayers[0],
                    device=self.device,
                    dtype=self.dtype,
                ),
                torch.nn.Linear(
                    self.NNlayers[0],
                    self.NNlayers[1],
                    device=self.device,
                    dtype=self.dtype,
                ),
                torch.nn.Linear(
                    self.NNlayers[1],
                    self.control_dim,
                    device=self.device,
                    dtype=self.dtype,
                ),
                torch.nn.Hardtanh(),
            )

        self.controller.predict = self.controller.forward

    def set_controller(self, controller=None):
        if controller is None:  # old method, only for Linear controller
            # set controller params (this is for testing) W = [0.23619299, 0.01063023], b = [3.10975102]
            controller_dict = self.controller[0].state_dict()
            controller_dict["weight"] = self.tensor([[-0.23619299, -0.2]])
            controller_dict["bias"] = self.tensor(
                [0.0]
            )  # have to explicitly set to zero
            # controller_dict['bias'] = self.tensor([3.10975102])
            # reload the model with the values
            self.controller[0].load_state_dict(controller_dict)
        else:
            controller_dict = controller.state_dict()
            self.controller.load_state_dict(controller_dict)

    def augmentData(self):
        # first set all dV = 0
        self.Y[:, -1] = self.Y[:, -1] * 0
        stacked_inputs, _, _ = self.get_grid_stacked_inputs(n_x=70, n_y=20)
        GPinputs = self.tensor(
            torch.cat((stacked_inputs, stacked_inputs[:, 1, None] * 0), dim=1)
        )
        GPtargets = self.tensor(torch.zeros((GPinputs.shape[0], self.Y.shape[1])))
        self.X = torch.cat((self.X, GPinputs), dim=0)
        self.Y = torch.cat((self.Y, GPtargets), dim=0)

    def trainGPModel(self):
        if self.verbose > 0:
            print("training GP models on data...")
        # ---- Optimize GP ----#
        t1_modelTraining = time.time()  # time the training
        training_iter = self.GP_training_iter

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            [
                # Includes all submodel and all likelihood parameters
                {"params": self.model.parameters()},
            ],
            lr=0.1,
        )
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.X)
            # Calc loss and backprop gradients
            loss = -mll(output, self.Y)
            loss.backward()
            if self.verbose > 2:
                print(
                    "Iter {:d}/{:d} - Loss: {:.3f}  noise: {}\n lengthscales: {} ".format(
                        i + 1,
                        training_iter,
                        loss.item(),
                        self.model.likelihood.noise.item(),
                        self.model.covar_module.base_kernel.lengthscale.cpu()
                        .detach()
                        .numpy()
                        .ravel(),
                    )
                )
            optimizer.step()

        t2_modelTraining = time.time()
        elapsed_modelTraining = t2_modelTraining - t1_modelTraining
        if self.verbose > 0:
            print(
                f"GP models trained in {elapsed_modelTraining:.3f} seconds, with {len(self.X)} data points"
            )
        # set models into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

    def reset(self):
        self.kStep = 0  # current simulation step number
        self.done = False
        self.timeReached = 1e4  # checkpoint for payload within acceptable limits

        # generate init/goal states
        initial_state = self.generate_init_states()
        self.target_state = self.generate_goal()

        self.state = torch.zeros(
            (self.Horizon, self.state_dim), device=self.device, dtype=self.dtype
        )
        self.state[self.kStep] = initial_state
        self.reward = torch.zeros((self.Horizon), device=self.device, dtype=self.dtype)

        self.obs_torch = initial_state
        if self.verbose > 3:
            print("reset() complete: observation:")
            print(self.obs_torch)
            print(f"observation type: {type(self.obs_torch)}")
        # self.obs_numpy = self.obs_torch.cpu().numpy()
        return self.obs_torch

    def predict(self, X_):  # predict the output from input X* using GP models
        if self.verbose > 3:
            # //TODO: try passing tensors around to see if it improves speed
            print("getting prediction(s) from GP Model:")
        with gpytorch.settings.fast_pred_var():  # torch.no_grad(),
            observed_pred = self.likelihood(self.model(X_))
        return observed_pred

    def step(self, u):
        x = self.state[self.kStep]
        # concatenate (x,u) for GP input
        u = torch.tensor(u, dtype=float_type)
        X_ = torch.cat((self.obs_torch, u))  # //TODO fix concat
        # take step
        self.state_difference_torch = self.predict(X_[None, :])
        self.obs_torch = self.obs_torch.add(self.state_difference_torch)
        self.obs_numpy = self.obs_torch.numpy()
        self.state[self.kStep + 1] = self.obs_numpy
        self.kStep += 1
        # set outputs
        reward = self.getReward(self.obs_numpy[0])
        self.reward[self.kStep] = reward
        self.checkDone()

        if self.verbose > 3:
            print(
                f"taking step #{self.kStep}: u={u}, X={x}, X_new={self.obs_numpy}, reward={reward}"
            )
        return self.obs_numpy, reward, self.done, {}

    def checkDone(self):
        if self.kStep >= self.Horizon - 1 or self.isOOB():
            # and commenting this out for performance: #or self.isInfState()
            self.done = True

        if self.doneOnGoalReached:
            if self.isGoalReached():
                self.done = True

    def isOOB(self):
        # exit when out of bounds
        if (
            self.state[self.kStep, 0] > self.x_ub[0]
            or self.state[self.kStep, 0] < self.x_lb[0]
        ):
            return True
        else:
            return False

    def isGoalReached(self):
        # terminate simul after goal is reached
        if np.all(np.abs(self.get_state_error()) < self.acceptable_ss_error):
            if self.timeReached > 1e3:  # record time
                self.timeReached = self.timeVec[self.kStep]
            return True
        else:
            return False

    def Rew_single_goal_pos(self, x):
        kStep = self.kStep
        state_error = self.get_state_error(x)

        # total reward
        self.reward[kStep] = -self.Wrew * (state_error**2) + self.max_positive_rew
        return self.reward[kStep]

    def Rew_single_goal_neg(self, obs):
        pass

    def get_state_error(self, x):
        state_error = x - self.target_normalized
        return state_error[0]

    def Rew_exponential_PILCO(self, x):
        sigma2 = self.W_R[0, 0] ** 2
        state_error = x - self.target_normalized[0]
        reward = np.exp(-np.divide((state_error) ** 2, sigma2))
        return reward

    # M is a batch of trajs with dim (N_traj, D, Horizon)
    def rew_exp_torchBatch_all(self, x):
        sigma2 = self.W_R[0, 0] ** 2
        state_error = x - self.target_normalized[0]
        reward = torch.exp(-torch.divide(torch.pow(state_error, 2), sigma2))
        # returns sum of all rewards over all horizon over all trajs
        return torch.sum(reward)

    # def Rew_exponential_PILCO_Torch(self, x):
    #     '''
    #     Saturating reward function used in PILCO, modified from https://github.com/jaztsong/PILCO-gpytorch
    #     '''
    #     state_error = self.get_state_error(x)
    #     s = 0
    #     SW = s @ self.W_R

    #     X, _ = torch.solve(torch.t(self.W_R),(torch.eye(self.state_dim, dtype=float_type).cuda() + SW) )

    #     muR = torch.exp(-(state_error) @ torch.t(X) @ torch.t(state_error)/2) / \
    #             torch.sqrt(torch.det(torch.eye(self.state_dim, dtype=float_type).cuda() + SW))

    #     X, _ = torch.solve(torch.t(self.W_R),(torch.eye(self.state_dim, dtype=float_type).cuda() + 2 * SW) )

    #     r2 =  torch.exp(-(state_error) @ torch.t(X) @ torch.t(state_error)) / \
    #             torch.sqrt(torch.det(torch.eye(self.state_dim, dtype=float_type).cuda() + 2 * SW))

    #     # sR = r2 - muR @ muR
    #     return muR.reshape((1,1)) #, sR.reshape((1,1))

    def plotReward(self, file_path):
        # calc normalized 2Dmap
        Xgd_2Dnormalized, Vgd_2Dnormalized = make_2DnormalizedGrid(
            self.x_lb, self.x_ub, n_x=30
        )
        rewards = self.getReward(Xgd_2Dnormalized.ravel())
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
        if self.verbose > 1:
            print("reward plot saved...")

    def plot_policy(self, model=None, iter=1):  # this is a torch.nn policy
        if model is None:
            model = self.controller
        n_x = 100
        # calc normalized 2Dmap
        (
            stacked_inputs,
            Xgd_2Dnormalized,
            Vgd_2Dnormalized,
        ) = self.get_grid_stacked_inputs(
            n_x, n_x
        )  # actions = self.linearModel(stacked_inputs) #debugging
        actions = model.predict(stacked_inputs)
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
        plt.savefig(f"{self.optimLogPath}policy_plot_{iter}.png")
        if self.verbose > 0:
            print("policy plot saved...")

    def get_grid_stacked_inputs(self, n_x=100, n_y=20):
        Xgd_2Dnormalized, Vgd_2Dnormalized = make_2DnormalizedGrid(
            self.x_lb, self.x_ub, n_x=n_x, n_y=n_y
        )
        stacked_inputs = np.stack(
            (Xgd_2Dnormalized.ravel(), Vgd_2Dnormalized.ravel())
        ).T
        stacked_inputs = self.tensor(stacked_inputs)
        return stacked_inputs, Xgd_2Dnormalized, Vgd_2Dnormalized

    # generate initial states, does not generate if Deterministic
    def generate_init_states(self):
        if (
            not self.isDeterministic_init
        ):  # in deterministic, default values are already in config
            # initial state distribution
            if self.initialDistr == "full":  # non-colliding with obstacle
                init_states = np.random.uniform(self.x_lb[:2], self.x_ub[:2])

            elif self.initialDistr == "constrained":
                init_states = np.random.uniform(
                    self.x_lb + self.angle_goal_offset,
                    self.x_ub - self.angle_goal_offset,
                )

            elif self.initialDistr == "grid":
                return (
                    self.Rinit,
                    self.Sinit,
                )  # pass and let the MC plotter handle initial states
            else:
                raise NotImplementedError()
            return self.tensor(init_states)
        else:
            return self.tensor(self.init_states)

    def tensor(self, data):
        return torch.tensor(data, dtype=self.dtype, device=self.device)

    def generate_goal(self):
        if (
            not self.isDeterministic_goal
        ):  # in deterministic, default values are already in config
            if self.goalDistr == "full":
                goal_state = np.random.uniform(self.x_lb, self.x_ub)

            elif self.goalDistr == "constrained-safe":  # safe means far from bounds
                goal_state = np.random.uniform(
                    self.x_lb + self.angle_goal_offset,
                    self.x_ub - self.angle_goal_offset,
                )
            else:
                raise NotImplementedError()
            return goal_state
        else:
            return self.target_normalized

    def test_GP_model(self):
        # quantify how good it is agaist test data
        if self.verbose > 2:
            print("Starting batch querying GP with test data")
        t1_GPQueryingBatch = time.time()
        Xt = self.Xtest
        Yt = self.Ytest
        y_pred = self.predict(Xt)
        t2_GPQueryingBatch = time.time()
        elapsed_GPQueryingBatch = t2_GPQueryingBatch - t1_GPQueryingBatch
        if self.verbose > 1:
            print(
                f"GP models queried in {elapsed_GPQueryingBatch:.3f} seconds, with {len(Xt)} data points"
            )
        # calculate MSE
        y_actual = Yt.numpy()

        def calcMSE(x1, x2):
            return sum(np.sqrt(x1**2 + x2**2))

        MSE_1 = calcMSE(y_pred[:, 0], y_actual[:, 0])
        MSE_2 = calcMSE(y_pred[:, 1], y_actual[:, 1])
        print(f"MSE_1= {MSE_1:.2f}, MSE_2= {MSE_2:.4f}")

    def analyze_data(self):
        # plot the training data

        def plot_data(t, x, xlbl, ylbl, title, filepath):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.scatter(t, x)
            ax.set_xlabel(xlbl)
            ax.set_ylabel(ylbl)
            ax.set_title(title)
            plt.savefig(f"{filepath}.png")

        plot_data(
            self.X.cpu().detach().numpy()[:, -1],
            self.Y.cpu().detach().numpy()[:, 0],
            xlbl="u-input",
            ylbl="state difference x",
            title="state difference x",
            filepath="./results/dx_plot",
        )
        plot_data(
            self.X.cpu().detach().numpy()[:, -1],
            self.Y.cpu().detach().numpy()[:, 1],
            xlbl="u-input",
            ylbl="state difference v",
            title="state difference v",
            filepath="./results/dv_plot",
        )

    def calc_realizations(self):
        N_traj = self.N_trajs  # number of (parallel) trajectories
        # initialize big tensor, keeping track of all variables
        M = self.tensor(
            torch.zeros((N_traj, self.state_dim + self.control_dim, self.Horizon))
        )
        # initial state+u and concat
        state = torch.cat(N_traj * [self.obs_torch[None, :]], dim=0)
        u = self.controller.forward(state)
        self.randTensor = self.tensor(torch.randn((self.N_trajs, 2, self.Horizon)))

        # assign data to M, memory := sys.getsizeof(M.storage())
        M[:, :-1, 0] = state  # M[:,:,k] := torch.cat( (state, u), dim = 1)
        if self.verbose > 0:
            print("starting trajectory realization...")
            t1 = time.time()
        for k in range(self.Horizon - 1):
            # get u:  state -> controller -> u
            M[:, -1, k] = self.controller.forward(M[:, :-1, k])[:, 0]
            # predict next state: s_{t+1} = GP(s,u)
            with torch.no_grad():
                predictions = self.predict(M[:, :, k])

            # px = torch.distributions.Normal(predictions.mean, predictions.stddev)
            # randtensor = px.sample()
            # randtensor = predictions.mean + predictions.stddev*self.tensor(torch.rand((N_traj, self.state_dim)))
            randtensor = (
                predictions.mean + predictions.stddev * self.randTensor[:, :, k]
            )
            # predict next state
            M[:, :-1, k + 1] = M[:, :-1, k] + randtensor

        if self.verbose > 0:
            t_elapsed = time.time() - t1
            print(f"predictions completed... elapsed time: {t_elapsed:.2f}")
            t1 = time.time()
        print(f"size of randomTensor is {randtensor.shape}")
        return M

    def calc_realizations_ref(self, reference_signal):  # with reference signal
        reference_signal.reference = reference_signal.reference * np.pi / 180
        initial_state = reference_signal.reference.iloc[0]
        # initialize big tensor, keeping track of all variables
        #############
        N_traj = self.N_trajs  # number of (parallel) trajectories

        init_state_normalized = np.divide(
            np.array([initial_state, 0]) - self.mean_states, self.std_states
        )
        self.Horizon = len(reference_signal)
        M = self.tensor(
            torch.zeros((N_traj, self.state_dim + self.control_dim, self.Horizon))
        )
        self.obs_torch = self.tensor(torch.tensor(init_state_normalized))
        ################
        # initial state+u and concat
        state = torch.cat(N_traj * [self.obs_torch[None, :]], dim=0)
        u = self.controller.forward(state)
        self.randTensor = self.tensor(torch.randn((self.N_trajs, 2, self.Horizon)))
        self.Horizon - 1
        # assign data to M, memory := sys.getsizeof(M.storage())
        M[:, :-1, 0] = state  # M[:,:,k] := torch.cat( (state, u), dim = 1)
        ref_normalized = np.divide(
            reference_signal.reference.to_numpy() - self.mean_states[0],
            self.std_states[0],
        )

        ref_torch = self.tensor(
            np.concatenate(
                (ref_normalized[None, :], ref_normalized[None, :] * 0), axis=0
            ).T
        )
        if self.verbose > 0:
            print("starting trajectory realization...")
            t1 = time.time()
        for k in range(self.Horizon - 1):
            # get u:  state -> controller -> u

            M[:, -1, k] = self.controller.forward(M[:, :-1, k] - ref_torch[k])[:, 0]
            # predict next state: s_{t+1} = GP(s,u)
            with torch.no_grad():
                predictions = self.predict(M[:, :, k])

            # px = torch.distributions.Normal(predictions.mean, predictions.stddev)
            # randtensor = px.sample()
            # randtensor = predictions.mean + predictions.stddev*self.tensor(torch.rand((N_traj, self.state_dim)))
            randtensor = (
                predictions.mean + predictions.stddev * self.randTensor[:, :, k]
            )
            # predict next state
            M[:, :-1, k + 1] = M[:, :-1, k] + randtensor

        if self.verbose > 0:
            t_elapsed = time.time() - t1
            print(f"predictions completed... elapsed time: {t_elapsed:.2f}")
            t1 = time.time()
        print(f"size of randomTensor is {randtensor.shape}")
        return M

    # calculate predictions+reward without having everything in one big tensor
    def optstep_loss(self):
        N_traj = self.N_trajs  # number of (parallel) trajectories
        # initialize big tensor, keeping track of all variables
        # M = self.tensor(torch.zeros((N_traj, self.state_dim+self.control_dim, Horizon)))
        # initial state+u and concat
        state = torch.cat(N_traj * [self.obs_torch[None, :]], dim=0)
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

    # def optstep_loss_seq(self): #calculate predictions+reward without having everything in one big tensor
    #     N_traj = self.N_trajs # number of (parallel) trajectories
    #     #initialize big tensor, keeping track of all variables
    #     # M = self.tensor(torch.zeros((N_traj, self.state_dim+self.control_dim, Horizon)))
    #     #initial state+u and concat
    #     state = torch.cat(N_traj*[self.obs_torch[None,:]],dim=0)
    #     u = self.controller.forward(state)
    #     rew = self.rew_exp_torchBatch_all(state[:,0])
    #     #assign data to M, memory := sys.getsizeof(M.storage())
    #     # M[:,:-1,0] = state    #M[:,:,k] := torch.cat( (state, u), dim = 1)
    #     if self.verbose > 1:
    #         print('starting trajectory realization...')
    #         t1 = time.time()
    #     for k in range(self.Horizon-1):
    #         #get u:  state -> controller -> u
    #         u = self.controller.forward(state)[:,0]
    #         #predict next state: s_{t+1} = GP(s,u)
    #         GPinput = torch.cat( (state, u[:,None]), dim=1)
    #         with   gpytorch.settings.fast_pred_var():  #torch.no_grad(),
    #             observed_pred = self.likelihood(self.model(X_))
    #         predictions = self.predict_seq(GPinput)
    #         # px = torch.distributions.Normal(predictions.mean, predictions.stddev) ##BAD..
    #         # use torch.normal(predictions.mean, predictions.stddev) or as below
    #         # predict next state
    #         next_state = state + predictions.mean + torch.mul(predictions.stddev, self.randTensor[:,:,k])
    #         # get reward
    #         rewards = self.rew_exp_torchBatch_all(next_state[:,0])
    #         rew = rew + rewards
    #         state = next_state
    #     if self.verbose > 1:
    #         t_elapsed = time.time() - t1
    #         print(f'predictions completed... elapsed time: {t_elapsed:.2f}')
    #         t1 = time.time()
    #     return rew

    def calc_realization_mean(self):  # mean of GP predictions, no sampling
        N_traj = 1  # number of (parallel) trajectories
        # initialize big tensor, keeping track of all variables
        M = self.tensor(
            torch.zeros((N_traj, self.state_dim + self.control_dim, self.Horizon))
        )
        print(f"tensor M memory usage: {sys.getsizeof(M.storage())/(1048576*8)} MB")
        # initial state+u and concat
        state = torch.cat(N_traj * [self.obs_torch[None, :]], dim=0)
        u = self.controller.forward(state)

        # assign data to M, memory := sys.getsizeof(M.storage())
        M[:, :-1, 0] = state  # M[:,:,k] := torch.cat( (state, u), dim = 1)
        if self.verbose > 0:
            print("starting trajectory realization...")
            t1 = time.time()
        for k in range(self.Horizon - 1):
            # get u:  state -> controller -> u
            M[:, -1, k] = self.controller.forward(M[:, :-1, k])[:, 0]
            # predict next state: s_{t+1} = GP(s,u)
            GP_in = M[:, :, k]
            GP_in[:, 1] = GP_in[:, 1] * 0
            predictions = self.predict(GP_in)
            px = predictions.mean
            # predict next state
            M[:, :-1, k + 1] = M[:, :-1, k] + px
        if self.verbose > 0:
            t_elapsed = time.time() - t1
            print(f"predictions completed... elapsed time: {t_elapsed:.2f}")
            t1 = time.time()
        return M

    def MC_oneSweep(self):

        M_mean = self.calc_realization_mean().cpu().detach().numpy()
        M_trajs = self.calc_realizations().cpu().detach().numpy()
        self.plot_MC(M_mean, M_trajs, "./results/MC/oneSweep", savefig=True)

    def plot_MC(self, M_mean, M_trajs, saveDir=None, savefig=False):
        plt.rc("font", family="serif")

        fig_p, ax_p = plt.subplots(1, 1)
        fig2D, ax2D = plt.subplots(1, 1)
        # fig2, ax2 = plt.subplots(2,1)
        # fig3, ax3 = plt.subplots(2,1)
        fig4, ax4 = plt.subplots(1, 1)  # (, figsize=(10,15)  )
        # fig5, ax5 = plt.subplots(4,1) #  (, figsize=(10,15)  )
        # fig6, ax6 = plt.subplots(6,1, figsize=(10,15)) #  (, figsize=(10,15)  )
        fig_p.set_size_inches((7, 6))

        timeVec = np.arange(0, self.dt * (M_trajs.shape[2]), self.dt)
        # one-time plots
        targetlineopt = "g--"
        targetlinewidth = 1.6
        targetlinealpha = 0.9
        # rad2deg = 180 / np.pi
        # virtual 'z' for each line layer (overlapping lines)
        zo1 = 20
        # target R
        ax_p.plot(
            self.timeVec,
            np.repeat(self.target_normalized[0], len(self.timeVec)),
            targetlineopt,
            linewidth=targetlinewidth,
            alpha=targetlinealpha,
            zorder=zo1,
        )
        # obstacle point on X,Y plots
        # ax_p[2].plot(obstacle_time, self.obstacle_pos[0], 'ro', markersize=11, label = 'obstacle-x', zorder = zo1)
        # ax_p[3].plot(obstacle_time, self.obstacle_pos[1], 'ro', markersize=11, label = 'obstacle-y', zorder = zo1)

        ax2D.plot(
            self.target_normalized[0],
            self.target_normalized[1],
            "o",
            markersize=13,
            label="goal",
            markeredgewidth=1,
            markeredgecolor="r",
            markerfacecolor="None",
            zorder=zo1,
        )

        # plot 2D

        # plot operation bounds

        x_bounds = [
            self.x_lb[0],
            self.x_ub[0],
            self.x_ub[0],
            self.x_lb[0],
            self.x_lb[0],
        ]
        v_bounds = [
            self.x_lb[1],
            self.x_lb[1],
            self.x_ub[1],
            self.x_ub[1],
            self.x_lb[1],
        ]

        ax2D.plot(x_bounds, v_bounds, "-", color="black")

        # Monte Carlo trajectories
        total_realizations = M_trajs.shape[0] + 1  # + mean trajectory
        for k in range(total_realizations):
            if k == total_realizations - 1:  # last entry
                lineopt = "r-"  # red line (nominal)
                zo = 20  # zorder (top plot layer)
                default_alpha = 0.9
                x_data = M_mean[0, 0, :]
                v_data = M_mean[0, 1, :]
                u_data = M_mean[0, -1, :]
            # elif k == 10:
            #     lineopt = 'r-' #red line (worst case)
            #     lineopt = 'c-' # let's overwrite!
            #     zo = 5 #zorder (mid plot layer)
            else:
                x_data = M_trajs[k, 0, :]
                v_data = M_trajs[k, 1, :]
                u_data = M_trajs[k, -1, :]
                lineopt = "b-"  # blue line (Monte Carlo M_trajs)
                zo = 0  # zorder (lowest plot layer)
                default_alpha = 0.01

            # plot pos, vel
            ax_p.plot(timeVec, x_data, lineopt, alpha=default_alpha, zorder=zo)
            # ax_p.title('Trolley R,V')
            # ax_p.xlabel('Time (s)')
            # ax_p[1].plot(timeVec, v_data, lineopt, alpha=default_alpha, zorder = zo)

            ax2D.plot(
                x_data[0],
                v_data[0],
                "bx",
                markersize=7,
                label="start-pos",
                alpha=default_alpha,
                zorder=zo1,
            )
            ax2D.plot(x_data, v_data, lineopt, alpha=default_alpha, zorder=zo)

            # control inputs/
            # actions
            ax4.plot(timeVec, u_data, lineopt, alpha=default_alpha, zorder=zo)

            # rewards
            # ax5[3].plot(timeVec, self.reward, lineopt, zorder = zo)
        plot_max_x = 1.2 * self.x_ub[0]
        ax_p.set_ylabel("Position Normalized")
        ax_p.set_ylim((-plot_max_x, plot_max_x))
        # ax_p[1].set_ylabel('Velocity Normalized')
        # ax_p[1].set_xlabel('Time (s)')
        # ax_p[2].set_ylabel('X (m)')
        ax_p.set_xlim((0, timeVec[-1]))
        # ax_p[1].set_xlim((0, timeVec[-1]))
        # # ax_p[2].legend()
        # ax_p[3].set_ylabel('Y (m)')
        # ax_p[3].set_xlabel('Time (s)')
        # ax_p[3].legend()
        fig2D.set_size_inches((6, 6))
        ax2D.set_ylabel("V")
        ax2D.set_xlabel("X")
        # ax2D.legend()
        ax2D.set_aspect("equal", adjustable="box")

        ax2D.set_ylim((-plot_max_x, plot_max_x))
        ax2D.set_xlim((-plot_max_x, plot_max_x))

        ax4.set_ylabel("control input")

        # ax5[3].set_ylabel('rew_total')

        ax2D.grid(linestyle="--")
        for Q in [ax_p, ax4]:
            Q.grid(linestyle="--")
            Q.set_xlim((0, timeVec[-1]))

        if savefig:
            # save figures but do not show
            os.makedirs(saveDir, exist_ok=True)
            fig_p.savefig(saveDir + "_fig_1RV.png")
            fig2D.savefig(saveDir + "_fig_2D.png")
            # fig2.savefig(saveDir + '_fig_2slew.png')
            # fig3.savefig(saveDir + '_fig_3swing.png')
            fig4.savefig(saveDir + "_fig_4control.png")
            # fig5.savefig(saveDir + '_fig_5rew.png')
            # fig6.savefig(saveDir + '_fig_6obs.png')

            # save figures pdf
            fig_p.savefig(saveDir + "_fig_1RV.pdf")
            fig2D.savefig(saveDir + "_fig_2D.pdf")
            # fig2.savefig(saveDir + '_fig_2slew.pdf')
            # fig3.savefig(saveDir + '_fig_3swing.pdf')
            fig4.savefig(saveDir + "_fig_4control.pdf")
            # fig5.savefig(saveDir + '_fig_5rew.pdf')
            # fig6.savefig(saveDir + '_fig_6obs.pdf')

        else:
            plt.show()

    # for SGD directly on policy parameters

    def optimize_policy(self):
        """
        Optimize controller's parameter's
        """
        maxiter = self.configs["optimopts"]["max_iter"]
        trials = self.configs["optimopts"]["trials"]
        allOptimData = {"allOptimData": []}
        os.makedirs(self.optimLogPath, exist_ok=True)
        for tl in range(trials):
            # optimization algorithm:  self.configs['optimopts']['optimizer']
            # self.controller.randomize()
            if self.verbose > 0:
                print(f"starting optimization trial {tl}...")
                # print(f'resetting controller parameters...')
            self.createController()
            self.randTensor = self.tensor(torch.randn((self.N_trajs, 2, self.Horizon)))
            controller_initial = deepcopy(self.controller)

            optim_method = self.configs["optimopts"]["optimizer"]
            if optim_method == "RMSprop":
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
            elif optim_method == "Adam":
                self.optimizer = torch.optim.Adam(
                    [
                        {"params": self.controller.parameters()},
                    ],
                    lr=0.05,
                )
            elif optim_method == "SGD":
                self.optimizer = torch.optim.SGD(self.controller.parameters(), lr=0.05)
            else:
                raise NotImplementedError

            # set fixed randnumber
            t_start = time.time()
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
                if self.verbose > 0:
                    # - params: {self.controller[0].weight}, {self.controller[0].bias.cpu().detach.numpy()}
                    print(
                        f"Optimize Policy: Iter {i}/{maxiter} - Loss: {loss.item():.3f}"
                    )
                self.optimizer.step()
                # keep track of runtime and loss
                t2 = time.time() - t_start
                optimInfo["loss"].append(loss.item())
                optimInfo["time"].append(t2)

            print(
                "Controller's optimization: done in %.1f seconds with reward=%.3f."
                % (t2, loss.item())
            )
            saveInfo_trial = {
                "optimInfo": optimInfo,
                "controller_initial": controller_initial,
                "controller_final": deepcopy(self.controller),
                "mean_states": self.mean_states,
                "std_states": self.std_states,
            }
            save_data(f"{self.optimLogPath}_trial_{tl}.pkl", saveInfo_trial)
            # collect trial data into one
            allOptimData["allOptimData"].append(saveInfo_trial)
        save_data(f"{self.optimLogPath}_all.pkl", allOptimData)

    def optimize_policy_sequentially(self):  # for testing
        """
        Optimize controller's parameter's
        """
        realBatchSize = self.N_trajs
        self.N_trajs = 1
        maxiter = self.configs["optimopts"]["max_iter"]
        trials = self.configs["optimopts"]["trials"]
        allOptimData = {"allOptimData": []}
        os.makedirs(self.optimLogPath, exist_ok=True)
        for tl in range(trials):
            # optimization algorithm:  self.configs['optimopts']['optimizer']
            # self.controller.randomize()
            if self.verbose > 0:
                print(f"starting optimization trial {tl}...")
                # print(f'resetting controller parameters...')
            self.createController()
            self.randTensor = self.tensor(torch.randn((self.N_trajs, 2, self.Horizon)))
            controller_initial = deepcopy(self.controller)

            optim_method = self.configs["optimopts"]["optimizer"]
            if optim_method == "RMSprop":
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
            elif optim_method == "Adam":
                self.optimizer = torch.optim.Adam(
                    [
                        {"params": self.controller.parameters()},
                    ],
                    lr=0.05,
                )
            elif optim_method == "SGD":
                self.optimizer = torch.optim.SGD(self.controller.parameters(), lr=0.05)
            else:
                raise NotImplementedError

            # set fixed randnumber
            t_start = time.time()
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
                if self.verbose > 0:
                    # - params: {self.controller[0].weight}, {self.controller[0].bias.cpu().detach.numpy()}
                    print(
                        f"Optimize Policy: Iter {i}/{maxiter} - Loss: {loss.item():.3f}"
                    )
                self.optimizer.step()
                # keep track of runtime and loss
                t2 = time.time() - t_start
                # small tweak of variables to get the real batch loss (to compare optimization)
                self.N_trajs = realBatchSize
                with torch.no_grad():
                    lossBATCH = -self.optstep_loss()
                self.N_trajs = 1
                optimInfo["loss"].append(lossBATCH.item())
                optimInfo["time"].append(t2)

            print(
                "Controller's optimization: done in %.1f seconds with reward=%.3f."
                % (t2, lossBATCH.item())
            )
            saveInfo_trial = {
                "optimInfo": optimInfo,
                "controller_initial": controller_initial,
                "controller_final": deepcopy(self.controller),
                "mean_states": self.mean_states,
                "std_states": self.std_states,
            }
            save_data(f"{self.optimLogPath}_trial_{tl}.pkl", saveInfo_trial)
            # collect trial data into one
            allOptimData["allOptimData"].append(saveInfo_trial)
        save_data(f"{self.optimLogPath}_all.pkl", allOptimData)

    def opt_predict(self, Horizon=100):
        N_traj = self.Horizon  # number of (parallel) trajectories
        # initialize big tensor, keeping track of all variables
        M = self.tensor(
            torch.zeros((N_traj, self.state_dim + self.control_dim, Horizon))
        )
        # initial state+u and concat
        state = torch.cat(N_traj * [self.obs_torch[None, :]], dim=0)
        u = self.controller.forward(state)

        # assign data to M, memory := sys.getsizeof(M.storage())
        M[:, :-1, 0] = state  # M[:,:,k] := torch.cat( (state, u), dim = 1)
        if self.verbose > 1:
            print("starting trajectory realization...")
            t1 = time.time()
        for k in range(Horizon - 1):
            # get u:  state -> controller -> u
            M[:, -1, k] = self.controller.forward(M[:, :-1, k])[:, 0]
            # predict next state: s_{t+1} = GP(s,u)
            predictions = self.predict(M[:, :, k])
            px = torch.distributions.Normal(predictions.mean, predictions.stddev)
            # predict next state
            M[:, :-1, k + 1] = M[:, :-1, k] + px.sample()
        if self.verbose > 1:
            t_elapsed = time.time() - t1
            print(f"predictions completed... elapsed time: {t_elapsed:.2f}")
            t1 = time.time()
        # sum up all rewards
        rew = self.rew_exp_torchBatch_all(M[:, 0, :])
        return rew

    def opt_predict_old(self, Horizon=100):
        N_traj = self.Horizon  # number of (parallel) trajectories
        # initialize big tensor, keeping track of all variables
        M = self.tensor(
            torch.zeros((N_traj, self.state_dim + self.control_dim, Horizon))
        )
        # initial state+u and concat
        state = torch.cat(N_traj * [self.obs_torch[None, :]], dim=0)
        u = self.controller.forward(state)

        # assign data to M, memory := sys.getsizeof(M.storage())
        M[:, :-1, 0] = state  # M[:,:,k] := torch.cat( (state, u), dim = 1)
        if self.verbose > 1:
            print("starting trajectory realization...")
            t1 = time.time()
        for k in range(Horizon - 1):
            # get u:  state -> controller -> u
            M[:, -1, k] = self.controller.forward(M[:, :-1, k])[:, 0]
            # predict next state: s_{t+1} = GP(s,u)
            predictions = self.predict(M[:, :, k])
            px = torch.distributions.Normal(predictions.mean, predictions.stddev)
            # predict next state
            M[:, :-1, k + 1] = M[:, :-1, k] + px.sample()
        if self.verbose > 1:
            t_elapsed = time.time() - t1
            print(f"predictions completed... elapsed time: {t_elapsed:.2f}")
            t1 = time.time()
        # sum up all rewards
        rew = self.rew_exp_torchBatch_all(M[:, 0, :])
        return rew


# GP Classes PYTORCH


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()
    ):
        # The Gaussian Likelihood  assumes a homoskedastic noise model (i.e. all inputs have the same observational noise).
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        )

    def forward(self, x):
        """the forward method takes in some n×d data x and returns a MultivariateNormal
        with the prior mean and covariance evaluated at x. In other words, we return the
        vector μ(x) and the n×n matrix Kxx representing the prior mean and covariance matrix
        of the GP"""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


"""train() mode is for optimizing model hyperameters. - .eval() mode is for computing predictions through the model posterior."""


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2]), ard_num_dims=3),
            batch_shape=torch.Size([2]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
