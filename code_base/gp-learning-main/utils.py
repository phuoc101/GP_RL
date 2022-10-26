import pickle
import numpy as np

# from copy import copy
import torch
import gpytorch

# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import pandas as pd


def load_data(file):  # load data from Avant experiments
    with open(file, "rb") as f:
        return pickle.load(f)


def save_data(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)


def get_boom_angles(data):
    return data[:, 71]


# def get_boom_cmd(data):
#     tmp = np.array(data[:, 52], dtype='uint8')
#     boom_cmd = np.array(tmp.astype('int8'), dtype=float)
#     return boom_cmd / 90
#


def load_trainingData(dir_data, numInputs=3, output="torch", normalize_=True):
    # numInputs == 2 gives inputs as (pos, u)
    # numInputs == 3 gives inputs as (pos, vel, u)
    """call this to load *TrainingData.pkl that was saved in processing
    *normalizes data
    *converts to torch tensors
    *returns mean, std, lower and upper bounds on data"""
    training_data = load_data(f"{dir_data}")
    # get X, Y based on the number of inputs (to GP) that's desired
    if numInputs == 3:
        X = training_data["X1_xvu"]
        Y = np.vstack((training_data["Y1"], training_data["Y2"])).T
    else:
        X = training_data["X1_xu"]
        Y = training_data["Y1"]
    # normalize X, Y
    mean_states, std_states = getMS_XY(X)
    mean_states = 0 * mean_states  # for simplicity
    if normalize_:
        X, Y = normalize_XY_fromMS(X, Y, mean_states, std_states)
    else:
        mean_states = mean_states * 0
        std_states = std_states * 0 + 1
    xlb = np.min(X, 0)  # these are "Normalized" bounds
    xub = np.max(X, 0)
    # multiply inputs U
    X[:, -1] = X[:, -1] * 5

    if output == "torch":
        X = torch.Tensor(X).contiguous()
        Y = torch.Tensor(Y).contiguous()  # for some reason this is needed
    return X, Y, mean_states, std_states, xlb, xub


def get_predictions_GPytorch(model, likelihood, Xt):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(Xt))
        return observed_pred, observed_pred.mean.numpy(), observed_pred.stddev.numpy()


def getMS_XY(
    X1,
):  # normalize the data from its own mean, std - use for first collected data
    mean_states = np.mean(X1[:, :-1], 0)
    std_states = np.std(X1[:, :-1], 0)
    return mean_states, std_states


# normalize the data given mean and std, use for new data
def normalize_XY_fromMS(X1, Y1, mean_states, std_states):
    X = np.zeros(X1.shape)
    X[:, :-1] = np.divide(X1[:, :-1] - mean_states, std_states)
    X[:, -1] = X1[:, -1]  # control inputs are not normalised
    Y = np.divide(Y1, std_states)
    return X, Y


def make_2DnormalizedGrid(xlb, xub, n_x=30, n_y=30):
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


def load_data_numpy(file):
    p = load_data(file)
    t = np.array(p["time"])
    data = np.array(p["data"])
    u = np.array(p["u_control"])[:, 3]
    return t, data, u


def get_boom_cmd(data):
    tmp = np.array(data[:, 52], dtype="uint8")
    boom_cmd = np.array(tmp.astype("int8"), dtype=float)
    return boom_cmd / np.max(boom_cmd) * 0.2  # scale to action commands


def get_df_machine(filepath=None, Tf=None):
    """
    load the Avant run data and return a data frame having time, position of boom, reference signal (goal), controller output and actual input to the machine
    """
    if filepath is None:
        filepath = "./avant-control/ML_experiment_Pytorch2_test_3/rollout_0/_data.pkl"
    t, data, u = load_data_numpy(filepath)
    boom_ang = get_boom_angles(data)
    actual_cmd = get_boom_cmd(data)
    if Tf is not None:
        idx = t <= Tf
        t = t[idx]
        boom_ang = boom_ang[idx]
        actual_cmd = actual_cmd[idx]
        u = u[idx]

    df = pd.DataFrame(
        {
            "time": t,
            "boom_angle": boom_ang * 180 / np.pi,
            "u_cmd": u,
            "actual_cmd": actual_cmd,
        }
    )
    return df


def get_df_referenceSignal(init_state=None, experimentNo=1, Tf=None):
    if init_state is None:
        raise NotImplementedError
    # make reference signals based on experiment no.
    if experimentNo == 1:
        Tf = Tf if Tf is not None else 100
        schedule = {"t": np.array([5, 100, 200]), "goal": np.array([init_state, 0, 0])}
    else:
        Tf = Tf if Tf is not None else 150
        schedule = {
            "t": np.array([5, 20, 40, 60, 80, 100, 120, 140, 160, 400]),
            "goal": np.array([init_state, 0, -20, 10, 30, 0, 20, 0, -20, -20]),
        }
    t_s = np.arange(0, Tf, step=0.01)
    y_s = t_s * 0
    idx_goal = 0

    for k in range(len(t_s)):
        if t_s[k] >= schedule["t"][idx_goal]:
            idx_goal += 1
        y_s[k] = schedule["goal"][idx_goal]
    return pd.DataFrame({"time": t_s, "reference": y_s})
    # return t_s, y_s


def get_predictions_closedloop(GPmodel, initial_state, Tf):
    time_data = np.arange(0, Tf, step=0.05)
    GPmodel.Horizon = len(time_data)
    init_state_normalized = np.divide(
        np.array([initial_state, 0]) - GPmodel.mean_states, GPmodel.std_states
    )
    GPmodel.obs_torch = GPmodel.tensor(torch.tensor(init_state_normalized))
    M = GPmodel.calc_realizations().cpu().detach().numpy()
    # convert boom_angle, u back to original values
    M[:, 0, :] = M[:, 0, :] * GPmodel.std_states[0] * 180 / np.pi
    M[:, -1, :] = M[:, -1, :] * 0.2
    data = {"time": [], "boom_ang": [], "u": []}
    idx_data = []
    for j in range(M.shape[0]):  # loop over M_trajs
        t = time_data  # [j]
        b_ = M[j, 0, :]
        u_ = M[j, -1, :]
        idx_data = np.concatenate((np.array(idx_data), np.array(len(t) * [j])), axis=0)
        data["time"] = np.concatenate((np.array(data["time"]), t), axis=0)
        data["boom_ang"] = np.concatenate((np.array(data["boom_ang"]), b_), axis=0)
        data["u"] = np.concatenate((np.array(data["u"]), u_), axis=0)
    idx_data = np.array(idx_data, dtype=np.int16)
    data["trajectory"] = idx_data + 1

    df = pd.DataFrame(data)
    return df


def get_predictions_closedloop_ref(GPmodel, reference_signal, initial_state, Tf):
    def get_realizations_segment(reference_signal1):
        reference_signal1 = reference_signal1.iloc[::5, :].copy()
        M = GPmodel.calc_realizations_ref(reference_signal1).cpu().detach().numpy()
        # convert boom_angle, u back to original values
        M[:, 0, :] = M[:, 0, :] * GPmodel.std_states[0] * 180 / np.pi
        M[:, -1, :] = M[:, -1, :] * 0.2
        data = {"time": [], "boom_ang": [], "u": []}
        idx_data = []
        time_data = reference_signal1.time.to_numpy()

        for j in range(M.shape[0]):  # loop over M_trajs
            t = time_data  # [j]
            b_ = M[j, 0, :]
            u_ = M[j, -1, :]
            idx_data = np.concatenate(
                (np.array(idx_data), np.array(len(t) * [j])), axis=0
            )
            data["time"] = np.concatenate((np.array(data["time"]), t), axis=0)
            data["boom_ang"] = np.concatenate((np.array(data["boom_ang"]), b_), axis=0)
            data["u"] = np.concatenate((np.array(data["u"]), u_), axis=0)
        idx_data = np.array(idx_data, dtype=np.int16)
        data["trajectory"] = idx_data + 1
        return pd.DataFrame(data)
        # end def

    schedule = {
        "t": np.array([20, 40, 60, 80, 100, 120, 140, 160]),
        # normalized, from
        "goal": np.divide(
            np.array([0, -20, 10, 30, 0, 20, 0, -20]) * np.pi / 180
            - GPmodel.mean_states[0],
            GPmodel.std_states[0],
        ),
    }
    maindf = []
    idx = [p - 1 for p in [1, 500, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 14998]]
    for _counter in range(len(idx) - 1):
        ref1 = reference_signal.iloc[idx[_counter] : idx[_counter + 1]].copy()
        df1 = get_realizations_segment(ref1)
        df1["segment"] = _counter
        maindf.append(df1)
    df = pd.concat(maindf, ignore_index=True)
    return df


def load_optim_data(path_, divide_by=1):
    optim_info = load_data(path_)
    loss_data = []
    time_data = []
    for optimdata in optim_info["allOptimData"]:
        loss_data.append(optimdata["optimInfo"]["loss"])
        time_data.append(optimdata["optimInfo"]["time"])
    loss_data = np.array(loss_data)
    time_data = np.array(time_data)
    time_data -= time_data[:, 0, None]
    # choose a sampling time as average of all of them
    t_s = np.mean(time_data, 0)
    l_s = np.array([np.interp(t_s, t_, l_) for t_, l_ in zip(time_data, loss_data)])
    time_data = t_s
    loss_data = l_s
    data = {"time": [], "loss": []}
    idx_data = []
    for j in range(loss_data.shape[0]):
        t = time_data  # [j]
        l = loss_data[j]
        if np.all(l[-5:] > -5000):
            print("skipping trial #{j}")
            continue
        l = l / divide_by
        idx_data = np.concatenate((np.array(idx_data), np.array(len(t) * [j])), axis=0)
        data["time"] = np.concatenate((np.array(data["time"]), t), axis=0)
        data["loss"] = np.concatenate((np.array(data["loss"]), l), axis=0)
    idx_data = np.array(idx_data, dtype=np.int16)
    data["trial"] = idx_data
    df = pd.DataFrame(data=data)
    return df
