import numpy as np
import time
import torch
from loguru import logger

from gp_rl.utils.torch_utils import get_tensor

DEFAULT_DEVICE = torch.device("cuda:0")
DEFAULT_DTYPE = torch.float32


def generate_init_state(
    is_det,
    n_trajs,
    initial_distr=None,
    x_lb=None,
    x_ub=None,
    state_dim=None,
    default_init_state=None,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    # in deterministic, default values are already in config
    if not is_det:
        # initial state distribution
        if initial_distr == "full":  # non-colliding with obstacle
            init_state = get_tensor(
                np.random.uniform(
                    x_lb[0:state_dim],
                    x_ub[0:state_dim],
                    size=(n_trajs, state_dim),
                ),
                device=device,
                dtype=dtype,
            )
        else:
            raise NotImplementedError()
        return get_tensor(data=init_state, device=device, dtype=dtype)
    else:
        return get_tensor(
            data=np.array(n_trajs * [default_init_state]),
            device=device,
            dtype=dtype,
        )


def generate_goal(
    is_det,
    goal_distr=None,
    x_lb=None,
    x_ub=None,
    state_dim=None,
    default_target_state=None,
):
    # in deterministic, default values are already in config
    if not is_det:
        if goal_distr == "full":
            goal_state = np.random.uniform(
                x_lb[0:state_dim],
                x_ub[0:state_dim],
                size=state_dim,
            )
        else:
            raise NotImplementedError()
        return goal_state
    else:
        return default_target_state


def calc_realizations(
    gp_model,
    controller,
    n_trajectories,
    state_dim,
    control_dim,
    dt,
    init_state,
    target_state,
    tf=None,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Calculate realizations for multiple trajectories using prediction uncertainty
    from GP prediction

    Args:
        gp_model (type: GPModel): The GP Model
        controller (type: Controller): The controller
        n_trajectories : number of trajectories to calculate
        state_dim : number of observable states
        control_dim : number of controllable inputs
        dt : sampling time
        init_state : initial states
        target_state : goal state to reach
        tf : time to run the simulation

    Returns:
        realized trajectories, with GP uncertainty
    """
    # initialize big tensor, keeping track of all variables
    if tf is None:
        horizon = 1
    else:
        horizon = round(tf / dt)
    M = get_tensor(
        torch.zeros((n_trajectories, state_dim + control_dim, horizon + 1)),
        device=device,
        dtype=dtype,
    )
    # initial state+u and concat
    # state = deepcopy(obs_torch)
    state = generate_init_state(
        is_det=True, n_trajs=n_trajectories, default_init_state=init_state
    )
    # assign data to M, memory := sys.getsizeof(M.storage())
    M[:, :-1, 0] = state  # M[:,:,k] := torch.cat( (state, u), dim = 1)
    logger.debug("Starting Monte Carlo trajectory realization...")
    t1 = time.perf_counter()
    # convert target state to tensor if not alr is
    if not isinstance(target_state, torch.Tensor):
        target_state = get_tensor(
            data=generate_goal(is_det=True, default_target_state=target_state),
            device=device,
            dtype=dtype,
        )
    rand_tensor = get_tensor(
        data=torch.randn((n_trajectories, state_dim, horizon + 1)),
        device=device,
        dtype=dtype,
    )

    for k in range(1, horizon + 1):
        with torch.no_grad():
            # get u:  state -> controller -> u
            M[:, -1, k - 1] = controller(M[:, :-1, k - 1] - target_state)[:, 0]
            # predict next state: s_{t+1} = GP(s,u)
            predictions = gp_model.predict(M[:, :, k - 1])
        randtensor = predictions.mean + predictions.stddev * rand_tensor[:, :, k]
        # predict next state
        M[:, :-1, k] = M[:, :-1, k - 1] + randtensor

    t_elapsed = time.perf_counter() - t1
    logger.debug(f"predictions completed... elapsed time: {t_elapsed:.2f}s")
    # logger.debug(f"size of randomTensor is {randtensor.shape}")
    return M


def calc_realization_mean(
    gp_model,
    controller,
    state_dim,
    control_dim,
    dt,
    init_state,
    target_state,
    tf=None,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Calculate mean realizations from GP prediction

    Args:
        gp_model (type: GPModel): The GP Model
        controller (type: Controller): The controller
        state_dim : number of observable states
        control_dim : number of controllable inputs
        dt : sampling time
        init_state : initial states
        target_state : goal state to reach
        tf : time to run the simulation

    Returns:
        mean realized trajectories
    """
    # mean of GP predictions, no sampling
    if tf is None:
        horizon = 1
    else:
        horizon = round(tf / dt)
    n_trajectories = 1  # number of (parallel) trajectories
    # initialize big tensor, keeping track of all variables
    M = get_tensor(
        data=torch.zeros((n_trajectories, state_dim + control_dim, horizon + 1)),
        device=device,
        dtype=dtype,
    )
    state = generate_init_state(
        is_det=True, n_trajs=n_trajectories, default_init_state=init_state
    )

    # assign data to M, memory := sys.getsizeof(M.storage())
    M[:, :-1, 0] = state  # M[:,:,k] := torch.cat( (state, u), dim = 1)
    # convert target state to tensor if not alr is
    if not isinstance(target_state, torch.Tensor):
        target_state = get_tensor(
            data=generate_goal(is_det=True, default_target_state=target_state),
            device=device,
            dtype=dtype,
        )
    logger.debug("Starting mean trajectory realization...")
    t1 = time.perf_counter()
    for k in range(1, horizon + 1):
        # get u:  state -> controller -> u
        with torch.no_grad():
            M[:, -1, k - 1] = controller(M[:, :-1, k - 1] - target_state)[:, 0]
            # predict next state: s_{t+1} = GP(s,u)
            predictions = gp_model.predict(M[:, :, k - 1])
        px = predictions.mean
        # predict next state
        M[:, :-1, k] = M[:, :-1, k - 1] + px
    t_elapsed = time.perf_counter() - t1
    logger.debug(f"predictions completed... elapsed time: {t_elapsed:.2f}s")
    return M


def calc_realizations_non_det_init(
    n_trajs_sim,
    gp_model,
    controller,
    state_dim,
    control_dim,
    x_lb,
    x_ub,
    dt,
    init_state,
    target_state,
    tf=None,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Calculate mean realized trajectories from multiple random starting points

    Args:
        n_trajs_sim : number of trajectories to simulate
        gp_model (type: GPModel): The GP model
        controller (type: Controller): The controller
        state_dim : Number of observable states
        control_dim : number of controllable inputs
        x_lb : Lower bound of observable states
        x_ub : Upper bound of observable states
        dt : sampling time
        init_state : initial states
        target_state : goal states to reach
        tf : time to run simulation for

    Returns:
        Realized trajectories from multiple random starting points
    """
    if tf is None:
        horizon = 1
    else:
        horizon = round(tf / dt)
    # initialize big tensor, keeping track of all variables
    M = get_tensor(
        torch.zeros((n_trajs_sim, state_dim + control_dim, horizon + 1)),
        device=device,
        dtype=dtype,
    )
    # initial state+u and concat
    state = generate_init_state(
        is_det=False,
        n_trajs=n_trajs_sim,
        default_init_state=init_state,
        x_lb=x_lb,
        x_ub=x_ub,
        state_dim=state_dim,
        initial_distr="full",
    )
    # assign data to M, memory := sys.getsizeof(M.storage())
    M[:, :-1, 0] = state  # M[:,:,k] := torch.cat( (state, u), dim = 1)
    # convert target state to tensor if not alr is
    if not isinstance(target_state, torch.Tensor):
        target_state = get_tensor(
            data=generate_goal(
                is_det=True,
                default_target_state=target_state,
                goal_distr="full",
                x_lb=x_lb,
                x_ub=x_ub,
                state_dim=state_dim,
            ),
            device=device,
            dtype=dtype,
        )
    logger.debug(
        "Starting trajectory realization with non-deterministic initialization..."
    )
    t1 = time.perf_counter()
    for k in range(1, horizon + 1):
        with torch.no_grad():
            # get u:  state -> controller -> u
            M[:, -1, k - 1] = controller(M[:, :-1, k - 1] - target_state)[:, 0]
            # predict next state: s_{t+1} = GP(s,u)
            predictions = gp_model.predict(M[:, :, k - 1])
        px = predictions.mean
        # predict next state
        M[:, :-1, k] = M[:, :-1, k - 1] + px

    t_elapsed = time.perf_counter() - t1
    logger.debug(f"predictions completed... elapsed time: {t_elapsed:.2f}s")
    return M
