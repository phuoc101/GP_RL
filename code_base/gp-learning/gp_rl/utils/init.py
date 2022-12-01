import numpy as np

from utils.torch_utils import get_tensor


def generate_init_state(self, is_det, n_trajs):
    # in deterministic, default values are already in config
    if not is_det:
        # initial state distribution
        if self.initial_distr == "full":  # non-colliding with obstacle
            init_state = get_tensor(
                np.random.uniform(
                    self.x_lb[0 : self.state_dim],
                    self.x_ub[0 : self.state_dim],
                    size=(n_trajs, self.state_dim),
                ),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise NotImplementedError()
        return get_tensor(data=init_state, device=self.device, dtype=self.dtype)
    else:
        return get_tensor(
            data=np.array(n_trajs * [self.init_state]),
            device=self.device,
            dtype=self.dtype,
        )


def generate_goal(self, is_det):
    # in deterministic, default values are already in config
    if not is_det:
        if self.goal_distr == "full":
            goal_state = np.random.uniform(
                self.x_lb[0 : self.state_dim],
                self.x_ub[0 : self.state_dim],
                size=self.state_dim,
            )

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
