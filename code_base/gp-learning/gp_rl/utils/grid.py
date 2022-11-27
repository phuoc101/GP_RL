import numpy as np


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
