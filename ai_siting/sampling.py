"""Parameter sampling: central values, Monte Carlo draws and Sobol designs."""
import numpy as np
from scipy import stats
from . import config as C


def central(params=None):
    params = params or C.PARAMS
    out = {}
    for k, (dist, a, m, b, *_rest) in params.items():
        out[k] = m if dist == "tri" else 0.5 * (a + b)
    return out


def _ppf(dist, a, m, b, u):
    if dist == "tri":
        c = (m - a) / (b - a) if b > a else 0.5
        return stats.triang.ppf(u, c, loc=a, scale=b - a)
    return a + u * (b - a)


def monte_carlo(n, seed=C.SEED, params=None):
    """Latin hypercube sample of all parameters (dict of arrays, length n)."""
    params = params or C.PARAMS
    rng = np.random.default_rng(seed)
    keys = list(params)
    d = len(keys)
    # Latin hypercube: one stratified draw per interval, independently permuted per dimension
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n
    return {k: _ppf(params[k][0], params[k][1], params[k][2], params[k][3], u[:, i])
            for i, k in enumerate(keys)}


def sobol_problem(keys, params=None):
    params = params or C.PARAMS
    return {"num_vars": len(keys), "names": list(keys), "bounds": [[0.0, 1.0]] * len(keys)}


def from_unit(u_matrix, keys, params=None):
    """Map a unit-hypercube design (n x d) onto the parameter distributions."""
    params = params or C.PARAMS
    out = central(params)
    for i, k in enumerate(keys):
        dist, a, m, b = params[k][:4]
        out[k] = _ppf(dist, a, m, b, np.clip(u_matrix[:, i], 1e-9, 1 - 1e-9))
    return out


def with_overrides(params, overrides):
    """Return a copy of PARAMS with some entries replaced (used after calibration)."""
    new = dict(params)
    new.update(overrides)
    return new
