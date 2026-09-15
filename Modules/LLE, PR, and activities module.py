#################### Imports ####################
import numpy as np
from scipy.stats import linregress
from numba import njit


################### Activation functions and rate-network dynamics ####################

def sigmoid(x):
    """Logistic sigmoid activation: 1 / (1 + e^-x)."""
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    """Hyperbolic-tangent activation (thin NumPy wrapper, kept for API parity
    with `sigmoid` so either can be passed around as a plain function)."""
    return np.tanh(x)


@njit(fastmath=True)
def rate_network_numba(X, W, I, tau, activation=np.tanh):
    """
    Rate-network right-hand side: dX/dt = (-X + W @ activation(X) + I) / tau.

    Numba-jitted for speed. `activation` defaults to tanh but can be
    overridden with any elementwise-compatible function.

    Args:
        X: current state, shape (N,)
        W: recurrent weight matrix, shape (N, N)
        I: external input at this time step, shape (N,)
        tau: time constant
        activation: elementwise nonlinearity applied to X before the W @ ... term

    Returns:
        dX/dt, shape (N,)
    """
    return (-X + W @ activation(X) + I) / tau


def euler_rate_network(network, x0, t, W, I, tau, activation):
    """
    Generic forward-Euler integrator for a rate network.

    Works with ANY network update function that has the signature
    network(activation, x, W, I, tau) -> dx/dt, so it's the flexible/
    pure-Python counterpart to euler_rate_network_numba below (which is
    faster but hard-codes tanh as the nonlinearity).

    Args:
        network: callable implementing the dynamics, e.g. dx/dt = network(...)
        x0: initial state, shape (N,)
        t: time vector
        W: recurrent weight matrix
        I: external input, shape (len(t), N)
        tau: time constant
        activation: nonlinearity passed through to `network`

    Returns:
        x: trajectory of shape (len(t), len(x0))
    """
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    h = t[1] - t[0]  # assumes a uniform time step
    for i in range(1, len(t)):
        x[i] = x[i - 1] + h * network(activation, x[i - 1], W, I[i - 1], tau)
    return x


@njit(fastmath=True)
def euler_rate_network_numba(W, tau, dt, t_steps, x0, I):
    """
    Numba-compiled forward-Euler integration of the tanh rate network,
    returning the full state trajectory. This is the fast, hard-coded
    (tanh-only) counterpart to euler_rate_network() above.

    Args:
        W: recurrent weight matrix
        tau: time constant
        dt: integration time step
        t_steps: number of time steps to simulate
        x0: initial state, shape (N,)
        I: external input; either constant (shape (N,)) or time-varying
           (shape (t_steps, N))

    Returns:
        x: (t_steps, N) state trajectory
    """
    N = len(x0)
    x = np.zeros((t_steps, N), dtype=x0.dtype)
    x[0, :] = x0
    X = np.copy(x0)

    for i in range(1, t_steps):
        # Use the preceding input for the interval ending at time i.
        # Select time-varying vs. constant input for this step.
        if I.ndim == 2:
            Ii = I[i - 1, :]
        else:
            Ii = I

        X = X + dt * rate_network_numba(X, W, Ii, tau)
        x[i, :] = X

    return x  # always 2D: (t_steps, N)


##################### Weight-matrix generation ####################

def compute_weight_matrix(*args):
    """
    Build a sparse, randomly-connected recurrent weight matrix.

    Each row i has its Gaussian mean shifted along a line from -MSH to +MSH
    (a "mean-shift" / heterogeneity parameter), std = gain/sqrt(p*N) so that
    the spectral radius stays ~gain regardless of N or sparsity, then the
    matrix is thinned to connection probability p and self-connections
    are removed.

    Accepts either 5 positional args (p, N, MSH, gain, rng_weights), or 4
    (N, MSH, gain, rng_weights) in which case p defaults to 1.0 (fully
    connected) -- this preserves compatibility with call sites that predate
    the sparsity parameter.

    Args:
        p: connection probability (fraction of nonzero entries)
        N: number of neurons
        MSH: mean-shift half-range for the per-row Gaussian means
        gain: controls the overall coupling strength / spectral radius
        rng_weights: a numpy Generator (for reproducibility via seeding)

    Returns:
        W: (N, N) sparse random weight matrix with zero diagonal
    """
    if len(args) == 5:
        p, N, MSH, gain, rng_weights = args
    elif len(args) == 4:
        p = 1.0
        N, MSH, gain, rng_weights = args
    else:
        raise TypeError(
            "weight_matrix_fast expects either 4 or 5 positional arguments."
        )

    # Per-row target means for the incoming-weight distribution, evenly
    # spaced across [-MSH, MSH] -- this is what makes the matrix
    # heterogeneous rather than a single shared distribution.
    mus = np.linspace(-MSH, MSH, N)

    # Draw all weights at once; broadcasting assigns one mean per row.
    W = rng_weights.normal(
        loc=mus[:, None],
        scale=gain / np.sqrt(p * N),
        size=(N, N),
    )

    # Apply the connection mask only when the network is not fully connected.
    # rng_weights.random() draws from [0, 1), so this thresholding is what
    # introduces sparsity: if p == 1, every draw is < 1 and nothing is masked.
    if p < 1.0:
        mask = rng_weights.random(W.shape) < p
        W *= mask

    # Self-connections are excluded in both original implementations.
    np.fill_diagonal(W, 0)
    return W


##################### Statistical measures ####################

def participation_ratio(X):
    """
    Compute the participation ratio of activity data -- an estimate of the
    effective dimensionality of the population activity, based on the
    eigenvalue spectrum of its covariance matrix. PR = (sum(eig))^2 / sum(eig^2);
    it ranges from 1 (activity along a single dimension) up to N (activity
    spread evenly across all dimensions).

    Parameters
    ----------
    X : ndarray, shape (time points, neurons)
        Activity matrix.
    """
    # Center each neuron's activity trace on its own mean over time.
    X_centered = X - np.mean(X, axis=0)
    covariance = np.cov(X_centered.T)  # (neurons x neurons) covariance matrix
    eigenvalues = np.linalg.eigvalsh(covariance)  # eigvalsh: covariance is symmetric
    return np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)


##################### Lyapunov-exponent calculations ####################

def lyapunov_slopes(running_time, distances, reset_steps):
    """
    Estimate the local Lyapunov exponent as the average slope of
    log(distance) vs. time, computed piecewise over windows of length
    `reset_steps` (matching the intervals between perturbation resets).

    Args:
        running_time: 1D array of time points
        distances: 1D array of ||X2 - X1|| at each time point
        reset_steps: number of steps per fitting window

    Returns:
        Mean of the per-window slopes (the largest Lyapunov exponent estimate).
    """
    log_distances = np.log(distances)
    slopes = []

    for start in range(0, len(log_distances), reset_steps):
        stop = start + reset_steps
        if stop > len(log_distances):
            break  # drop the final partial window

        # Linear fit of log-distance vs. time within this window; its slope
        # is the local exponential growth/decay rate of the perturbation.
        slope = linregress(
            running_time[start:stop],
            log_distances[start:stop],
        ).slope
        slopes.append(slope)

    return np.mean(slopes)


@njit(fastmath=True)
def lyapunov_exponent_numba(running_time, reset_steps, tau, perturbation,
                            N, W, dt, X0):
    """
    Numba-compatible Lyapunov-exponent calculation (two-trajectory method).

    Simulates two trajectories (X1 from X0, and X2 = X0 + a small
    perturbation) forward under identical dynamics, tracking how far
    apart they drift. Every `reset_steps` steps the perturbed trajectory
    is rescaled back to the original perturbation magnitude (in the
    direction it had diverged), which keeps the two trajectories from
    fully separating while still capturing the local divergence rate.
    The local log-distance growth rate in each window is the local
    Lyapunov exponent; their mean estimates the largest Lyapunov exponent.

    Args:
        running_time: 1D array of timesteps
        reset_steps: number of steps between perturbation resets
        tau: network time constant
        perturbation: magnitude of the initial/reset perturbation
        N: number of neurons
        W: recurrent weight matrix (N, N)
        dt: integration time step
        X0: initial state, shape (N,)

    Returns:
        Estimated largest Lyapunov exponent (float).
    """
    steps = len(running_time)
    X1 = np.copy(X0)
    X2 = np.copy(X0 + perturbation / np.sqrt(N))

    distances = np.empty(steps, dtype=X0.dtype)
    distances[0] = np.linalg.norm(X2 - X1)
    I = np.zeros(N)  # no external input; dynamics are autonomous

    for t in range(1, steps):
        # Integrate both trajectories one Euler step under the same W, I.
        X1 = X1 + dt * rate_network_numba(X1, W, I, tau)
        X2 = X2 + dt * rate_network_numba(X2, W, I, tau)

        dist_vec = X2 - X1
        dist_mag = np.linalg.norm(dist_vec)
        distances[t] = dist_mag

        # Periodically rescale X2 back to the original perturbation distance
        # from X1, preserving the direction of divergence.
        if t % reset_steps == 0:
            X2 = X1 + perturbation * dist_vec / dist_mag

    # Numba does not support scipy.stats.linregress, so calculate slopes
    # explicitly using the least-squares formula (equivalent to
    # lyapunov_slopes() above, duplicated here so this function stays
    # fully jit-compilable).
    lyaps = []
    for start in range(0, steps - reset_steps, reset_steps):
        t_chunk = running_time[start:start + reset_steps]
        d_chunk = np.log(distances[start:start + reset_steps])

        t_mean = t_chunk.mean()
        d_mean = d_chunk.mean()
        numerator = ((t_chunk - t_mean) * (d_chunk - d_mean)).sum()
        denominator = ((t_chunk - t_mean) ** 2).sum()
        lyaps.append(numerator / denominator)

    return np.mean(np.array(lyaps))


# Keep the original slow/reference implementation available.
def lyapunov_exponent(running_time, reset_steps, tau, perturbation,
                      N, W, dt, rng_initial):
    """
    Reference (non-jitted) Lyapunov-exponent calculation using NumPy/SciPy.
    Slower than lyapunov_exponent_numba but useful for validation, since it
    reuses lyapunov_slopes() (which relies on scipy.stats.linregress and
    therefore can't run inside a @njit function).

    Args:
        running_time: 1D array of timesteps
        reset_steps: number of steps between perturbation resets
        tau: network time constant
        perturbation: magnitude of the initial/reset perturbation
        N: number of neurons
        W: recurrent weight matrix (N, N)
        dt: integration time step
        rng_initial: numpy Generator used to draw the initial state X0

    Returns:
        Estimated largest Lyapunov exponent (float).
    """
    X0 = rng_initial.uniform(-1.0, 1.0, size=N)
    X2 = X0 + perturbation / np.sqrt(N)
    distances = np.empty(len(running_time), dtype=X0.dtype)
    distances[0] = np.linalg.norm(X2 - X0)

    I = np.zeros(N)
    X1 = X0.copy()

    for t in range(1, len(running_time)):
        # NOTE: this previously called an undefined `_rate_network`, which
        # would raise a NameError as soon as this function was called.
        # rate_network_numba is the only function in this module matching
        # the (X, W, I, tau) call signature used here, so that's restored.
        X1 = X1 + dt * rate_network_numba(X1, W, I, tau)
        X2 = X2 + dt * rate_network_numba(X2, W, I, tau)

        dist_vec = X2 - X1
        distances[t] = np.linalg.norm(dist_vec)

        if t % reset_steps == 0:
            X2 = X1 + perturbation * dist_vec / distances[t]

    return lyapunov_slopes(running_time, distances, reset_steps)


def lyapunov_exponent_and_activity(running_time, reset_steps, tau,
                                   perturbation, N, W, dt, rng_initial):
    """
    Like lyapunov_exponent(), but also returns the two full distance/activity
    trajectories: the pairwise distance at every step, plus every-10th-neuron
    activity samples for both trajectories, restricted to the last 3000
    timesteps (to keep memory usage bounded for long runs).

    Args:
        running_time, reset_steps, tau, perturbation, N, W, dt, rng_initial:
            same as lyapunov_exponent() above.

    Returns:
        Tuple of:
            largest Lyapunov exponent (float)
            distances: (steps,) array of ||X2 - X1|| at every step
            activity_X1: sampled activity of trajectory 1 (see subsampling below)
            activity_X2: sampled activity of trajectory 2
    """
    X0 = rng_initial.uniform(-1.0, 1.0, size=N)
    X2 = X0 + perturbation / np.sqrt(N)

    steps = len(running_time)
    distances = np.empty(steps, dtype=X0.dtype)
    distances[0] = np.linalg.norm(X2 - X0)

    X1 = np.empty((steps, N), dtype=X0.dtype)
    X2_history = np.empty((steps, N), dtype=X0.dtype)
    X1[0] = X0
    X2_history[0] = X2

    activity_X1 = []
    activity_X2 = []
    last_3000_time = steps - 3000  # only keep the final 3000 steps of activity
    # Record every 10th neuron (of the first 1000) at t=0.
    activity_X1.append(X0[0:1000:10])
    activity_X2.append(X2[0:1000:10])

    I = np.zeros(N)

    for t in range(1, steps):
        # NOTE: fixed -- previously called an undefined `_rate_network`.
        X1[t] = X1[t - 1] + dt * rate_network_numba(X1[t - 1], W, I, tau)
        X2_history[t] = X2_history[t - 1] + dt * rate_network_numba(
            X2_history[t - 1], W, I, tau
        )

        dist_vec = X2_history[t] - X1[t]
        distances[t] = np.linalg.norm(dist_vec)

        if t >= last_3000_time:
            activity_X1.append(X1[t, 0:1000:10])
            activity_X2.append(X2_history[t, 0:1000:10])

        if t % reset_steps == 0:
            X2_history[t] = X1[t] + perturbation * dist_vec / distances[t]

    return (
        # NOTE: fixed -- previously called an undefined `_lyapunov_slopes`;
        # the function actually defined in this module is `lyapunov_slopes`.
        lyapunov_slopes(running_time, distances, reset_steps),
        distances,
        np.asarray(activity_X1),
        np.asarray(activity_X2),
    )


def activities(running_time, reset_steps, tau, perturbation, N, W, dt,
               rng_initial):
    """
    Simulate two initially-nearby trajectories (as in the Lyapunov
    calculations above) but only return their sampled activity, without
    computing the Lyapunov exponent itself.

    Note the perturbation-reset step here differs subtly from the
    lyapunov_* functions: it resets X2 to X1 + perturbation (a fixed
    per-neuron offset) rather than X1 + perturbation * unit_vector
    (rescaling along the current divergence direction). This matches the
    original implementation and is preserved as-is.

    Args:
        running_time, reset_steps, tau, perturbation, N, W, dt, rng_initial:
            same as lyapunov_exponent() above.

    Returns:
        Tuple of (activity_X1, activity_X2), each an array of every-10th-neuron
        samples for the last 3000 timesteps (plus the initial sample at t=0).
    """
    X0 = rng_initial.uniform(-1.0, 1.0, size=N)
    X2 = X0 + perturbation / np.sqrt(N)

    steps = len(running_time)
    X1 = np.empty((steps, N), dtype=X0.dtype)
    X2_history = np.empty((steps, N), dtype=X0.dtype)
    X1[0] = X0
    X2_history[0] = X2

    activity_X1 = [X0[0:1000:10]]
    activity_X2 = [X2[0:1000:10]]
    last_3000_time = steps - 3000
    I = np.zeros(N)

    for t in range(1, steps):
        # NOTE: fixed -- previously called an undefined `_rate_network`.
        X1[t] = X1[t - 1] + dt * rate_network_numba(X1[t - 1], W, I, tau)
        X2_history[t] = X2_history[t - 1] + dt * rate_network_numba(
            X2_history[t - 1], W, I, tau
        )

        if t >= last_3000_time:
            activity_X1.append(X1[t, 0:1000:10])
            activity_X2.append(X2_history[t, 0:1000:10])

        if t % reset_steps == 0:
            X2_history[t] = X1[t] + perturbation

    return np.asarray(activity_X1), np.asarray(activity_X2)