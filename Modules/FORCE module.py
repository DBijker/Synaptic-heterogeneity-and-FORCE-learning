#################### Imports ####################
import numpy as np
from numba import njit               # JIT-compiles the rate-network integrators for speed
from typing import Dict, Tuple
from tqdm import tqdm                 # progress bar used by FORCETrainer.train()/.test()

#################### Utilities ####################

def create_complex_target(running_time, dt, freq=1 / 60, amps=None):
    """
    Build a multi-harmonic sine-wave target signal (a standard FORCE-learning
    benchmark target, following Sussillo & Abbott-style construction).

    Args:
        running_time: time array the target is evaluated over
        dt: time step (kept for API symmetry; not used directly below)
        freq: base frequency
        amps: harmonic amplitudes [amp1, amp2, ...]; defaults to a fixed
              4-harmonic mix if not provided

    Returns:
        Target: array of the same shape as running_time
    """
    if amps is None:
        amps = [0.7, 0.35, 0.1166, 0.2333]

    # Harmonic frequencies: freq, 2*freq, 3*freq, ...
    freqs = [freq * i for i in range(1, len(amps) + 1)]

    Target = sum(amp * np.sin(np.pi * f * running_time)
                 for amp, f in zip(amps, freqs))

    Target /= 1.5  # normalize overall amplitude
    return Target


def random_search(param_ranges: Dict[str, Tuple], gain, HSM,
                   n_iterations: int = 30,
                   n_trials: int = 3,
                   train_time: float = 10000,
                   test_time: float = 5000,
                   verbose: bool = True):
    """
    Random search over FORCETrainer hyperparameters (more efficient than a
    full grid search for this many dimensions).

    Args:
        param_ranges: dict of parameter name -> (min, max), e.g.
                      {'eta': (0.01, 0.5), 'Dt': (1.0, 5.0)}
        gain, HSM: fixed network parameters passed through to FORCETrainer
        n_iterations: number of random hyperparameter configurations to try
        n_trials: number of repeated trials (different seeds) per configuration
        train_time: training duration (ms)
        test_time: testing duration (ms)
        verbose: print progress

    Returns:
        List of result dicts, sorted by ascending mean test MAE (best first)
    """
    # Fixed (non-searched) parameters
    N = 1000
    p = 1.0
    dt = 0.1
    tau = 10.0
    alpha = 1.0

    results = []

    for i in range(n_iterations):
        # Sample one random value per hyperparameter, uniformly from its range
        params = {
            name: np.random.uniform(min_val, max_val)
            for name, (min_val, max_val) in param_ranges.items()
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Iteration {i + 1}/{n_iterations}: {params}")
            print(f"{'=' * 60}")

        trial_maes = []
        trial_train_maes = []

        for trial in range(n_trials):
            trainer = FORCETrainer(
                N=N,
                p=p,
                gain=gain,
                HSM=HSM,
                tau=tau,
                dt=dt,
                Dt=params.get('Dt', 2.0),
                alpha=alpha,
                eta=params.get('eta', 0.1),
                weights_seed=trial,
                initialisation_seed=trial,  # can be changed independently if needed
            )

            # Train
            running_time = np.arange(0, train_time, dt)
            target_train = create_complex_target(running_time, dt)
            train_results = trainer.train(target_train, running_time, verbose=False)

            train_mae = np.mean(np.abs(train_results['all_z_out'] - target_train))
            trial_train_maes.append(train_mae)

            # Test
            test_running_time = np.arange(0, test_time, dt)
            target_test = create_complex_target(test_running_time, dt)
            test_results = trainer.test(target_test, test_running_time, verbose=False)

            trial_maes.append(test_results['mae'])

            if verbose:
                print(f"  Trial {trial + 1}/{n_trials}: "
                      f"Train MAE={train_mae:.6f}, Test MAE={test_results['mae']:.6f}")

        result = {
            **params,
            'test_maes': np.array(trial_maes),
            'train_maes': np.array(trial_train_maes),
            'mean_test_mae': np.mean(trial_maes),
            'std_test_mae': np.std(trial_maes),
            'mean_train_mae': np.mean(trial_train_maes),
            'std_train_mae': np.std(trial_train_maes),
            'best_test_mae': np.min(trial_maes),
            'worst_test_mae': np.max(trial_maes),
        }
        results.append(result)

        if verbose:
            print(f"  Average: Train={result['mean_train_mae']:.6f}, "
                  f"Test={result['mean_test_mae']:.6f} \u00b1 {result['std_test_mae']:.6f}")

    # Best configuration first
    results.sort(key=lambda x: x['mean_test_mae'])
    return results


#################### FORCE trainer ####################

class FORCETrainer:
    """
    FORCE-learning trainer: a recurrent tanh rate network trained online
    with Recursive Least Squares (RLS) on the readout weights, plus a
    proportional update of the recurrent weights J (scaled by eta).
    """

    def __init__(self, N=1000, p=1.0, gain=1.5, HSM=0,
                 tau=1, dt=0.1, Dt=2.0, alpha=1.0, eta=0.1,
                 weights_seed=1, initialisation_seed=1):
        """
        Args:
            N: number of neurons
            p: connection probability
            gain: gain / coupling-strength parameter for the recurrent weights
            HSM: mean-shift parameter for the recurrent weight distribution
            tau: network time constant (ms)
            dt: integration time step (ms)
            Dt: learning update interval (ms)
            alpha: RLS regularization / forgetting factor (sets initial P = I/alpha)
            eta: learning rate for the recurrent weight (J) updates
            weights_seed: seed for the recurrent/readout weight RNG
            initialisation_seed: seed for the initial-state RNG
        """
        self.N = N
        self.p = p
        self.gain = gain
        self.HSM = HSM
        self.tau = tau
        self.dt = dt
        self.Dt = Dt
        self.alpha = alpha
        self.eta = eta

        # Independent, reproducible RNG streams for weights vs. initial state.
        # Spawning from a shared entropy value keeps runs reproducible across seeds.
        weights_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, weights_seed))
        initialisation_seq = np.random.SeedSequence(
            entropy=654321, spawn_key=(0, weights_seed, initialisation_seed)
        )
        self.rng_weights = np.random.default_rng(weights_seq)
        self.rng_initialisation = np.random.default_rng(initialisation_seq)

        self._init_network()

    def _init_network(self):
        """Initialize recurrent weights J, readout weights w_out, state, and RLS matrix P."""
        self.J = weight_matrix_fast(self.p, self.N, self.HSM, self.gain, self.rng_weights)
        self.I = np.zeros((self.N, 1))
        self.x = 0.5 * self.rng_initialisation.normal(0, 1, size=(self.N, 1))
        self.r = np.tanh(self.x)

        # Readout weights: sparse random init, scaled and re-centered to zero mean
        self.w_out = self.rng_initialisation.normal(
            loc=0, scale=1 / np.sqrt(self.p * self.N), size=(self.N, 1)
        )
        mask = self.rng_initialisation.random(self.w_out.shape) < self.p
        self.w_out *= mask
        self.w_out -= self.w_out.mean()
        self.w_out *= (1 / np.sqrt(self.p * self.N))

        # RLS covariance-like matrix, initialized as I/alpha
        self.P = np.identity(self.N) / self.alpha

        # Keep copies of the untrained weights for reference/comparison
        self.J_initial = self.J.copy()
        self.w_out_initial = self.w_out.copy()

    def train(self, target, running_time, learn_start=500, verbose=True):
        """
        Train the network with FORCE learning (RLS on w_out, proportional
        update on J).

        Args:
            target: target signal array
            running_time: time array (ms)
            learn_start: time (ms) at which learning updates begin
            verbose: show a tqdm progress bar

        Returns:
            dict with the full output trajectory and timing info
        """
        total_time_steps = len(running_time)
        self.total_time_steps = total_time_steps

        learn_every = int(self.Dt / self.dt)       # steps between learning updates
        learn_start_idx = int(learn_start / self.dt)

        all_z_out = []

        # Local copies of state so training can be re-run without mutating
        # the network's initial conditions until it finishes
        x = self.x.copy()
        r = self.r.copy()
        J = self.J.copy()
        w_out = self.w_out.copy()
        P = self.P.copy()

        iterator = tqdm(range(total_time_steps)) if verbose else range(total_time_steps)

        for t in iterator:
            # Network update (forward-Euler step of the tanh rate network)
            x += (self.dt / self.tau) * (-x + J @ r + self.I)
            r = np.tanh(x)

            # Readout output at every step
            z = (w_out.T @ r)[0, 0]
            all_z_out.append(z)

            # Learning update, applied every `learn_every` steps once past learn_start_idx
            if t >= learn_start_idx and t % learn_every == 0:
                target_idx = t - learn_start_idx  # align target's clock with learning onset

                if target_idx < len(target):  # safety check against running past the target
                    e_min = z - target[target_idx]

                    # RLS update of the readout weights
                    Pr = P @ r                          # (N, 1)
                    rTPr = (r.T @ Pr)[0, 0]
                    normalizer = 1.0 / (1.0 + rTPr)

                    delta_w = -normalizer * (Pr * e_min)  # (N, 1)
                    w_out += delta_w

                    P -= normalizer * (Pr @ Pr.T)         # update RLS covariance matrix

                    # Recurrent weight update: broadcast delta_w across all rows,
                    # scaled by the learning rate eta
                    J += self.eta * (np.ones((self.N, 1)) @ delta_w.T)

        # Store trained parameters / final state for use by test()
        self.J_trained = J.copy()
        self.w_out_trained = w_out.copy()
        self.x_final = x.copy()
        self.r_final = r.copy()

        return {
            'all_z_out': np.array(all_z_out),
            'running_time': running_time,
            'total_time_steps': total_time_steps
        }

    def test(self, target, test_time, from_final_state=True, verbose=True):
        """
        Run the trained (frozen) network forward and evaluate against a target.

        Args:
            target: target signal for testing
            test_time: time array for testing
            from_final_state: continue from the final training state (True)
                               or start from a fresh random state (False)
            verbose: show a tqdm progress bar

        Returns:
            dict with the test output trajectory and mean absolute error (MAE)
        """
        if from_final_state:
            x_test = self.x_final.copy()
            r_test = self.r_final.copy()
        else:
            x_test = 0.5 * self.rng_initialisation.normal(size=(self.N, 1))
            r_test = np.tanh(x_test)

        z_tests = []

        iterator = tqdm(range(len(test_time))) if verbose else range(len(test_time))

        for i in iterator:
            # Same network update rule as training, but with weights frozen
            x_test += (self.dt / self.tau) * (-x_test + self.J_trained @ r_test + self.I)
            r_test = np.tanh(x_test)

            z_test = (self.w_out_trained.T @ r_test)[0, 0]
            z_tests.append(z_test)

        z_tests = np.array(z_tests)
        mae = np.mean(np.abs(z_tests - target[:len(z_tests)]))

        return {
            'z_out': z_tests,
            'test_time': test_time,
            'mae': mae
        }