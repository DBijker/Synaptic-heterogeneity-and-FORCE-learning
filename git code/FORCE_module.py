#################imports########################
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal, poisson, norm, expon, chi2, multinomial
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import polyfit
from scipy.stats import linregress
from itertools import groupby
from numba import njit, prange
import subprocess


# from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Callable, Tuple
from functools import partial
import itertools

#################Functions######################

def euler_rate_network(network, x0, t, W, I, tau, activation):
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    h = t[1] - t[0]
    for i in range(1, len(t)):
        x[i] = x[i-1] + h*network(activation, x[i-1], W, I[i-1], tau)
    return x

def weight_matrix_fast(p, N, MSH, gain, rng_weights):
    
    mus = np.linspace(-MSH, MSH, N)

    # Generate full matrix of normal random numbers in one call
    W = rng_weights.normal(
        loc=mus[:, None],             # broadcast mus over columns
        scale=gain/np.sqrt(p*N),
        size=(N, N)
    )

    mask = rng_weights.random(W.shape) < p #rng_weights.random generates numbers in [0,1) so this line acts as a filter for introducing sparsity. if p = 1 than no filtering happen because all generated numbers are below 1
    W *= mask
    
    # Remove self-connections
    np.fill_diagonal(W, 0)
    return W

# def connection_matrix_fast(p, N, MSH, gain, rng_weights):
    
#     mus = np.linspace(-MSH, MSH, N)

#     # Generate full matrix of normal random numbers in one call
#     J = rng_weights.normal(
#         loc=mus[:, None],             # broadcast mus over columns
#         scale=1,
#         size=(N, N)
#     )

#     # mask = rng_weights.random(J.shape) < p
#     # J *= mask
#     # J *= (1/np.sqrt(p*N))
    
#     mask = rng_weights.random(J.shape) < p #rng_weights.random generates numbers in [0,1) so this line acts as a filter for introducing sparsity. if p = 1 than no filtering happen because all generated numbers are below 1
#     J *= mask
#     #J -= J.mean()
#     J *= (gain/np.sqrt(p*N))

#     #setting diagonal values to 0
#     np.fill_diagonal(J, 0)
    
#     return J

@njit(fastmath=True)
def rate_network_numba(X, J, I, tau):
    return (-X + J @ np.tanh(X) + I)/tau

@njit(fastmath=True)
def euler_rate_network_numba_full_trajectory(W, tau, dt, t_steps, x0, I):
    N = len(x0)
    x = np.zeros((t_steps, N), dtype=x0.dtype)
    x[0, :] = x0 # Explicitly set the first row

    X = np.copy(x0)

    for i in range(1, t_steps):
        # select input (time varying or constant)
        if I.ndim == 2:
            Ii = I[i-1, :]
        else: 
            Ii = I
        
        X = X + dt * rate_network_numba(X, W, Ii, tau)

        x[i, :] = X # Explicitly assign the 1D array X to the 2D slice x[i, :]

    return x # x is always 2D



def show(name, var):
    try:
        shape = var.shape
    except AttributeError:
        shape = "no .shape attribute"
    try:
        length = len(var)
    except TypeError:
        length = "no len()"
    print(f"{name:10s} | shape: {shape} | len: {length}")


def create_complex_target(running_time, dt, freq=1/60, amps=None):
    """
    Create your complex target function
    
    Args:
        running_time: Time array
        dt: Time step
        freq: Base frequency
        amps: Amplitudes for harmonics [amp1, amp2, ..., amp6]
    """
    if amps is None:
        #amps = [0.3, 0.5, 0.7, 0.2, 0.1, 0.9]
        amps = [0.7, 0.35, 0.1166, 0.2333]
        
    # convertion_ratio = 1000
    # Time = int(len(running_time) * dt)
    # xs = np.arange(0, Time, dt)
    
    freqs = [freq * i for i in range(1, (len(amps)+1))]
    
    #Target = sum(amp * np.sin(2*np.pi*f*xs/convertion_ratio) 
    #             for amp, f in zip(amps, freqs))
    
    Target = sum(amp * np.sin(np.pi*f*running_time)
                 for amp, f in zip(amps, freqs))

    #Sussillo's equivalent way: Target = (amp/1.0)*np.sin(1.0*np.pi*freq*running_time) + (amp/2.0)*np.sin(2.0*np.pi*freq*running_time) + (amp/6.0)*np.sin(3.0*np.pi*freq*running_time) + (amp/3.0)*np.sin(4.0*np.pi*freq*running_time)
   
    Target /= 1.5

    
    return Target

def random_search(param_ranges: Dict[str, Tuple], gain, HSM,
                                   n_iterations: int = 30,
                                   n_trials: int = 3,
                                   train_time: float = 10000,
                                   test_time: float = 5000,
                                   verbose: bool = True):
    """
    Random search over hyperparameters (more efficient than grid search)
    
    Args:
        param_ranges: Dictionary of parameter names to (min, max) tuples
                     e.g., {'eta': (0.01, 0.5), 'alpha': (1.0, 50.0)}
        n_iterations: Number of random configurations to try
        n_trials: Number of trials per configuration
        train_time: Training duration (ms)
        test_time: Testing duration (ms)
        verbose: Print progress
        
    Returns:
        List of result dictionaries sorted by performance
    """
    # Fixed parameters
    N = 1000
    p = 1.0
    #gain = 1.5
    dt = 0.1
    tau = 10.0

    alpha = 1.0
    
    results = []
    
    for i in range(n_iterations):
        # Sample random parameters
        params = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            # if param_name in ['Dt']:
            #     # Sample on log scale for Dt
            #     params[param_name] = 10 ** np.random.uniform(np.log10(min_val), np.log10(max_val))
            # else:
            params[param_name] = np.random.uniform(min_val, max_val)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Iteration {i+1}/{n_iterations}: {params}")
            print(f"{'='*60}")
        
        trial_maes = []
        trial_train_maes = []
        
        for trial in range(n_trials):
            # Create trainer
            trainer = FORCETrainer(
                N=N,
                p=p,
                gain=gain,
                HSM=HSM,
                tau=tau,
                dt=dt,
                Dt=params.get('Dt', 2.0),
                #alpha=params.get('alpha', 10.0),
                alpha=alpha,
                eta=params.get('eta', 0.1),
                weights_seed=trial,
                initialisation_seed=trial #can be changed
            )
            
            # Create target
            running_time = np.arange(0, train_time , dt)
            target_train = create_complex_target(running_time, dt)
            
            # Train
            train_results = trainer.train(
                target_train, 
                running_time,
                verbose=False
            )
            
            # Calculate training MAE
            train_mae = np.mean(np.abs(train_results['all_z_out'] - target_train))
            trial_train_maes.append(train_mae)
            
            # Test
            test_running_time = np.arange(0, test_time, dt)
            target_test = create_complex_target(test_running_time, dt)
            test_results = trainer.test(
                target_test,
                test_running_time,
                verbose=False
            )
            
            trial_maes.append(test_results['mae'])
            
            if verbose:
                print(f"  Trial {trial+1}/{n_trials}: Train MAE={train_mae:.6f}, Test MAE={test_results['mae']:.6f}")
        
        result = {
            **params,
            'test_maes': np.array(trial_maes),
            'train_maes': np.array(trial_train_maes),
            'mean_test_mae': np.mean(trial_maes),
            'std_test_mae': np.std(trial_maes),
            'mean_train_mae': np.mean(trial_train_maes),
            'std_train_mae': np.std(trial_train_maes),
            'best_test_mae': np.min(trial_maes),
            'worst_test_mae': np.max(trial_maes)
        }
        results.append(result)
        
        if verbose:
            print(f"  Average: Train={result['mean_train_mae']:.6f}, Test={result['mean_test_mae']:.6f} ± {result['std_test_mae']:.6f}")
    
    # Sort by mean test MAE
    results.sort(key=lambda x: x['mean_test_mae'])
    
    return results



# def create_complex_target(running_time, dt, freq=0.5, amps=None):
#     """
#     Create your complex target function
    
#     Args:
#         running_time: Time array
#         dt: Time step
#         freq: Base frequency
#         amps: Amplitudes for harmonics [amp1, amp2, ..., amp6]
#     """
#     if amps is None:
#         amps = [0.3, 0.5, 0.7, 0.2, 0.1, 0.9]
    
#     convertion_ratio = 1000
#     Time = int(len(running_time) * dt)
#     xs = np.arange(0, Time, dt)
    
#     freqs = [freq * i for i in range(1, 7)]
    
#     Target = sum(amp * np.sin(2*np.pi*f*xs/convertion_ratio) 
#                  for amp, f in zip(amps, freqs))
    
#     return Target

# def create_complex_target(running_time, dt, freq=1/60, amps=None):
#     """
#     Create your complex target function
    
#     Args:
#         running_time: Time array
#         dt: Time step
#         freq: Base frequency
#         amps: Amplitudes for harmonics [amp1, amp2, ..., amp6]
#     """
#     if amps is None:
#         #amps = [0.3, 0.5, 0.7, 0.2, 0.1, 0.9]
#         amps = [0.7, 0.35, 0.1166, 0.2333]
        
#     # convertion_ratio = 1000
#     # Time = int(len(running_time) * dt)
#     # xs = np.arange(0, Time, dt)
    
#     freqs = [freq * i for i in range(1, (len(amps)+1))]
    
#     #Target = sum(amp * np.sin(2*np.pi*f*xs/convertion_ratio) 
#     #             for amp, f in zip(amps, freqs))
    
#     Target = sum(amp * np.sin(np.pi*f*running_time)
#                  for amp, f in zip(amps, freqs))

#     #Sussillo's equivalent way: Target = (amp/1.0)*np.sin(1.0*np.pi*freq*running_time) + (amp/2.0)*np.sin(2.0*np.pi*freq*running_time) + (amp/6.0)*np.sin(3.0*np.pi*freq*running_time) + (amp/3.0)*np.sin(4.0*np.pi*freq*running_time)
   
#     Target /= 1.5

    
    # return Target


# def random_search(param_ranges: Dict[str, Tuple], gain, HSM,
#                                    n_iterations: int = 30,
#                                    n_trials: int = 3,
#                                    train_time: float = 10000,
#                                    test_time: float = 5000,
#                                    verbose: bool = True):
#     """
#     Random search over hyperparameters (more efficient than grid search)
    
#     Args:
#         param_ranges: Dictionary of parameter names to (min, max) tuples
#                      e.g., {'eta': (0.01, 0.5), 'alpha': (1.0, 50.0)}
#         n_iterations: Number of random configurations to try
#         n_trials: Number of trials per configuration
#         train_time: Training duration (ms)
#         test_time: Testing duration (ms)
#         verbose: Print progress
        
#     Returns:
#         List of result dictionaries sorted by performance
#     """
#     # Fixed parameters
#     N = 1000
#     p = 1.0
#     #gain = 1.5
#     dt = 0.1
#     tau = 10.0

#     alpha = 1.0
    
#     results = []
    
#     for i in range(n_iterations):
#         # Sample random parameters
#         params = {}
#         for param_name, (min_val, max_val) in param_ranges.items():
#             # if param_name in ['Dt']:
#             #     # Sample on log scale for Dt
#             #     params[param_name] = 10 ** np.random.uniform(np.log10(min_val), np.log10(max_val))
#             # else:
#             params[param_name] = np.random.uniform(min_val, max_val)
        
#         if verbose:
#             print(f"\n{'='*60}")
#             print(f"Iteration {i+1}/{n_iterations}: {params}")
#             print(f"{'='*60}")
        
#         trial_maes = []
#         trial_train_maes = []
        
#         for trial in range(n_trials):
#             # Create trainer
#             trainer = FORCETrainer(
#                 N=N,
#                 p=p,
#                 gain=gain,
#                 HSM=HSM,
#                 tau=tau,
#                 dt=dt,
#                 Dt=params.get('Dt', 2.0),
#                 #alpha=params.get('alpha', 10.0),
#                 alpha=alpha,
#                 eta=params.get('eta', 0.1),
#                 weights_seed=trial,
#                 initialisation_seed=trial #can be changed
#             )
            
#             # Create target
#             running_time = np.arange(0, train_time, dt)
#             target_train = create_complex_target(running_time, dt)
            
#             # Train
#             train_results = trainer.train(
#                 target_train, 
#                 running_time,
#                 verbose=False
#             )
            
#             # Calculate training MAE
#             train_mae = np.mean(np.abs(train_results['all_z_out'] - target_train))
#             trial_train_maes.append(train_mae)
            
#             # Test
#             test_running_time = np.arange(0, test_time, dt)
#             target_test = create_complex_target(test_running_time, dt)
#             test_results = trainer.test(
#                 target_test,
#                 test_running_time,
#                 verbose=False
#             )
            
#             trial_maes.append(test_results['mae'])
            
#             if verbose:
#                 print(f"  Trial {trial+1}/{n_trials}: Train MAE={train_mae:.6f}, Test MAE={test_results['mae']:.6f}")
        
#         result = {
#             **params,
#             'test_maes': np.array(trial_maes),
#             'train_maes': np.array(trial_train_maes),
#             'mean_test_mae': np.mean(trial_maes),
#             'std_test_mae': np.std(trial_maes),
#             'mean_train_mae': np.mean(trial_train_maes),
#             'std_train_mae': np.std(trial_train_maes),
#             'best_test_mae': np.min(trial_maes),
#             'worst_test_mae': np.max(trial_maes)
#         }
#         results.append(result)
        
#         if verbose:
#             print(f"  Average: Train={result['mean_train_mae']:.6f}, Test={result['mean_test_mae']:.6f} ± {result['std_test_mae']:.6f}")
    
#     # Sort by mean test MAE
#     results.sort(key=lambda x: x['mean_test_mae'])
    
#     return results



#################FORCE class####################
class FORCETrainer:
    """
    FORCE trainer using YOUR update rule and parameters
    Optimizes over: eta, alpha, Dt, tau
    """
    
    def __init__(self, N=1000, p=1.0, gain=1.5, HSM=0, 
                 tau=1, dt=0.1, Dt=2.0, alpha=1.0, eta=0.1,
                 weights_seed=1, initialisation_seed=1):
        """
        Initialize with your original parameters
        
        Args:
            N: Number of neurons
            p: Connection probability
            gain: Gain parameter
            HSM: Mean shift for connection matrix
            tau: Time constant (ms) - OPTIMIZE THIS
            dt: Integration time step (ms)
            Dt: Learning update interval (ms) - OPTIMIZE THIS
            alpha: RLS forgetting factor - OPTIMIZE THIS
            eta: Learning rate for J updates - OPTIMIZE THIS
            weights_seed: Seed for weight initialization
            initialisation_seed: Seed for state initialization
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
        
        # Initialize random number generators
        weights_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, weights_seed))
        initialisation_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, weights_seed, initialisation_seed))
        #initialisation_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, initialisation_seed))
        self.rng_weights = np.random.default_rng(weights_seq)
        self.rng_initialisation = np.random.default_rng(initialisation_seq)
        
        # Initialize network
        self._init_network()
    
    def _init_network(self):
        """Initialize network weights and state - your original method"""
        #self.J = connection_matrix_fast(self.p, self.N, self.HSM, self.gain, self.rng_weights)
        self.J = weight_matrix_fast(self.p, self.N, self.HSM, self.gain, self.rng_weights)
        self.I = np.zeros((self.N, 1))
        self.x = 0.5 * self.rng_initialisation.normal(0, 1, size=(self.N, 1))
        self.r = np.tanh(self.x)
        
        # Readout weights
        self.w_out = self.rng_initialisation.normal(loc=0, scale=1/(np.sqrt(self.p*self.N)), size=(self.N, 1))
        mask = self.rng_initialisation.random(self.w_out.shape) < self.p
        self.w_out *= mask
        self.w_out -= self.w_out.mean()
        self.w_out *= (1/(np.sqrt(self.p*self.N)))

        # self.w_out = self.rng_initialisation.normal(loc=0, scale=1/(np.sqrt(0.1*self.N)), size=(self.N, 1))
        # mask = self.rng_initialisation.random(self.w_out.shape) < 0.1
        # self.w_out *= mask
        # self.w_out -= self.w_out.mean()
        # self.w_out *= (1/(np.sqrt(0.1*self.N)))
        
        # RLS matrix
        self.P = np.identity(self.N) / self.alpha

        #to save initial copies
        self.J_initial = self.J.copy()
        self.w_out_initial = self.w_out.copy()

    
    def train(self, target, running_time, learn_start=500, verbose=True):
    # def train(self, target, running_time, learn_start=500, 
    #           J_train_end=15000, w_train_end=30000, verbose=True):
        
        """
        Train network using FORCE learning - your original structure
        
        Args:
            target: Target signal array
            running_time: Time array (ms)
            learn_start: When to start learning (ms)
            J_train_end: When to stop J learning (ms)
            w_train_end: When to stop w_out learning (ms)
            verbose: Show progress bar
            
        Returns:
            dict with training results
        """
        total_time_steps = len(running_time)
        self.total_time_steps = total_time_steps
        
        learn_every = int(self.Dt / self.dt)
        learn_start_idx = int(learn_start / self.dt)
        #J_train_end_idx = int(J_train_end / self.dt)
        #w_train_end_idx = int(w_train_end / self.dt)
        
        # Storage
        z_out = []
        z_time = []
        all_z_out = []
        
        # State
        x = self.x.copy()
        r = self.r.copy()
        J = self.J.copy()
        w_out = self.w_out.copy()
        P = self.P.copy()
 
        iterator = tqdm(range(total_time_steps)) if verbose else range(total_time_steps)
        
        for t in iterator:
            # YOUR ORIGINAL UPDATE RULE
            x += (self.dt / self.tau) * (-x + J @ r + self.I)
            r = np.tanh(x)
            
            # Compute output at every step
            z = (w_out.T @ r)[0, 0]
            # z = (w_out.T @ r)
            
            all_z_out.append(z)
            
            # Learning updates
            if t >= learn_start_idx and t % learn_every == 0:
                # Error (keep as array for proper broadcasting)
                target_idx = t - learn_start_idx

                # Safety check (important!)
                if target_idx < len(target):
                    e_min = z - target[target_idx]#to let the target start at the same time as the learning
                    e_min_arr = np.array([[e_min]])  # Shape (1, 1)
                    
                    # RLS update
                    Pr = P @ r  # Shape (N, 1)
                    rTPr = (r.T @ Pr)[0, 0]
                    normalizer = 1.0 / (1.0 + rTPr)
                    
                    # Update readout weights
                
                    delta_w = -normalizer * (Pr * e_min)  # Shape (N, 1)
                    # maybe: delta_w = -normalizer * (Pr @ e_min.T)
                    
                    w_out += delta_w
                    
                    # Update P matrix
                    P -= normalizer * (Pr @ Pr.T)
                    
                    # Update internal weights with eta
                    
                    J += self.eta * (np.ones((self.N, 1)) @ delta_w.T)
    
                    
                    # z_out.append(z)
                    # z_time.append(running_time[t])
        
        # Store trained parameters
        self.J_trained = J.copy()
        self.w_out_trained = w_out.copy()
        self.x_final = x.copy()
        self.r_final = r.copy()
        
        return {
            # 'z_out': np.array(z_out),
            # 'z_time': np.array(z_time),
            'all_z_out': np.array(all_z_out),
            'running_time': running_time,
            'total_time_steps': total_time_steps
        }
    
    def test(self, target, test_time, from_final_state=True, verbose=True):
        """
        Test trained network
        
        Args:
            target: Target signal for testing
            test_time: Time array for testing
            from_final_state: Start from final training state (True) or random (False)
            verbose: Show progress bar
        """
        # Initialize state
        if from_final_state:
            x_test = self.x_final.copy()
            r_test = self.r_final.copy()
        else:
            x_test = 0.5 * self.rng_initialisation.normal(size=(self.N, 1))
            r_test = np.tanh(x_test)
        
        z_tests = []
        
        iterator = tqdm(range(len(test_time))) if verbose else range(len(test_time))
        
        for i in iterator:
            # YOUR ORIGINAL UPDATE RULE
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
