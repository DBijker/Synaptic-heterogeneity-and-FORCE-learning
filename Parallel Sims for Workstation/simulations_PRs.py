import numpy as np
import module as md
from mpi4py import MPI
from itertools import product
import os

# ==== MPI setup ====
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

results_folder = "PRs_HSMs0-0.1_Gains1-3_W5-9_and_Init0-3"

if rank == 0:
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

comm.Barrier()  # all ranks wait until rank 0 has created the folder

# ==== Fixed network parameters ====
N = 1000
tau = 10
dt = 0.1
p = 1

# Simulation duration for the participation-ratio runs
running_time = np.arange(0, 2000 + dt, dt)
time_steps = int(len(running_time))

# ==== Parameter sweep ranges ====
n_weight_seeds = np.array([5, 6, 7, 8, 9])
n_init_seeds = np.array([0, 1, 2, 3])
MSHs = np.arange(0, 0.101, 0.001)
gains = np.arange(1, 3.2, 0.2)

# No external input during these runs
I = np.zeros((len(running_time), N))

# Every (gain, weights_seed, MSH, init_seed) combination to run
tasks = list(product(gains, n_weight_seeds, MSHs, n_init_seeds))

local_tasks = tasks[rank::size]

# Per-rank result accumulators
local_PRs = []
local_Weights_Seeds = []
local_Initialisation_Seeds = []
local_MSHs = []
local_Gains = []

for gain, weights_seed, MSH, initialisation_seed in local_tasks:

    # Independent, reproducible RNG streams for weights vs. initial state
    weights_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, weights_seed))
    initialisation_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, weights_seed, initialisation_seed))

    rng_weights = np.random.default_rng(weights_seq)
    rng_initialisation = np.random.default_rng(initialisation_seq)

    # Recurrent weight matrix for this (gain, MSH, weights_seed) config
    W = md.weight_matrix_fast(p, N, MSH, gain, rng_weights)

    # Simulate the full activity trajectory from a random initial state
    X0 = rng_initialisation.uniform(-1.0, 1.0, size=N)
    X_full_activity = md.euler_rate_network_numba_full_trajectory(W, tau, dt, time_steps, X0, I)

    # Drop the first 2000 steps (200 ms at dt=0.1) so the network has time
    # to settle from its random initial condition before measuring
    # dimensionality on the remaining activity.
    x = X_full_activity[2000:, :]

    PR = md.participation_ratio(np.tanh(x))
    local_PRs.append(PR)

    # Track parameters alongside results for later identification
    local_Gains.append(gain)
    local_Weights_Seeds.append(weights_seed)
    local_Initialisation_Seeds.append(initialisation_seed)
    local_MSHs.append(MSH)

# Convert accumulators to arrays before saving
local_PRs = np.array(local_PRs)
local_Weights_Seeds = np.array(local_Weights_Seeds)
local_Initialisation_Seeds = np.array(local_Initialisation_Seeds)
local_MSHs = np.array(local_MSHs)
local_Gains = np.array(local_Gains)

filename = f"Zoomed_in_gains1-3_HSMs0-0.1_PRs_W_and_Init_rank{rank}.npz"
output_path = os.path.join(results_folder, filename)

np.savez(
    output_path,
    PRs=local_PRs,
    Gains=local_Gains,
    Weights_Seeds=local_Weights_Seeds,
    Initialisation_Seeds=local_Initialisation_Seeds,
    MSHs=local_MSHs,
)

print(f"hello from rank {rank} of {size}")
