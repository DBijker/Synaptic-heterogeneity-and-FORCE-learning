import numpy as np
import module as md
from mpi4py import MPI
from itertools import product
import os

# ==== MPI setup ====
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

results_folder = "LLEs_HSMs0-0.1_Gains1-3_W5-9_and_Init0-3"

if rank == 0:
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

comm.Barrier()  # all ranks wait until rank 0 has created the folder

# ==== Fixed network parameters ====
N = 1000
tau = 10
dt = 0.1
p = 1

running_time = np.arange(0, 3000 + dt, dt)
interval = 200                        # ms between perturbation resets
reset_steps = int(interval / dt)
perturbation = 1e-8

# ==== Parameter sweep ranges ====
n_weight_seeds = np.array([5, 6, 7, 8, 9])
n_init_seeds = np.array([0, 1, 2, 3])
HSMs = np.arange(0, 0.101, 0.001)
gains = np.arange(1, 3.2, 0.2)

# Every (gain, weights_seed, HSM, init_seed) combination to run
tasks = list(product(gains, n_weight_seeds, HSMs, n_init_seeds))

local_tasks = tasks[rank::size]

# Per-rank result accumulators
local_LLEs = []
local_Weights_Seeds = []
local_Initialisation_Seeds = []
local_HSMs = []
local_Gains = []

for gain, weights_seed, HSM, initialisation_seed in local_tasks:

    # Independent, reproducible RNG streams for weights vs. initial state,
    # spawned from a shared entropy value. spawn_key=(experiment, repetition,
    # trial) -- here (0, weights_seed, initialisation_seed).
    weights_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, weights_seed))
    initialisation_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, weights_seed, initialisation_seed))

    rng_weights = np.random.default_rng(weights_seq)
    rng_initialisation = np.random.default_rng(initialisation_seq)

    # Recurrent weight matrix for this (gain, HSM, weights_seed) config
    W = md.weight_matrix_fast(p, N, HSM, gain, rng_weights)

    # md.lyapunov_exponent_numba needs an explicit initial-state array
    # (rather than an rng object) since it's numba-jitted
    X0 = rng_initialisation.uniform(-1.0, 1.0, size=N)
    LLE = md.lyapunov_exponent_numba(running_time, reset_steps, tau, perturbation, N, W, dt, X0)

    local_LLEs.append(LLE)

    # Track parameters alongside results for later identification
    local_Gains.append(gain)
    local_Weights_Seeds.append(weights_seed)
    local_Initialisation_Seeds.append(initialisation_seed)
    local_HSMs.append(HSM)

# Convert accumulators to arrays before saving
local_LLEs = np.array(local_LLEs)
local_Weights_Seeds = np.array(local_Weights_Seeds)
local_Initialisation_Seeds = np.array(local_Initialisation_Seeds)
local_HSMs = np.array(local_HSMs)
local_Gains = np.array(local_Gains)

filename = f"Zoomed_in_gains1-3_HSMs0-0.1_LLEs_W_and_Init_rank{rank}.npz"
output_path = os.path.join(results_folder, filename)

np.savez(
    output_path,
    LLEs=local_LLEs,
    Gains=local_Gains,
    Weights_Seeds=local_Weights_Seeds,
    Initialisation_Seeds=local_Initialisation_Seeds,
    HSMs=local_HSMs,
)

print(f"hello from rank {rank} of {size}")
