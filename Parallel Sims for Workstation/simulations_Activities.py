import numpy as np
import LLE_PR_activities_module as md
from mpi4py import MPI
from itertools import product
import os

# ==== MPI setup ====
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Shared folder all ranks write their per-rank result file into
results_folder = "Results_Activities_LLEs_allranks_10_times_sigma_different_initial_conditions"

# Only rank 0 creates the output folder (avoids a race between ranks)
if rank == 0:
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

comm.Barrier()  # all ranks wait until the folder exists before proceeding

# ==== Fixed network parameters ====
N = 1000
tau = 10
dt = 0.1

# Simulation duration for the actual activity runs (30000 ms)
running_time = np.arange(0, 30000 + dt, dt)

interval = 200                        # ms between perturbation resets
reset_steps = int(interval / dt)
perturbation = 1e-8

n_sets = 4  # number of seed values on each axis (weights_seed, init_seed)

# HSM values (heterogeneity/coupling HSM) to sweep
HSMs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0])
gain = np.array([1.0])  # fixed gain value for this simulation

# Every (weights_seed, HSM, initialisation_seed) combination to run
tasks = list(product(range(n_sets + 1), HSMs, range(n_sets + 1)))

# Split tasks across MPI ranks: rank r gets tasks[r], tasks[r+size], ...
local_tasks = tasks[rank::size]

# Per-rank result accumulators
local_activity_X1 = []
local_activity_X2 = []
local_Weights_Seeds = []
local_Initialisation_Seeds = []
local_HSMs = []

for weights_seed, HSM, initialisation_seed in local_tasks:

    # Independent, reproducible RNG streams for weights vs. initial state,
    # spawned from a shared entropy value so runs are reproducible per seed.
    # spawn_key=(experiment, repetition, trial) -- here (0, weights_seed[, initialisation_seed])
    weights_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, weights_seed))
    initialisation_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, weights_seed, initialisation_seed))

    rng_weights = np.random.default_rng(weights_seq)
    rng_initialisation = np.random.default_rng(initialisation_seq)

    W = md.compute_weight_matrix(N, HSM, gain, rng_weights)

    # Two-trajectory activity sampling (see md.activities for details)
    activity_X1, activity_X2 = md.activities(running_time, reset_steps, tau, perturbation, N, W, dt, rng_initialisation)

    local_activity_X1.append(activity_X1)
    local_activity_X2.append(activity_X2)

    # Track parameters alongside results for later identification
    local_Weights_Seeds.append(weights_seed)
    local_Initialisation_Seeds.append(initialisation_seed)
    local_HSMs.append(HSM)

# Convert accumulators to arrays before saving
local_activity_X1 = np.array(local_activity_X1)
local_activity_X2 = np.array(local_activity_X2)
local_Weights_Seeds = np.array(local_Weights_Seeds)
local_Initialisation_Seeds = np.array(local_Initialisation_Seeds)
local_HSMs = np.array(local_HSMs)

filename = f"10_times_sigma_Activities_LLEs_results_different_initial_conditions_rank{rank}.npz"
output_path = os.path.join(results_folder, filename)

np.savez(
    output_path,
    Activity_X1=local_activity_X1,
    Activity_X2=local_activity_X2,
    Weights_Seeds=local_Weights_Seeds,
    Initialisation_Seeds=local_Initialisation_Seeds,
    HSMs=local_HSMs,
)

print(f"hello from rank {rank} of {size}")
