import numpy as np
import FORCE_module as FC
from mpi4py import MPI
from itertools import product
import os

# ==== MPI setup ====
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

results_folder = "FORCE_100randoms_complex_periodic_1seed_W_and_Init"

if rank == 0:
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

comm.Barrier()  # all ranks wait until rank 0 has created the folder

# ==== Parameter sweep ====
n_weight_seeds = range(1, 2)          # currently a single seed
gains = np.array([0.2, 1.0, 1.5, 10])

# Every (gain, weights_seed) combination to run
tasks = list(product(gains, n_weight_seeds))

local_tasks = tasks[rank::size]

# Hyperparameter ranges searched per task
param_ranges = {
    'eta': (0.01, 1),
    'Dt': (0.5, 20.0),
}

# Per-rank result accumulators
local_Etas = []
local_Dts = []
local_Weights_Seeds = []
local_Gains = []
local_mean_test_mae = []
local_std_test_mae = []

for gain, weights_seed in local_tasks:

    random_results = FC.random_search(
        param_ranges,
        gain=gain,
        # FIX: FC.random_search's `HSM` parameter has no default and is
        # required -- this call previously omitted it entirely, which would
        # raise a TypeError before ever running. This script doesn't sweep
        # HSM anywhere else, so HSM=0 (no heterogeneity) is used as the
        # baseline here. Confirm this is the value you actually want.
        HSM=0,
        n_iterations=100,       # number of random hyperparameter draws to try
        n_trials=weights_seed,  # trials per draw (only one seed of interest for now)
        train_time=20000,
        test_time=10000,
        verbose=True,
    )

    # random_results is sorted by ascending mean_test_mae, so [0] is the best config
    best_params = random_results[0]
    eta = best_params['eta']
    Dt = best_params['Dt']

    # FIX: the original had a stray trailing comma after mean_test_mae's
    # value, e.g. `mean_test_mae=best_params[...],` which silently makes
    # mean_test_mae a 1-element tuple instead of a float. Removed here.
    mean_test_mae = best_params['mean_test_mae']
    std_test_mae = best_params['std_test_mae']

    local_Etas.append(eta)
    local_Dts.append(Dt)
    local_mean_test_mae.append(mean_test_mae)
    local_std_test_mae.append(std_test_mae)

    # Track parameters alongside results for later identification
    local_Gains.append(gain)
    local_Weights_Seeds.append(weights_seed)

# Convert accumulators to arrays before saving
local_Etas = np.array(local_Etas)
local_Dts = np.array(local_Dts)
local_Weights_Seeds = np.array(local_Weights_Seeds)
local_Gains = np.array(local_Gains)
local_mean_test_mae = np.array(local_mean_test_mae)
local_std_test_mae = np.array(local_std_test_mae)

filename = f"FORCE_complex_periodic_1seed_W_and_Init{rank}.npz"
output_path = os.path.join(results_folder, filename)

np.savez(
    output_path,
    Etas=local_Etas,
    Dts=local_Dts,
    Gains=local_Gains,
    mean_mae=local_mean_test_mae,
    std_mae=local_std_test_mae,
    Weights_Seeds=local_Weights_Seeds,
)

print(f"hello from rank {rank} of {size}")
