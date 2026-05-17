import numpy as np
import FORCE_module as FC
from mpi4py import MPI
from itertools import product
import os


# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#shared folder name
results_folder = "FORCE_100randoms_complex_periodic_1seed_W_and_Init"

#only rank 0 creates the folder
if rank == 0:
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

#waiting for rank 0 to finish creating the directory
comm.Barrier() # Ensures all ranks wait until the directory is created

# Fixed parameters
N = 1000
tau = 10
dt = 0.1

#change n_sets for test
n_weight_seeds = range(1, 2)
#n_init_seeds = range(1, 2)
#gains = np.arange(0, 10.2, 0.2)
gains = np.array([0.2, 1.0, 1.5, 10])

tasks = list(product(gains, n_weight_seeds)) #26-8-2025 latest updat n_sets+10 for testing with different seeds 

# Distribute tasks among ranks
local_tasks = tasks[rank::size] 

param_ranges = {
    'eta': (0.01, 1),
    #'alpha': (1.0, 50.0),
    'Dt': (0.5, 20.0),
    #'tau': (0.1, 30.0)
}

# Local storage for results
local_Etas = []
local_Dts = []
local_Weights_Seeds = []
#local_Initialisation_Seeds = []
#local_MSHs = []
local_Gains = []

local_mean_test_mae = []
local_std_test_mae= []

# Perform local computation
for gain, weights_seed in local_tasks:

    random_results = FC.random_search(
    param_ranges,
    gain = gain,
    n_iterations=100,  # Try 20 random configurations
    n_trials=weights_seed, #number of seeds it will try, but I am only interested in one seed for now so one trial
    train_time=20000,
    test_time=10000,
    verbose=True)

    best_params = random_results[0]
    eta = best_params['eta']
    Dt = best_params['Dt']

    mean_test_mae=best_params['mean_test_mae'],
    std_test_mae=best_params['std_test_mae']


    local_Etas.append(eta)
    local_Dts.append(Dt)

    local_mean_test_mae.append(mean_test_mae)
    local_std_test_mae.append(std_test_mae)

    #appending gainss, seeds and bounds for reference
    local_Gains.append(gain)
    local_Weights_Seeds.append(weights_seed)
    #local_Initialisation_Seeds.append(initialisation_seed)

# Each rank saves its own results file
local_Etas = np.array(local_Etas)
local_Dts = np.array(local_Dts)
local_Weights_Seeds = np.array(local_Weights_Seeds)
#local_Initialisation_Seeds = np.array(local_Initialisation_Seeds)
local_Gains = np.array(local_Gains)
local_mean_test_mae = np.array(local_mean_test_mae)
local_std_test_mae = np.array(local_std_test_mae)

filename = f"FORCE_complex_periodic_1seed_W_and_Init{rank}.npz"
output_path = os.path.join(results_folder, filename)
#np.savez(output_path, Etas=local_Etas, Dts = local_Dts, Gains = local_Gains, Weights_Seeds=local_Weights_Seeds, Initialisation_Seeds=local_Initialisation_Seeds, MSHs=local_MSHs)
np.savez(output_path, Etas=local_Etas, Dts = local_Dts, Gains = local_Gains, mean_mae = local_mean_test_mae, std_mae = local_std_test_mae, Weights_Seeds=local_Weights_Seeds)




print(f"hello from rank {rank} of {size}")
