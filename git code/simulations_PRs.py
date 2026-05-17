import numpy as np
import module as md
from mpi4py import MPI
from itertools import product
import os


# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#shared folder name
results_folder = "LLEs_HSMs0-0.1_Gains1-3_W5-9_and_Init0-3"

#only rank 0 creates the folder
if rank == 0:
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

#waiting for rank 0 to finish creating the directory
comm.Barrier() # Ensures all ranks wait until the directory is created


# Parameters
N = 1000
tau = 10
dt = 0.1
p = 1



#settings for TR1
#running_time = np.arange(0, 3000+dt, dt) for old time equal to the running time of MLE
running_time = np.arange(0, 2000+dt, dt)
time_steps = int(len(running_time))

n_weight_seeds = np.array([5, 6, 7, 8, 9])
n_init_seeds = np.array([0, 1, 2, 3])
MSHs = np.arange(0, 0.101, 0.001)
gains = np.arange(1, 3.2, 0.2)


# n_weight_seeds = range(0, 2)
# n_init_seeds = 10
# MSHs = np.arange(0, 5.2, 0.2)
# MSHs = np.append(MSHs, 10)
# gains = np.arange(0, 10.2, 0.2)


#settings for test run on my own computer
# running_time = np.arange(0, 300+dt, dt)
# time_steps = int(len(running_time))

# Test_n_weight_seeds = range(0, 2)
# Test_n_inits_seeds = 2
# Test_MSHs = np.array([0.0, 0.2])
# Test_gains = np.arange(0, 0.4, 0.2)

#independent of test or TR1
I = np.zeros((len(running_time), N))


# Generate all (i, bound) combinations
tasks = list(product(gains, n_weight_seeds, MSHs, n_init_seeds)) #26-8-2025 latest updat n_sets+10 for testing with different seeds 
# tasks = list(product(Test_gains, Test_n_weight_seeds, Test_MSHs, range(Test_n_inits_seeds))) #26-8-2025 latest updat n_sets+10 for testing with different seeds 


# Distribute tasks among ranks
local_tasks = tasks[rank::size] #python list slicing, which distributes the tasks evenly across all ranks, 
# you can also see it as tasks[start:end:step] where start is the rank, end is the total number of tasks, and step is the number of ranks
#we didn't specify an end so it will take all the tasks from the start to the end of the list

# Local storage for results
local_PRs = []
local_Weights_Seeds = []
local_Initialisation_Seeds = []
local_MSHs = []
local_Gains = []

# Perform local computation
for gain, weights_seed, MSH, initialisation_seed in local_tasks:

    weights_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, weights_seed))# entropy = parents seed, spawn_key=(2,3,0) means spawn_key=(#experiment, #repetition, #trial) or trial 0 of rep 3 of exp 2. Creates for loop over trials
    initialisation_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, weights_seed, initialisation_seed))

    rng_weights = np.random.default_rng(weights_seq)
    rng_initialisation = np.random.default_rng(initialisation_seq)


    # Weight matrix
    W = md.weight_matrix_fast(p, N, MSH, gain, rng_weights)


    # PR computation
    X0 = rng_initialisation.uniform(-1.0, 1.0, size = N)
    X_full_activity = md.euler_rate_network_numba_full_trajectory(W, tau, dt, time_steps, X0, I) 
    #for with and without tanh(x)

    #skipping the first 200 ms of x to let the see what the dimensionality does
    #if the initial positions of the activity has worked out or have become ordered again or settled
    x = X_full_activity[2000::, :]

    PR = md.participation_ratio(np.tanh(x))
    #PR = md.participation_ratio(x)
    local_PRs.append(PR)

    #appending seeds and bounds for reference
    local_Gains.append(gain)
    local_Weights_Seeds.append(weights_seed)
    local_Initialisation_Seeds.append(initialisation_seed)
    local_MSHs.append(MSH)

# Each rank saves its own results file
local_PRs = np.array(local_PRs)
local_Weights_Seeds = np.array(local_Weights_Seeds)
local_Initialisation_Seeds = np.array(local_Initialisation_Seeds)
local_MSHs = np.array(local_MSHs)
local_Gains = np.array(local_Gains)


filename = f"Zoomed_in_gains1-3_HSMs0-0.1_PRs_W_and_Init_rank{rank}.npz"
output_path = os.path.join(results_folder, filename)
# np.savez(filename, PRs=local_PRs, LLEs=local_LLEs, Seeds=local_Seeds, Bounds=local_Bounds)
np.savez(output_path, PRs=local_PRs, Gains = local_Gains, Weights_Seeds=local_Weights_Seeds, Initialisation_Seeds=local_Initialisation_Seeds, MSHs=local_MSHs)

# print(f"[Rank {rank}] Saved results to {filename}")

print(f"hello from rank {rank} of {size}")