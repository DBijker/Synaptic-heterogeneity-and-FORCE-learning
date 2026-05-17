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
results_folder = "Results_Activities_LLEs_allranks_10_times_sigma_different_initial_conditions"

#only rank 0 creates the folder
if rank == 0:
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

#waiting for rank 0 to finish creating the directory
comm.Barrier() # Ensure all ranks wait until the directory is created

# Parameters
N = 1000
tau = 10
dt = 0.1

t = np.arange(0, 1000, dt)
I = np.zeros((len(t), N))

#change running_time for test
#running_time = np.arange(0, 300+dt, dt)
running_time = np.arange(0, 30000+dt, dt)
interval = 200
reset_steps = int(interval/dt)
perturbation = 1e-8

#change n_sets for test
n_sets = 4
#n_sets = 10
# bounds = np.linspace(0, 1, n_sets+1)
bounds = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0])
seeds_linspace = np.linspace(0, 1, n_sets+1)


# Generate all (i, bound) combinations
tasks = list(product(range(n_sets+1), bounds, range(n_sets+1)))

# Distribute tasks among ranks
local_tasks = tasks[rank::size] #python list slicing, which distributes the tasks evenly across all ranks, 
# you can also see it as tasks[start:end:step] where start is the rank, end is the total number of tasks, and step is the number of ranks
#we didn't specify an end so it will take all the tasks from the start to the end of the list

# Local storage for results
local_activity_X1 = []
local_activity_X2 = []
local_Weights_Seeds = []
local_Initialisation_Seeds = []
local_Bounds = []

# Perform local computation
for weights_seed, bound, initialisation_seed in local_tasks:

    weights_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, weights_seed))# entropy = parents seed, spawn_key=(2,3,0) means spawn_key=(#experiment, #repetition, #trial) or trial 0 of rep 3 of exp 2. Creates for loop over trials
    initialisation_seq = np.random.SeedSequence(entropy=654321, spawn_key=(0, weights_seed, initialisation_seed))

    rng_weights = np.random.default_rng(weights_seq)
    rng_initialisation = np.random.default_rng(initialisation_seq)

    # Weight matrix
    W = md.weight_matrix(N, bound, rng_weights)

    # activities computation
    activity_X1, activity_X2 = md.activities(running_time, reset_steps, tau, perturbation, N, W, dt, rng_initialisation)
    # print(f"[Rank {rank}] LLE: {LLE:.5f}, bound: {bound}, seed: {int(i*10)}")
    local_activity_X1.append(activity_X1)
    local_activity_X2.append(activity_X2)

    #appending seeds and bounds for reference
    local_Weights_Seeds.append(weights_seed)
    local_Initialisation_Seeds.append(initialisation_seed)
    local_Bounds.append(bound)

# Each rank saves its own results file
# local_PRs = np.array(local_PRs)
local_activity_X1 = np.array(local_activity_X1)
local_activity_X2 = np.array(local_activity_X2)
local_Weights_Seeds = np.array(local_Weights_Seeds)
local_Initialisation_Seeds = np.array(local_Initialisation_Seeds)
local_Bounds = np.array(local_Bounds)

filename = f"10_times_sigma_Activities_LLEs_results_different_initial_conditions_rank{rank}.npz"
output_path = os.path.join(results_folder, filename)

np.savez(output_path, Activity_X1=local_activity_X1, Activity_X2=local_activity_X2, Weights_Seeds=local_Weights_Seeds, Initialisation_Seeds=local_Initialisation_Seeds, Bounds=local_Bounds)


print(f"hello from rank {rank} of {size}")
