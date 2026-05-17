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

t = np.arange(0, 1000, dt)
I = np.zeros((len(t), N))

#change running_time for test
#running_time = np.arange(0, 300+dt, dt)
running_time = np.arange(0, 3000+dt, dt)
interval = 200
reset_steps = int(interval/dt)
perturbation = 1e-8

n_weight_seeds = np.array([5, 6, 7, 8, 9])
n_init_seeds = np.array([0, 1, 2, 3])
# MSHs = np.arange(0, 5.2, 0.2)
# gains = np.arange(0, 10.2, 0.2)
MSHs = np.arange(0, 0.101, 0.001)
gains = np.arange(1, 3.2, 0.2)

#change n_sets for test
#n_seeds = 1 
# n_weight_seeds = range(0, 2)
# n_init_seeds = 10
# MSHs = np.arange(0, 5.2, 0.2)
# MSHs = np.append(MSHs, 10)
# #MSHs = np.array([0.0, 0.2])
# gains = np.arange(0, 10.2, 0.2)
#gains = np.arange(0, 0.4, 0.2)
# Generate all (i, bound) combinations
tasks = list(product(gains, n_weight_seeds, MSHs, n_init_seeds)) #26-8-2025 latest updat n_sets+10 for testing with different seeds 

# Distribute tasks among ranks
local_tasks = tasks[rank::size] #python list slicing, which distributes the tasks evenly across all ranks, 
# you can also see it as tasks[start:end:step] where start is the rank, end is the total number of tasks, and step is the number of ranks
#we didn't specify an end so it will take all the tasks from the start to the end of the list

# Local storage for results
local_LLEs = []
#local_distances = []
# local_activity_X1 = []
# local_activity_X2 = []
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

    # seed = int(i * 10)  # Convert to integer seed for reproducibility
    # initialisation_seed = int(initialisation_idx * 10)
    # rng = np.random.default_rng(np.random.SeedSequence(entropy=654321, spawn_key=(2, i_rep, i_trail))) #farhad's way, not sure how to implement
    # rng_weights = np.random.default_rng(seed)
    # rng_initialisation = np.random.default_rng(initialisation_seed)

    # Weight matrix
    W = md.weight_matrix_fast(p, N, MSH, gain, rng_weights)

    # # PR computation
    # x0 = rng.uniform(-1, 1, N)
    # x = md.euler_rate_network_numba(W, tau, dt, len(t), x0 , I, return_full = True)
    # x_final = md.euler_rate_network_numba(W, tau, dt, len(t), x0, I, return_full=False) # for if you only need the final state (so not the full trajectory)
    # x = md.euler_rate_network(md.rate_network, W, md.tanh, N, t, x0, I, tau)
    # PR = md.participation_ratio(md.tanh(x))
    # local_PRs.append(PR)

    # LLE computation
    #LLE, distances, activity_X1, activity_X2 = md.lyapunov_exponent(running_time, reset_steps, tau, perturbation, N, W, dt, rng_initialisation)
    #LLE, distances = md.lyapunov_exponent(running_time, reset_steps, tau, perturbation, N, W, dt, rng_initialisation)
    # initial state (numba version needs explicit array instead of rng inside)
    X0 = rng_initialisation.uniform(-1.0, 1.0, size = N)
    LLE = md.lyapunov_exponent_numba(running_time, reset_steps, tau, perturbation, N, W, dt, X0)
    # print(f"[Rank {rank}] LLE: {LLE:.5f}, bound: {bound}, seed: {int(i*10)}")
    local_LLEs.append(LLE)
    # local_distances.append(distances)
    # local_activity_X1.append(activity_X1)
    # local_activity_X2.append(activity_X2)

    #appending gainss, seeds and bounds for reference
    local_Gains.append(gain)
    local_Weights_Seeds.append(weights_seed)
    local_Initialisation_Seeds.append(initialisation_seed)
    local_MSHs.append(MSH)

# Each rank saves its own results file
# local_PRs = np.array(local_PRs)
local_LLEs = np.array(local_LLEs)
# local_distances = np.array(local_distances)
# local_activity_X1 = np.array(local_activity_X1)
# local_activity_X2 = np.array(local_activity_X2)
local_Weights_Seeds = np.array(local_Weights_Seeds)
local_Initialisation_Seeds = np.array(local_Initialisation_Seeds)
local_MSHs = np.array(local_MSHs)
local_Gains = np.array(local_Gains)

filename = f"Zoomed_in_gains1-3_HSMs0-0.1_LLEs_W_and_Init_rank{rank}.npz"
output_path = os.path.join(results_folder, filename)
# np.savez(filename, PRs=local_PRs, LLEs=local_LLEs, Seeds=local_Seeds, Bounds=local_Bounds)
# np.savez(output_path, LLEs=local_LLEs, Distances=local_distances, Activity_X1=local_activity_X1, Activity_X2=local_activity_X2, Weights_Seeds=local_Weights_Seeds, Initialisation_Seeds=local_Initialisation_Seeds, Bounds=local_Bounds)
np.savez(output_path, LLEs=local_LLEs, Gains = local_Gains, Weights_Seeds=local_Weights_Seeds, Initialisation_Seeds=local_Initialisation_Seeds, MSHs=local_MSHs)

# print(f"[Rank {rank}] Saved results to {filename}")

print(f"hello from rank {rank} of {size}")

# # Gather results to rank 0
# all_PRs = comm.gather(local_PRs, root=0)
# all_LLEs = comm.gather(local_LLEs, root=0)
# all_Seeds = comm.gather(local_Seeds, root=0)
# all_Bounds = comm.gather(local_Bounds, root=0)

# # Combine and save (on root only)
# if rank == 0:
#     PRs = np.array([pr for sublist in all_PRs for pr in sublist])
#     LLEs = np.array([lle for sublist in all_LLEs for lle in sublist])
#     Seeds = np.array([seed for sublist in all_Seeds for seed in sublist])
#     Bounds = np.array([bound for sublist in all_Bounds for bound in sublist])
#     print("PRs shape:", PRs.shape, "LLEs shape:", LLEs.shape)
#     np.savez('results.npz', PRs=PRs, LLEs=LLEs, Seeds=Seeds, Bounds=Bounds)
