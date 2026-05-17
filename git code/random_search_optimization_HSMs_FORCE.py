import numpy as np
import FORCE_module as FC
from mpi4py import MPI
from itertools import product
import os
import datetime
import glob

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Shared folder name
results_folder = "FORCE_Gains_HSMs_100randoms_periodic_1seed_W_and_Init"

# Only rank 0 creates the folder
if rank == 0:
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

# Waiting for rank 0 to finish creating the directory
comm.Barrier()

# Fixed parameters
N = 1000
tau = 10
dt = 0.1

# Parameter ranges
# n_weight_seeds = range(1, 2)
# gains = np.arange(0, 0.4, 0.2)
# gains = gains[-1]
# extra_gains = np.arange(1, 11, 1)
# gains = np.append(gains, extra_gains)

#MSHs = np.arange(0, 0.4, 0.2)
#MSHs = MSHs[-1]
#extra = np.arange(1, 6, 1) ##############change this###############
#extra = np.arange(1, 2, 1)
#MSHs = np.append(MSHs, extra)

#MSHs = np.arange(4,5,1)
MSHs = np.array([4, 5])

#gains = np.array([0.2, 0.5, 1.0, 1.5, 2.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
gains = np.array([0, 0.5, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

#MSHs = np.array([0.4, 0.6, 0.8, 1.2])
#MSHs = np.array([0.0])

tasks = list(product(gains, n_weight_seeds, MSHs))

# Distribute tasks among ranks
local_tasks = tasks[rank::size]

param_ranges = {
    'eta': (0.01, 1),
#    'Dt': (0.5, 20.0), #################check###############
}

print(f"Rank {rank}/{size}: Processing {len(local_tasks)} tasks")

# Process each task and save immediately
for task_idx, (gain, weights_seed, MSH) in enumerate(local_tasks):
    print(f"\nRank {rank}: Starting task {task_idx+1}/{len(local_tasks)}")
    print(f"  gain={gain}, weights_seed={weights_seed}, MSH={MSH}")
    
    try:
        # Run random search
        random_results = FC.random_search(
            param_ranges,
            gain=gain,
            HSM=MSH,
            n_iterations=50,###############check################
            n_trials=weights_seed,
            train_time=1440,###############check################
            test_time=1440,###############check################
            verbose=False
        )
        
        best_params = random_results[0]
        
        # Extract results
        eta = best_params['eta']
        # Dt = best_params['Dt']
        mean_test_mae = best_params['mean_test_mae']
        std_test_mae = best_params['std_test_mae']
        
        # Create timestamp for unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save this single result immediately
        filename = f"result_rank{rank}_gain{gain:.1f}_MSH{MSH:.1f}_seed{weights_seed}_{timestamp}.npz"
        output_path = os.path.join(results_folder, filename)
        
        np.savez(
            output_path,
            eta=eta,
            Dt=Dt,
            gain=gain,
            MSH=MSH,
            weights_seed=weights_seed,
            mean_test_mae=mean_test_mae,
            std_test_mae=std_test_mae
        )
        
        print(f"Rank {rank}: Saved task {task_idx+1}/{len(local_tasks)} to {filename}")
        print(f"  Best eta={eta:.4f}, Dt={Dt:.2f}, MAE={mean_test_mae:.6f}")
        
    except Exception as e:
        print(f"Rank {rank}: ERROR in task {task_idx+1}/{len(local_tasks)}")
        print(f"  gain={gain}, MSH={MSH}, seed={weights_seed}")
        print(f"  Error: {str(e)}")
        
        # Save error information
        error_filename = f"ERROR_rank{rank}_gain{gain:.1f}_MSH{MSH:.1f}_seed{weights_seed}.txt"
        error_path = os.path.join(results_folder, error_filename)
        with open(error_path, 'w') as f:
            f.write(f"Error in task:\n")
            f.write(f"gain={gain}, MSH={MSH}, weights_seed={weights_seed}\n")
            f.write(f"Error: {str(e)}\n")

print(f"\nRank {rank}: All tasks completed!")

# Wait for all ranks to finish
comm.Barrier()

# ========================================================================
# ONLY RANK 0 COMBINES ALL RESULTS
# ========================================================================
if rank == 0:
    print("\n" + "="*60)
    print("Rank 0: Combining all results...")
    print("="*60)
    
    # Find all result files
    result_files = glob.glob(os.path.join(results_folder, "result_*.npz"))
    
    print(f"Found {len(result_files)} result files")
    
    # Collect all results
    all_etas = []
    all_Dts = []
    all_gains = []
    all_MSHs = []
    all_seeds = []
    all_mean_maes = []
    all_std_maes = []
    
    for filepath in result_files:
        try:
            data = np.load(filepath)
            all_etas.append(float(data['eta']))
            all_Dts.append(float(data['Dt']))
            all_gains.append(float(data['gain']))
            all_MSHs.append(float(data['MSH']))
            all_seeds.append(int(data['weights_seed']))
            all_mean_maes.append(float(data['mean_test_mae']))
            all_std_maes.append(float(data['std_test_mae']))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    # Convert to arrays
    all_etas = np.array(all_etas)
    all_Dts = np.array(all_Dts)
    all_gains = np.array(all_gains)
    all_MSHs = np.array(all_MSHs)
    all_seeds = np.array(all_seeds)
    all_mean_maes = np.array(all_mean_maes)
    all_std_maes = np.array(all_std_maes)
    
    # Sort by MSH, then Gain, then Seed (for consistent ordering)
    sort_idx = np.lexsort((all_MSHs, all_seeds,  all_gains))
    
    # Apply sorting
    Etas_sorted = all_etas[sort_idx]
    Dts_sorted = all_Dts[sort_idx]
    Gains_sorted = all_gains[sort_idx]
    MSHs_sorted = all_MSHs[sort_idx]
    Seeds_sorted = all_seeds[sort_idx]
    mean_maes_sorted = all_mean_maes[sort_idx]
    std_maes_sorted = all_std_maes[sort_idx]
    
    # Save combined results
    combined_path = os.path.join(results_folder, "FORCE_Gains_HSMs_100randoms_periodic_1seed_W_and_Init_combined_sorted.npz")
    np.savez(
        combined_path,
        Etas=Etas_sorted,
        Dts=Dts_sorted,
        Gains=Gains_sorted,
        MSHs=MSHs_sorted,
        Weights_Seeds=Seeds_sorted,
        mean_maes=mean_maes_sorted,
        std_maes=std_maes_sorted
    )
    
    print(f"\nCombined {len(result_files)} results into {combined_path}")
    print(f"Sorted by: Gains → MSHs → Seeds")
    
    # Print summary
    unique_gains = np.unique(Gains_sorted)
    unique_MSHs = np.unique(MSHs_sorted)
    unique_seeds = np.unique(Seeds_sorted)
    
    print(f"\nSummary:")
    print(f"  Unique gains: {len(unique_gains)} values")
    print(f"  Unique MSHs: {len(unique_MSHs)} values")
    print(f"  Unique seeds: {len(unique_seeds)} values")
    print(f"  Total combinations: {len(Etas_sorted)}")
    
    # Create a summary table
    print(f"\nFirst 10 entries:")
    print(f"{'Gain':>6} {'MSH':>6} {'Seed':>4} {'eta':>8} {'Dt':>8} {'MAE':>10}")
    print("-" * 60)
    for i in range(min(10, len(Etas_sorted))):
        print(f"{Gains_sorted[i]:6.1f} {MSHs_sorted[i]:6.1f} {Seeds_sorted[i]:4d} "
              f"{Etas_sorted[i]:8.4f} {Dts_sorted[i]:8.2f} {mean_maes_sorted[i]:10.6f}")

print(f"\nRank {rank}: Job complete")