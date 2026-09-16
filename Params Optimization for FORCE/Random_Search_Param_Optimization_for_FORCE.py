import numpy as np
import FORCE_module as FC
from mpi4py import MPI
from itertools import product
import os
import datetime
import glob

# ==== MPI setup ====
comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # this process's rank, 0..size-1
size = comm.Get_size()   # total number of parallel MPI processes

# Shared folder where every rank writes its per-task .npz results
results_folder = "FORCE_Gains_HSMs_100randoms_periodic_1seed_W_and_Init"

# Only rank 0 creates the output folder (avoids every rank racing to create it)
if rank == 0:
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

comm.Barrier()  # all ranks wait here until rank 0 has finished creating the folder

# ==== Parameter sweep ranges ====

# Number of weight-initialization seeds to run per (gain, MSH) combo.
# Currently a single seed (weights_seed = 1) -- matches the "_1seed_" in
# results_folder above.
n_weight_seeds = range(1, 2)

# Mean-shift heterogeneity (HSM) values to sweep
MSHs = np.array([4, 5])

# Recurrent gain values to sweep
gains = np.array([0, 0.5, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0,
                   3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

# Every (gain, weights_seed, MSH) combination to run
tasks = list(product(gains, n_weight_seeds, MSHs))

# Split tasks across MPI ranks: rank r gets tasks[r], tasks[r+size], tasks[r+2*size], ...
local_tasks = tasks[rank::size]

# Hyperparameter ranges passed to FC.random_search for each task.
# Only 'eta' is currently searched; 'Dt' search is disabled below, so
# FC.random_search / FORCETrainer fall back to their default Dt=2.0.
param_ranges = {
    'eta': (0.01, 1),
    # 'Dt': (0.5, 20.0),
}

print(f"Rank {rank}/{size}: Processing {len(local_tasks)} tasks")

# ==== Run the random search for each local task, saving results as we go ====
for task_idx, (gain, weights_seed, MSH) in enumerate(local_tasks):
    print(f"\nRank {rank}: Starting task {task_idx+1}/{len(local_tasks)}")
    print(f"  gain={gain}, weights_seed={weights_seed}, MSH={MSH}")

    try:
        random_results = FC.random_search(
            param_ranges,
            gain=gain,
            HSM=MSH,
            n_iterations=50,
            n_trials=weights_seed,   # number of repeated trials per hyperparameter draw
            train_time=1440,
            test_time=1440,
            verbose=False
        )

        # random_results is sorted by ascending mean_test_mae, so [0] is the best config
        best_params = random_results[0]

        eta = best_params['eta']
        # 'Dt' isn't in param_ranges, so it's never in best_params; report the
        # fixed default that FC.random_search actually trained with instead.
        Dt = best_params.get('Dt', 2.0)
        mean_test_mae = best_params['mean_test_mae']
        std_test_mae = best_params['std_test_mae']

        # Microsecond timestamp keeps filenames unique even for repeated (gain, MSH, seed)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = f"result_rank{rank}_gain{gain:.1f}_MSH{MSH:.1f}_seed{weights_seed}_{timestamp}.npz"
        output_path = os.path.join(results_folder, filename)

        # Save this task's result immediately (rather than buffering), so
        # progress survives even if a later task crashes
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
        # Log the failure and keep going instead of aborting the whole sweep
        print(f"Rank {rank}: ERROR in task {task_idx+1}/{len(local_tasks)}")
        print(f"  gain={gain}, MSH={MSH}, seed={weights_seed}")
        print(f"  Error: {str(e)}")

        error_filename = f"ERROR_rank{rank}_gain{gain:.1f}_MSH{MSH:.1f}_seed{weights_seed}.txt"
        error_path = os.path.join(results_folder, error_filename)
        with open(error_path, 'w') as f:
            f.write("Error in task:\n")
            f.write(f"gain={gain}, MSH={MSH}, weights_seed={weights_seed}\n")
            f.write(f"Error: {str(e)}\n")

print(f"\nRank {rank}: All tasks completed!")

comm.Barrier()  # wait for every rank to finish its tasks before aggregating results

# ========================================================================
# ONLY RANK 0 COMBINES ALL PER-TASK RESULTS INTO ONE SORTED FILE
# ========================================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("Rank 0: Combining all results...")
    print("=" * 60)

    result_files = glob.glob(os.path.join(results_folder, "result_*.npz"))
    print(f"Found {len(result_files)} result files")

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

    all_etas = np.array(all_etas)
    all_Dts = np.array(all_Dts)
    all_gains = np.array(all_gains)
    all_MSHs = np.array(all_MSHs)
    all_seeds = np.array(all_seeds)
    all_mean_maes = np.array(all_mean_maes)
    all_std_maes = np.array(all_std_maes)

    # np.lexsort sorts by its LAST key first: primary key = gains,
    # secondary = seeds, tertiary = MSHs
    sort_idx = np.lexsort((all_MSHs, all_seeds, all_gains))

    Etas_sorted = all_etas[sort_idx]
    Dts_sorted = all_Dts[sort_idx]
    Gains_sorted = all_gains[sort_idx]
    MSHs_sorted = all_MSHs[sort_idx]
    Seeds_sorted = all_seeds[sort_idx]
    mean_maes_sorted = all_mean_maes[sort_idx]
    std_maes_sorted = all_std_maes[sort_idx]

    combined_path = os.path.join(
        results_folder,
        "FORCE_Gains_HSMs_100randoms_periodic_1seed_W_and_Init_combined_sorted.npz"
    )
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
    print("Sorted by: Gains -> Seeds -> MSHs")

    unique_gains = np.unique(Gains_sorted)
    unique_MSHs = np.unique(MSHs_sorted)
    unique_seeds = np.unique(Seeds_sorted)

    print("\nSummary:")
    print(f"  Unique gains: {len(unique_gains)} values")
    print(f"  Unique MSHs: {len(unique_MSHs)} values")
    print(f"  Unique seeds: {len(unique_seeds)} values")
    print(f"  Total combinations: {len(Etas_sorted)}")

    print("\nFirst 10 entries:")
    print(f"{'Gain':>6} {'MSH':>6} {'Seed':>4} {'eta':>8} {'Dt':>8} {'MAE':>10}")
    print("-" * 60)
    for i in range(min(10, len(Etas_sorted))):
        print(f"{Gains_sorted[i]:6.1f} {MSHs_sorted[i]:6.1f} {Seeds_sorted[i]:4d} "
              f"{Etas_sorted[i]:8.4f} {Dts_sorted[i]:8.2f} {mean_maes_sorted[i]:10.6f}")

print(f"\nRank {rank}: Job complete")