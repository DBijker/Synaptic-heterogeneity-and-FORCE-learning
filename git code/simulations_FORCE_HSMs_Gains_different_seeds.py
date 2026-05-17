import numpy as np
import FORCE_module as FC
from mpi4py import MPI
from itertools import product
import os
import datetime
import time
import glob
from collections import defaultdict

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Shared folder name
results_folder = "VS_Code_FORCE_Gains_1-3_HSMs_0-0.1_periodic_W_5-9_and_Init_0-3"

# Only rank 0 creates the folder
if rank == 0:
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

# Waiting for rank 0 to finish creating the directory
comm.Barrier()

# Fixed parameters according to Sussillo's code
learn_start = 0
Dt = 0.2
eta = 1
dt = 0.1
alpha = 1.0
tau = 1.0
N = 1000
p = 1.0

learn_start_idx = int(learn_start / dt)

# Parameter ranges
n_weight_seeds = np.array([5, 6, 7, 8, 9])
n_init_seeds = np.array([0, 1, 2, 3])
# MSHs = np.arange(0, 5.2, 0.2)
# gains = np.arange(0, 10.2, 0.2)
MSHs = np.arange(0, 0.101, 0.001)
gains = np.arange(1, 3.2, 0.2)

# n_weight_seeds = np.array([5])
# n_init_seeds = np.array([1])
# MSHs = np.array([0.0, 0.2, 0.4])
# gains = np.array([1.4])

# Create tasks: organize by gain first for easier per-gain saving
tasks = list(product(gains, n_weight_seeds, MSHs, n_init_seeds))

# Distribute tasks among ranks
local_tasks = tasks[rank::size]

print(f"Rank {rank}/{size}: Processing {len(local_tasks)} tasks")

# Storage for results - organized by gain
results_by_gain = defaultdict(lambda: {
    'gains': [],
    'MSHs': [],
    'weight_seeds': [],
    'init_seeds': [],
    # 'etas': [],
    # 'Dts': [],
    # 'first_period_train_Rsquared': [],
    'average_train_Rsquared': [],
    'first_period_test_Rsquared': []
})

# Track which gains have been saved already
saved_gains = set()

# Process each task
for task_idx, (gain, weights_seed, MSH, initialisation_seed) in enumerate(local_tasks):
    print(f"\nRank {rank}: Starting task {task_idx+1}/{len(local_tasks)}")
    print(f"  gain={gain}, weights_seed={weights_seed}, MSH={MSH}, init_seed={initialisation_seed}")
    
    start = time.time()

    try:
        trainer = FC.FORCETrainer(
            N=N,
            p=p,
            gain=gain,
            HSM=MSH,
            tau=tau,
            dt=dt,
            Dt=Dt,
            alpha=alpha,
            eta=eta,
            weights_seed=weights_seed,
            initialisation_seed=initialisation_seed
        )

        # Train
        train_time = 1440 + learn_start
        running_time = np.arange(0, train_time, dt)
        target_train = FC.create_complex_target(running_time, dt)
        train_zs = trainer.train(target_train, running_time, learn_start, verbose=True)

        # Test
        after_training_time = 120
        test_time_arr = np.arange(0, after_training_time, dt)
        target_test = FC.create_complex_target(test_time_arr, dt, 1/60)
        test_zs = trainer.test(target_test, test_time_arr, verbose=True)

        steps_per_period = 1200
        period_time = np.arange(learn_start, train_time, dt)

        # Rsquared during training (all periods)
        Rsquareds_train = []
        Rsquareds_time_train = []

        for idx in range(0, len(period_time), steps_per_period):
            time_per_period = running_time[idx:idx+steps_per_period]
            target_per_period = target_train[idx:idx+steps_per_period]
            z_per_period = train_zs['all_z_out'][idx:idx+steps_per_period]
            
            if len(z_per_period) == steps_per_period:
                end_time_step = time_per_period[-1]
                sum_SS_res = ((target_per_period - z_per_period)**2).sum()
                sum_SS_tot = ((target_per_period - np.average(target_per_period))**2).sum()
                Rsquared = 1 - (sum_SS_res/sum_SS_tot)
                Rsquareds_train.append(Rsquared)
                Rsquareds_time_train.append(end_time_step)

        # Calculate training metrics
        if len(Rsquareds_train) > 0:
            # Rsquared_first_period_train = Rsquareds_train[0]
            Rsquared_average_training = np.mean(Rsquareds_train)
            # print(f"  First period train Rsquared: {Rsquared_first_period_train:.6f}")
            print(f"  Average train Rsquared: {Rsquared_average_training:.6f}")
        else:
            print(f"  Warning: No complete periods in training data")
            # Rsquared_first_period_train = np.nan
            Rsquared_average_training = np.nan

        # Rsquared after training (test)
        Rsquareds_test = []
        Rsquareds_time_test = []

        for idx in range(0, len(test_time_arr), steps_per_period):
            time_per_period = test_time_arr[idx:idx+steps_per_period]
            target_per_period = target_test[idx:idx+steps_per_period]
            z_per_period = test_zs['z_out'][idx:idx+steps_per_period]
            
            if len(z_per_period) == steps_per_period:
                end_time_step = time_per_period[-1]
                sum_SS_res = ((target_per_period - z_per_period)**2).sum()
                sum_SS_tot = ((target_per_period - np.average(target_per_period))**2).sum()
                Rsquared = 1 - (sum_SS_res/sum_SS_tot)
                Rsquareds_test.append(Rsquared)
                Rsquareds_time_test.append(end_time_step)

        # Calculate test metrics
        if len(Rsquareds_test) > 0:
            Rsquared_first_period_test = Rsquareds_test[0]
            print(f"  First period test Rsquared: {Rsquared_first_period_test:.6f}")
        else:
            print(f"  Warning: No complete periods in test data")
            Rsquared_first_period_test = np.nan

        # Store results in memory (organized by gain)
        results_by_gain[gain]['gains'].append(gain)
        results_by_gain[gain]['MSHs'].append(MSH)
        results_by_gain[gain]['weight_seeds'].append(weights_seed)
        results_by_gain[gain]['init_seeds'].append(initialisation_seed)
        # results_by_gain[gain]['etas'].append(eta)
        # results_by_gain[gain]['Dts'].append(Dt)
        # results_by_gain[gain]['first_period_train_Rsquared'].append(Rsquared_first_period_train)
        results_by_gain[gain]['average_train_Rsquared'].append(Rsquared_average_training)
        results_by_gain[gain]['first_period_test_Rsquared'].append(Rsquared_first_period_test)

        elapsed = time.time() - start
        print(f"  [OK] Completed in {elapsed:.1f}s")
        
    except Exception as e:
        print(f"Rank {rank}: ERROR in task {task_idx+1}/{len(local_tasks)}")
        print(f"  gain={gain}, MSH={MSH}, w_seed={weights_seed}, i_seed={initialisation_seed}")
        print(f"  Error: {str(e)}")
        
        # Store NaN for failed tasks
        results_by_gain[gain]['gains'].append(gain)
        results_by_gain[gain]['MSHs'].append(MSH)
        results_by_gain[gain]['weight_seeds'].append(weights_seed)
        results_by_gain[gain]['init_seeds'].append(initialisation_seed)
        # results_by_gain[gain]['etas'].append(eta)
        # results_by_gain[gain]['Dts'].append(Dt)
        # results_by_gain[gain]['first_period_train_Rsquared'].append(np.nan)
        results_by_gain[gain]['average_train_Rsquared'].append(np.nan)
        results_by_gain[gain]['first_period_test_Rsquared'].append(np.nan)
        
        # Save error information
        error_filename = f"ERROR_rank{rank}_gain{gain:.1f}_MSH{MSH:.1f}_wseed{weights_seed}_iseed{initialisation_seed}.txt"
        error_path = os.path.join(results_folder, error_filename)
        with open(error_path, 'w') as f:
            f.write(f"Error in task:\n")
            f.write(f"gain={gain}, MSH={MSH}, weights_seed={weights_seed}, init_seed={initialisation_seed}\n")
            f.write(f"Error: {str(e)}\n")

    # Check if we've completed all tasks for this gain across all ranks
    # Look ahead to see if this is the last task for this gain
    remaining_gains_in_local_tasks = [g for g, _, _, _ in local_tasks[task_idx+1:]]
    
    if gain not in remaining_gains_in_local_tasks and gain not in saved_gains:
        # This rank is done with this gain - save it
        print(f"\nRank {rank}: Finished all tasks for gain={gain}, saving...")
        
        # Convert lists to arrays
        gain_data = {}
        for key, val in results_by_gain[gain].items():
            gain_data[key] = np.array(val)
        
        # Save this gain's results
        filename = f"results_rank{rank}_gain{gain:.1f}.npz"
        filepath = os.path.join(results_folder, filename)
        np.savez(filepath, **gain_data)
        
        print(f"  [OK] Saved {len(gain_data['gains'])} results for gain={gain} to {filename}")
        
        # Mark as saved and free memory
        saved_gains.add(gain)
        del results_by_gain[gain]

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
    
    # Find all result files (from all ranks)
    result_files = glob.glob(os.path.join(results_folder, "results_rank*_gain*.npz"))
    
    print(f"Found {len(result_files)} result files from all ranks")
    
    # Collect all results
    all_gains = []
    all_MSHs = []
    all_weight_seeds = []
    all_init_seeds = []
    # all_etas = []
    # all_Dts = []
    # all_first_period_train_Rsquared = []
    all_average_train_Rsquared = []
    all_first_period_test_Rsquared = []
    
    for filepath in result_files:
        try:
            data = np.load(filepath)
            all_gains.extend(data['gains'])
            all_MSHs.extend(data['MSHs'])
            all_weight_seeds.extend(data['weight_seeds'])
            all_init_seeds.extend(data['init_seeds'])
            # all_etas.extend(data['etas'])
            # all_Dts.extend(data['Dts'])
            # all_first_period_train_Rsquared.extend(data['first_period_train_Rsquared'])
            all_average_train_Rsquared.extend(data['average_train_Rsquared'])
            all_first_period_test_Rsquared.extend(data['first_period_test_Rsquared'])
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    # Convert to arrays
    all_gains = np.array(all_gains)
    all_MSHs = np.array(all_MSHs)
    all_weight_seeds = np.array(all_weight_seeds)
    all_init_seeds = np.array(all_init_seeds)
    # all_etas = np.array(all_etas)
    # all_Dts = np.array(all_Dts)
    # all_first_period_train_Rsquared = np.array(all_first_period_train_Rsquared)
    all_average_train_Rsquared = np.array(all_average_train_Rsquared)
    all_first_period_test_Rsquared = np.array(all_first_period_test_Rsquared)
    
    # Sort by gain, then MSH, then weight_seed, then init_seed
    sort_idx = np.lexsort((all_init_seeds, all_weight_seeds, all_MSHs, all_gains))
    
    # Apply sorting
    gains_sorted = all_gains[sort_idx]
    MSHs_sorted = all_MSHs[sort_idx]
    weight_seeds_sorted = all_weight_seeds[sort_idx]
    init_seeds_sorted = all_init_seeds[sort_idx]
    # etas_sorted = all_etas[sort_idx]
    # Dts_sorted = all_Dts[sort_idx]
    # first_period_train_Rsquared_sorted = all_first_period_train_Rsquared[sort_idx]
    average_train_Rsquared_sorted = all_average_train_Rsquared[sort_idx]
    first_period_test_Rsquared_sorted = all_first_period_test_Rsquared[sort_idx]
    
    # Save combined results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_filename = f"FORCE_combined_sorted_{timestamp}.npz"
    combined_path = os.path.join(results_folder, combined_filename)
    
    np.savez(
        combined_path,
        gains=gains_sorted,
        MSHs=MSHs_sorted,
        weight_seeds=weight_seeds_sorted,
        init_seeds=init_seeds_sorted,
        # etas=etas_sorted,
        # Dts=Dts_sorted,
        # first_period_train_Rsquared=first_period_train_Rsquared_sorted,
        average_train_Rsquared=average_train_Rsquared_sorted,
        first_period_test_Rsquared=first_period_test_Rsquared_sorted
    )
    
    print(f"\n[OK] Combined {len(result_files)} files into {combined_filename}")
    print(f"  Total data points: {len(gains_sorted)}")
    print(f"  Sorted by: Gain MSH Weight_Seed Init_Seed")
    
    # Print summary
    unique_gains = np.unique(gains_sorted)
    unique_MSHs = np.unique(MSHs_sorted)
    unique_weight_seeds = np.unique(weight_seeds_sorted)
    unique_init_seeds = np.unique(init_seeds_sorted)
    
    print(f"\nSummary:")
    print(f"  Unique gains: {len(unique_gains)} values")
    print(f"  Unique MSHs: {len(unique_MSHs)} values")
    print(f"  Unique weight seeds: {len(unique_weight_seeds)} values")
    print(f"  Unique init seeds: {len(unique_init_seeds)} values")
    print(f"  Expected total: {len(unique_gains) * len(unique_MSHs) * len(unique_weight_seeds) * len(unique_init_seeds)}")
    print(f"  Actual total: {len(gains_sorted)}")
    
    # Check for missing data
    if len(gains_sorted) < 33150:
        print(f"  WARNING: Missing {33150 - len(gains_sorted)} data points!")
    
    # Create a summary table
    print(f"\nFirst 10 entries:")
    print(f"{'Gain':>6} {'MSH':>6} {'WSeed':>6} {'ISeed':>6} {'Train_Average_R^2':>10} {'Test_R^2':>10}")
    print("-" * 60)
    for i in range(min(10, len(gains_sorted))):
        print(f"{gains_sorted[i]:6.1f} {MSHs_sorted[i]:6.1f} {weight_seeds_sorted[i]:6d} "
              f"{init_seeds_sorted[i]:6d} {average_train_Rsquared_sorted[i]:10.6f} {first_period_test_Rsquared_sorted[i]:10.6f}")           
    
    print(f"\nLast 10 entries:")
    print(f"{'Gain':>6} {'MSH':>6} {'WSeed':>6} {'ISeed':>6} {'Train_Average_R^2':>10} {'Test_R^2':>10}")
    print("-" * 60)
    for i in range(max(0, len(gains_sorted)-10), len(gains_sorted)):
        print(f"{gains_sorted[i]:6.1f} {MSHs_sorted[i]:6.1f} {weight_seeds_sorted[i]:6d} "
              f"{init_seeds_sorted[i]:6d} {average_train_Rsquared_sorted[i]:10.6f} {first_period_test_Rsquared_sorted[i]:10.6f}")

print(f"\nRank {rank}: Job complete")