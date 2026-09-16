import numpy as np
import FORCE_module as FC
from mpi4py import MPI
from itertools import product
import os
import datetime
import time
import glob
from collections import defaultdict

# ==== MPI setup ====
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

results_folder = "VS_Code_FORCE_Gains_1-3_HSMs_0-0.1_periodic_W_5-9_and_Init_0-3"

if rank == 0:
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

comm.Barrier()  # all ranks wait until rank 0 has created the folder

# ==== Fixed training parameters, following Sussillo's FORCE-learning setup ====
learn_start = 0
Dt = 0.2
eta = 1
dt = 0.1
alpha = 1.0
tau = 1.0
N = 1000
p = 1.0

# ==== Parameter sweep ranges ====
n_weight_seeds = np.array([5, 6, 7, 8, 9])
n_init_seeds = np.array([0, 1, 2, 3])
HSMs = np.arange(0, 0.101, 0.001)
gains = np.arange(1, 3.2, 0.2)

# Every (gain, weights_seed, HSM, init_seed) combination to run.
# Ordered gain-first so all tasks for a given gain can be saved together below.
tasks = list(product(gains, n_weight_seeds, HSMs, n_init_seeds))

local_tasks = tasks[rank::size]

print(f"Rank {rank}/{size}: Processing {len(local_tasks)} tasks")

# Results are buffered per-gain and flushed to disk as soon as a rank
# finishes its last task for that gain (keeps memory bounded and gives
# incremental progress even if a later task fails).
results_by_gain = defaultdict(lambda: {
    'gains': [],
    'HSMs': [],
    'weight_seeds': [],
    'init_seeds': [],
    'average_train_Rsquared': [],
    'first_period_test_Rsquared': []
})

saved_gains = set()  # gains this rank has already flushed to disk

for task_idx, (gain, weights_seed, HSM, initialisation_seed) in enumerate(local_tasks):
    print(f"\nRank {rank}: Starting task {task_idx+1}/{len(local_tasks)}")
    print(f"  gain={gain}, weights_seed={weights_seed}, HSM={HSM}, init_seed={initialisation_seed}")

    start = time.time()

    try:
        trainer = FC.FORCETrainer(
            N=N, p=p, gain=gain, HSM=HSM, tau=tau, dt=dt, Dt=Dt, alpha=alpha, eta=eta,
            weights_seed=weights_seed,
            initialisation_seed=initialisation_seed,
        )

        # ---- Train ----
        train_time = 1440 + learn_start
        running_time = np.arange(0, train_time, dt)
        target_train = FC.create_complex_target(running_time, dt)
        train_zs = trainer.train(target_train, running_time, learn_start, verbose=True)

        # ---- Test ----
        after_training_time = 120
        test_time_arr = np.arange(0, after_training_time, dt)
        target_test = FC.create_complex_target(test_time_arr, dt, 1 / 60)
        test_zs = trainer.test(target_test, test_time_arr, verbose=True)

        steps_per_period = 1200
        period_time = np.arange(learn_start, train_time, dt)

        # ---- R^2 across every complete period during training ----
        Rsquareds_train = []
        Rsquareds_time_train = []

        for idx in range(0, len(period_time), steps_per_period):
            time_per_period = running_time[idx:idx + steps_per_period]
            target_per_period = target_train[idx:idx + steps_per_period]
            z_per_period = train_zs['all_z_out'][idx:idx + steps_per_period]

            if len(z_per_period) == steps_per_period:  # skip a trailing partial period
                end_time_step = time_per_period[-1]
                sum_SS_res = ((target_per_period - z_per_period) ** 2).sum()
                sum_SS_tot = ((target_per_period - np.average(target_per_period)) ** 2).sum()
                Rsquared = 1 - (sum_SS_res / sum_SS_tot)
                Rsquareds_train.append(Rsquared)
                Rsquareds_time_train.append(end_time_step)

        if len(Rsquareds_train) > 0:
            Rsquared_average_training = np.mean(Rsquareds_train)
            print(f"  Average train Rsquared: {Rsquared_average_training:.6f}")
        else:
            print("  Warning: No complete periods in training data")
            Rsquared_average_training = np.nan

        # ---- R^2 across every complete period during testing ----
        Rsquareds_test = []
        Rsquareds_time_test = []

        for idx in range(0, len(test_time_arr), steps_per_period):
            time_per_period = test_time_arr[idx:idx + steps_per_period]
            target_per_period = target_test[idx:idx + steps_per_period]
            z_per_period = test_zs['z_out'][idx:idx + steps_per_period]

            if len(z_per_period) == steps_per_period:
                end_time_step = time_per_period[-1]
                sum_SS_res = ((target_per_period - z_per_period) ** 2).sum()
                sum_SS_tot = ((target_per_period - np.average(target_per_period)) ** 2).sum()
                Rsquared = 1 - (sum_SS_res / sum_SS_tot)
                Rsquareds_test.append(Rsquared)
                Rsquareds_time_test.append(end_time_step)

        if len(Rsquareds_test) > 0:
            Rsquared_first_period_test = Rsquareds_test[0]
            print(f"  First period test Rsquared: {Rsquared_first_period_test:.6f}")
        else:
            print("  Warning: No complete periods in test data")
            Rsquared_first_period_test = np.nan

        # Buffer this task's results under its gain
        results_by_gain[gain]['gains'].append(gain)
        results_by_gain[gain]['HSMs'].append(HSM)
        results_by_gain[gain]['weight_seeds'].append(weights_seed)
        results_by_gain[gain]['init_seeds'].append(initialisation_seed)
        results_by_gain[gain]['average_train_Rsquared'].append(Rsquared_average_training)
        results_by_gain[gain]['first_period_test_Rsquared'].append(Rsquared_first_period_test)

        elapsed = time.time() - start
        print(f"  [OK] Completed in {elapsed:.1f}s")

    except Exception as e:
        print(f"Rank {rank}: ERROR in task {task_idx+1}/{len(local_tasks)}")
        print(f"  gain={gain}, HSM={HSM}, w_seed={weights_seed}, i_seed={initialisation_seed}")
        print(f"  Error: {str(e)}")

        # Store NaN placeholders so a failed task doesn't shift array alignment
        results_by_gain[gain]['gains'].append(gain)
        results_by_gain[gain]['HSMs'].append(HSM)
        results_by_gain[gain]['weight_seeds'].append(weights_seed)
        results_by_gain[gain]['init_seeds'].append(initialisation_seed)
        results_by_gain[gain]['average_train_Rsquared'].append(np.nan)
        results_by_gain[gain]['first_period_test_Rsquared'].append(np.nan)

        error_filename = f"ERROR_rank{rank}_gain{gain:.1f}_HSM{HSM:.1f}_wseed{weights_seed}_iseed{initialisation_seed}.txt"
        error_path = os.path.join(results_folder, error_filename)
        with open(error_path, 'w') as f:
            f.write("Error in task:\n")
            f.write(f"gain={gain}, HSM={HSM}, weights_seed={weights_seed}, init_seed={initialisation_seed}\n")
            f.write(f"Error: {str(e)}\n")

    # If this was the last local task for this gain, flush its buffer to disk
    remaining_gains_in_local_tasks = [g for g, _, _, _ in local_tasks[task_idx + 1:]]

    if gain not in remaining_gains_in_local_tasks and gain not in saved_gains:
        print(f"\nRank {rank}: Finished all tasks for gain={gain}, saving...")

        gain_data = {key: np.array(val) for key, val in results_by_gain[gain].items()}

        filename = f"results_rank{rank}_gain{gain:.1f}.npz"
        filepath = os.path.join(results_folder, filename)
        np.savez(filepath, **gain_data)

        print(f"  [OK] Saved {len(gain_data['gains'])} results for gain={gain} to {filename}")

        saved_gains.add(gain)
        del results_by_gain[gain]  # free memory now that it's on disk

print(f"\nRank {rank}: All tasks completed!")

comm.Barrier()  # wait for every rank to finish before rank 0 aggregates results

# ========================================================================
# ONLY RANK 0 COMBINES ALL RESULTS (from every rank's saved .npz files)
# ========================================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("Rank 0: Combining all results...")
    print("=" * 60)

    result_files = glob.glob(os.path.join(results_folder, "results_rank*_gain*.npz"))
    print(f"Found {len(result_files)} result files from all ranks")

    all_gains = []
    all_HSMs = []
    all_weight_seeds = []
    all_init_seeds = []
    all_average_train_Rsquared = []
    all_first_period_test_Rsquared = []

    for filepath in result_files:
        try:
            data = np.load(filepath)
            all_gains.extend(data['gains'])
            all_HSMs.extend(data['HSMs'])
            all_weight_seeds.extend(data['weight_seeds'])
            all_init_seeds.extend(data['init_seeds'])
            all_average_train_Rsquared.extend(data['average_train_Rsquared'])
            all_first_period_test_Rsquared.extend(data['first_period_test_Rsquared'])
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    all_gains = np.array(all_gains)
    all_HSMs = np.array(all_HSMs)
    all_weight_seeds = np.array(all_weight_seeds)
    all_init_seeds = np.array(all_init_seeds)
    all_average_train_Rsquared = np.array(all_average_train_Rsquared)
    all_first_period_test_Rsquared = np.array(all_first_period_test_Rsquared)

    # np.lexsort sorts by its LAST key first: primary=gain, then HSM, then
    # weight_seed, then init_seed
    sort_idx = np.lexsort((all_init_seeds, all_weight_seeds, all_HSMs, all_gains))

    gains_sorted = all_gains[sort_idx]
    HSMs_sorted = all_HSMs[sort_idx]
    weight_seeds_sorted = all_weight_seeds[sort_idx]
    init_seeds_sorted = all_init_seeds[sort_idx]
    average_train_Rsquared_sorted = all_average_train_Rsquared[sort_idx]
    first_period_test_Rsquared_sorted = all_first_period_test_Rsquared[sort_idx]

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_filename = f"FORCE_combined_sorted_{timestamp}.npz"
    combined_path = os.path.join(results_folder, combined_filename)

    np.savez(
        combined_path,
        gains=gains_sorted,
        HSMs=HSMs_sorted,
        weight_seeds=weight_seeds_sorted,
        init_seeds=init_seeds_sorted,
        average_train_Rsquared=average_train_Rsquared_sorted,
        first_period_test_Rsquared=first_period_test_Rsquared_sorted,
    )

    print(f"\n[OK] Combined {len(result_files)} files into {combined_filename}")
    print(f"  Total data points: {len(gains_sorted)}")
    print("  Sorted by: Gain -> HSM -> Weight_Seed -> Init_Seed")

    unique_gains = np.unique(gains_sorted)
    unique_HSMs = np.unique(HSMs_sorted)
    unique_weight_seeds = np.unique(weight_seeds_sorted)
    unique_init_seeds = np.unique(init_seeds_sorted)

    # Expected count derived from the actual sweep sizes, rather than a
    # hardcoded number, so this check stays correct if the ranges above change.
    # (The original had a hardcoded `33150` here, which would silently go
    # stale -- and therefore give a wrong warning -- whenever the sweep
    # ranges were edited.)
    expected_total = len(unique_gains) * len(unique_HSMs) * len(unique_weight_seeds) * len(unique_init_seeds)

    print("\nSummary:")
    print(f"  Unique gains: {len(unique_gains)} values")
    print(f"  Unique HSMs: {len(unique_HSMs)} values")
    print(f"  Unique weight seeds: {len(unique_weight_seeds)} values")
    print(f"  Unique init seeds: {len(unique_init_seeds)} values")
    print(f"  Expected total: {expected_total}")
    print(f"  Actual total: {len(gains_sorted)}")

    if len(gains_sorted) < expected_total:
        print(f"  WARNING: Missing {expected_total - len(gains_sorted)} data points!")

    print("\nFirst 10 entries:")
    print(f"{'Gain':>6} {'HSM':>6} {'WSeed':>6} {'ISeed':>6} {'Train_Average_R^2':>10} {'Test_R^2':>10}")
    print("-" * 60)
    for i in range(min(10, len(gains_sorted))):
        print(f"{gains_sorted[i]:6.1f} {HSMs_sorted[i]:6.1f} {weight_seeds_sorted[i]:6d} "
              f"{init_seeds_sorted[i]:6d} {average_train_Rsquared_sorted[i]:10.6f} {first_period_test_Rsquared_sorted[i]:10.6f}")

    print("\nLast 10 entries:")
    print(f"{'Gain':>6} {'HSM':>6} {'WSeed':>6} {'ISeed':>6} {'Train_Average_R^2':>10} {'Test_R^2':>10}")
    print("-" * 60)
    for i in range(max(0, len(gains_sorted) - 10), len(gains_sorted)):
        print(f"{gains_sorted[i]:6.1f} {HSMs_sorted[i]:6.1f} {weight_seeds_sorted[i]:6d} "
              f"{init_seeds_sorted[i]:6d} {average_train_Rsquared_sorted[i]:10.6f} {first_period_test_Rsquared_sorted[i]:10.6f}")

print(f"\nRank {rank}: Job complete")
