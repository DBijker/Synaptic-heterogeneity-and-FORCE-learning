import numpy as np
import FORCE_module as FC
from mpi4py import MPI
from itertools import product
import os
import datetime
import time
import glob



# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Shared folder name
results_folder = "FORCE_Gains_HSMs_100randoms_periodic_W_5-9_and_Init_1-5"

# Only rank 0 creates the folder
if rank == 0:
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

# Waiting for rank 0 to finish creating the directory
comm.Barrier()

# Fixed parameters according to Sussillo's code
learn_start = 0 ################CHECK########################
learn_start_idx = int(learn_start / dt)


Dt = 0.2
eta = 1
dt = 0.1
alpha = 1.0
tau = 1.0
N = 1000
p = 1.0


# Parameter ranges
# n_weight_seeds = range(1, 2)

#change n_sets for test
#n_seeds = 1 
n_weight_seeds = np.array([5, 6, 7, 8, 9])
n_init_seeds = np.array([1, 2, 3, 4, 5])
MSHs = np.arange(0, 5.2, 0.2)
#MSHs = np.arange(0, 0.6, 0.2)
gains = np.arange(0, 10.2, 0.2)



tasks = list(product(gains, n_weight_seeds, MSHs, n_init_seeds))

# Distribute tasks among ranks
local_tasks = tasks[rank::size]


print(f"Rank {rank}/{size}: Processing {len(local_tasks)} tasks")


# Storage for first-period test Rsquareds (across all configurations)
all_first_period_test_Rsquared = []
all_first_period_training_Rsquared = []

differences_after_training_Rsquared = []
differences_during_training_Rsquared = []
# Process each task and save immediately
for task_idx, (gain, weights_seed, MSH, initialisation_seed) in enumerate(local_tasks):
    print(f"\nRank {rank}: Starting task {task_idx+1}/{len(local_tasks)}")
    print(f"  gain={gain}, weights_seed={weights_seed}, MSH={MSH}")
    
    # Test best configuration
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



        #Train
        train_time = 1440 + learn_start  #######################CHECK THIS############################
        running_time = np.arange(0, train_time, dt)
        target_train = FC.create_complex_target(running_time, dt)
        train_zs = trainer.train(target_train, running_time, learn_start, verbose=True)

        #Test
        after_training_time = 120  # Can be changed to 4000, 6000, etc. #######################CHECK THIS############################
        test_time_arr = np.arange(0, after_training_time, dt)
        target_test = FC.create_complex_target(test_time_arr, dt, 1/60)
        test_zs = trainer.test(target_test, test_time_arr, verbose=True)

        # def create_complex_target(running_time, dt, freq=1/60, amps=None):
        steps_per_period = 1200  # 20000 steps #######################CHECK THIS############################
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
                sum_SS_res = ((target_per_period - z_per_period)**2).sum() #sum of squares of residuals
                sum_SS_tot = ((target_per_period - np.average(target_per_period))**2).sum()#total sum of squares proportional to the variance of the data
                Rsquared = 1 - (sum_SS_res/sum_SS_tot)
                Rsquareds_train.append(Rsquared)
                Rsquareds_time_train.append(end_time_step)

        # Extract ONLY the first period test Rsquared for saving
        if len(Rsquareds_train) > 0:
            Rsquared_first_period = Rsquareds_train[0]
            Rsquared_average_training = np.mean(Rsquareds_train)
            print(f"  First period train Rsquared: {Rsquared_first_period:.6f}")
            all_first_period_training_Rsquared.append(Rsquared_first_period)

            print(f"  Average train Rsquared: {Rsquared_average_training:.6f}")
            averages_training_Rsquared.append(Rsquared_average_training)
        
        else:
            print(f"  Warning: No complete periods in test data")
            Rsquared_average_training = np.nan
            averages_training_Rsquared.append(Rsquared_average_training)
            
            Rsquared_first_period = np.nan
            all_first_period_training_Rsquared.append(Rsquared_first_period)


        # Rsquared after training (ALL periods for plotting)
        Rsquareds_test = []
        Rsquareds_time_test = []

        for idx in range(0, len(test_time_arr), steps_per_period):
            time_per_period = test_time_arr[idx:idx+steps_per_period]
            target_per_period = target_test[idx:idx+steps_per_period]
            z_per_period = test_zs['z_out'][idx:idx+steps_per_period]
            
            if len(z_per_period) == steps_per_period:
                end_time_step = time_per_period[-1]
                sum_SS_res = ((target_per_period - z_per_period)**2).sum() #sum of squares of residuals
                sum_SS_tot = ((target_per_period - np.average(target_per_period))**2).sum()#total sum of squares proportional to the variance of the data
                Rsquared = 1 - (sum_SS_res/sum_SS_tot)
                Rsquareds_test.append(Rsquared)
                Rsquareds_time_test.append(end_time_step)

        # Extract ONLY the first period test Rsquared for saving
        if len(Rsquareds_test) > 0:
            Rsquared_first_period = Rsquareds_test[0]

            print(f"  First period test Rsquared: {Rsquared_first_period:.6f}")
            all_first_period_test_Rsquared.append(Rsquared_first_period)


        else:
            print(f"  Warning: No complete periods in test data")
            
            Rsquared_first_period = np.nan
            all_first_period_test_Rsquared.append(Rsquared_first_period)


        print(f"✓ Saved to: {filename}")
        
        
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
    print("Rank 0: makin summary files for averages during training and first period performance R^2 at testing or after training")
    print("="*60)
    
    summary_filepath = os.path.join(
    r"C:\Users\Lenovo\Documents\B. Master Neurophysics\Stage\Presentations and reports and code\code\FORCE\FORCE results\Changing gain and HSM periodic target\VScode gainHSM zs vs sussillos target and Rsquareds during and after training",
    f"first_period_test_Rsquared_summary_MSHs={MSHs}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    )

    np.savez(
        summary_filepath,
        first_period_test_Rsquareds=np.array(all_first_period_test_Rsquared),
        gains=Gains,
        MSHs=MSHs,
        etas=Etas,
        Dts=Dts,
        weights_seeds=Weights_Seeds
    )

    summary_differences_during_training_filepath = os.path.join(
    r"C:\Users\Lenovo\Documents\B. Master Neurophysics\Stage\Presentations and reports and code\code\FORCE\FORCE results\Changing gain and HSM periodic target\VScode gainHSM zs vs sussillos target and Rsquareds during and after training",
    f"summary_differences_during_training_filepath_MSHs={MSHs}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    )
    np.savez(
        summary_averages_during_training_filepath,
        averages_training_Rsquareds=np.array(averages_training_Rsquared),
        gains=gains,
        MSHs=MSHs,
        etas=eta,
        Dts=Dt,
        weights_seeds= n_weight_seeds)

print(f"\nRank {rank}: Job complete")