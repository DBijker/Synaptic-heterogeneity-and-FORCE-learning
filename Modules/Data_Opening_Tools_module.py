#################### Imports ####################
import numpy as np
import os
import glob
import datetime

# Data-opening / combining / sorting tools


def combined_sorted_PRs(foldername):
    """
    Load and combine Participation Ratio results (PRs, Seeds, HSMs) from a
    list of .npz files, sort by (Seeds, HSMs), and save the combined result.

    This is a simpler/older variant than gains_combined_sorted_PRs below
    (no Gains/Weights_Seeds/Initialisation_Seeds tracking) -- currently
    unused in either notebook, kept for reference.

    Args:
        foldername: list of .npz file paths (e.g. from glob.glob(...))

    Returns:
        (PRs_sorted, Seeds_sorted, HSMs_sorted)
    """
    all_PRs = []
    all_Seeds = []
    all_HSMs = []

    for file in foldername:
        data = np.load(file)
        all_PRs.extend(data["PRs"])
        all_Seeds.extend(data["Seeds"])
        all_HSMs.extend(_read_hsm_field(data, file))
        print(f"Loaded {file}: {len(data['PRs'])} entries")

    PRs = np.array(all_PRs)
    Seeds = np.array(all_Seeds)
    HSMs = np.array(all_HSMs)

    np.savez("PRs_results_combined.npz", PRs=PRs, Seeds=Seeds, HSMs=HSMs)
    print("Saved combined results to 'PRs_results_combined.npz'")

    sort_idx = np.lexsort((HSMs, Seeds))

    PRs_sorted = PRs[sort_idx]
    Seeds_sorted = Seeds[sort_idx]
    HSMs_sorted = HSMs[sort_idx]

    return PRs_sorted, Seeds_sorted, HSMs_sorted


def extended_combined_sorted_LLEs(foldername):
    """
    Load and combine Lyapunov-exponent results, including distances and
    sampled activity trajectories, from a list of .npz files, sort by
    (Weights_Seeds, Initialisation_Seeds, HSMs), and save the combined result.

    Currently unused in either notebook, kept for reference.

    Args:
        foldername: list of .npz file paths

    Returns:
        (Distances_sorted, LLEs_sorted, Activity_X1_sorted, Activity_X2_sorted,
         Weights_Seeds_sorted, Initialisation_Seeds_sorted, HSMs_sorted)
    """
    all_distances = []
    all_LLEs = []
    all_activity_X1 = []
    all_activity_X2 = []
    all_Weights_Seeds = []
    all_Initialisation_Seeds = []
    all_HSMs = []

    for file in foldername:
        data = np.load(file)
        all_distances.extend(data["Distances"])
        all_LLEs.extend(data["LLEs"])
        all_activity_X1.extend(data["Activity_X1"])
        all_activity_X2.extend(data["Activity_X2"])
        all_Weights_Seeds.extend(data["Weights_Seeds"])
        all_Initialisation_Seeds.extend(data["Initialisation_Seeds"])
        all_HSMs.extend(_read_hsm_field(data, file))
        print(f"Loaded {file}: {len(data['LLEs'])} entries")

    Distances = np.array(all_distances)
    LLEs = np.array(all_LLEs)
    Activity_X1 = np.array(all_activity_X1)
    Activity_X2 = np.array(all_activity_X2)
    Weights_Seeds = np.array(all_Weights_Seeds)
    Initialisation_Seeds = np.array(all_Initialisation_Seeds)
    HSMs = np.array(all_HSMs)

    np.savez(
        "extended_LLEs_results_combined.npz",
        LLEs=LLEs, Distances=Distances, Activity_X1=Activity_X1, Activity_X2=Activity_X2,
        Weights_Seeds=Weights_Seeds, Initialisation_Seeds=Initialisation_Seeds, HSMs=HSMs,
    )
    print("Saved combined results to 'extended_LLEs_results_combined.npz'")

    sort_idx = np.lexsort((HSMs, Initialisation_Seeds, Weights_Seeds))

    Distances_sorted = Distances[sort_idx]
    LLEs_sorted = LLEs[sort_idx]
    Activity_X1_sorted = Activity_X1[sort_idx]
    Activity_X2_sorted = Activity_X2[sort_idx]
    Weights_Seeds_sorted = Weights_Seeds[sort_idx]
    Initialisation_Seeds_sorted = Initialisation_Seeds[sort_idx]
    HSMs_sorted = HSMs[sort_idx]

    return (Distances_sorted, LLEs_sorted, Activity_X1_sorted, Activity_X2_sorted,
            Weights_Seeds_sorted, Initialisation_Seeds_sorted, HSMs_sorted)


def combined_sorted_LLEs(foldername):
    """
    Load and combine Lyapunov-exponent results (LLEs, distances, Seeds,
    HSMs only -- no Gains) from a list of .npz files, sort by (Seeds, HSMs),
    and save the combined result.

    Currently unused in either notebook, kept for reference.

    Args:
        foldername: list of .npz file paths

    Returns:
        (Distances_sorted, LLEs_sorted, Seeds_sorted, HSMs_sorted)
    """
    all_distances = []
    all_LLEs = []
    all_Seeds = []
    all_HSMs = []

    for file in foldername:
        data = np.load(file)
        all_distances.extend(data["Distances"])
        all_LLEs.extend(data["LLEs"])
        all_Seeds.extend(data["Seeds"])
        all_HSMs.extend(_read_hsm_field(data, file))
        print(f"Loaded {file}: {len(data['LLEs'])} entries")

    Distances = np.array(all_distances)
    LLEs = np.array(all_LLEs)
    Seeds = np.array(all_Seeds)
    HSMs = np.array(all_HSMs)

    np.savez("LLEs_results_combined.npz", LLEs=LLEs, Distances=Distances, Seeds=Seeds, HSMs=HSMs)
    print("Saved combined results to 'LLEs_results_combined.npz'")

    sort_idx = np.lexsort((HSMs, Seeds))

    Distances_sorted = Distances[sort_idx]
    LLEs_sorted = LLEs[sort_idx]
    Seeds_sorted = Seeds[sort_idx]
    HSMs_sorted = HSMs[sort_idx]

    return Distances_sorted, LLEs_sorted, Seeds_sorted, HSMs_sorted


def gains_combined_sorted_LLEs(foldername):
    """
    Load and combine Lyapunov-exponent results (LLEs, Gains, Weights_Seeds,
    Initialisation_Seeds, HSMs) from a list of .npz files, sort by
    (Gains, Weights_Seeds, Initialisation_Seeds, HSMs), and save the
    combined result. This is the function actually used in the LLE notebook.

    Args:
        foldername: list of .npz file paths

    Returns:
        (LLEs_sorted, Gains_sorted, Weights_Seeds_sorted,
         Initialisation_Seeds_sorted, HSMs_sorted)
    """
    all_LLEs = []
    all_Gains = []
    all_Weights_Seeds = []
    all_Initialisation_Seeds = []
    all_HSMs = []

    for file in foldername:
        data = np.load(file)
        all_LLEs.extend(data["LLEs"])
        all_Gains.extend(data["Gains"])
        all_Weights_Seeds.extend(data["Weights_Seeds"])
        all_Initialisation_Seeds.extend(data["Initialisation_Seeds"])
        all_HSMs.extend(_read_hsm_field(data, file))
        print(f"Loaded {file}: {len(data['LLEs'])} entries")

    LLEs = np.array(all_LLEs)
    Gains = np.array(all_Gains)
    Weights_Seeds = np.array(all_Weights_Seeds)
    Initialisation_Seeds = np.array(all_Initialisation_Seeds)
    HSMs = np.array(all_HSMs)

    np.savez(
        "gains_LLEs_results_combined.npz",
        LLEs=LLEs, Gains=Gains, Weights_Seeds=Weights_Seeds,
        Initialisation_Seeds=Initialisation_Seeds, HSMs=HSMs,
    )
    print("Saved combined results to 'gains_LLEs_results_combined.npz'")

    sort_idx = np.lexsort((HSMs, Initialisation_Seeds, Weights_Seeds, Gains))

    LLEs_sorted = LLEs[sort_idx]
    Gains_sorted = Gains[sort_idx]
    Weights_Seeds_sorted = Weights_Seeds[sort_idx]
    Initialisation_Seeds_sorted = Initialisation_Seeds[sort_idx]
    HSMs_sorted = HSMs[sort_idx]

    return LLEs_sorted, Gains_sorted, Weights_Seeds_sorted, Initialisation_Seeds_sorted, HSMs_sorted


def gains_combined_sorted_PRs(foldername):
    """
    Load and combine Participation Ratio results (PRs, Gains, Weights_Seeds,
    Initialisation_Seeds, HSMs) from a list of .npz files, sort by
    (Gains, Weights_Seeds, Initialisation_Seeds, HSMs), and save the
    combined result. This is the function actually used in the PR notebook.

    Args:
        foldername: list of .npz file paths

    Returns:
        (PRs_sorted, Gains_sorted, Weights_Seeds_sorted,
         Initialisation_Seeds_sorted, HSMs_sorted)
    """
    all_PRs = []
    all_Gains = []
    all_Weights_Seeds = []
    all_Initialisation_Seeds = []
    all_HSMs = []

    for file in foldername:
        data = np.load(file)
        all_PRs.extend(data["PRs"])
        all_Gains.extend(data["Gains"])
        all_Weights_Seeds.extend(data["Weights_Seeds"])
        all_Initialisation_Seeds.extend(data["Initialisation_Seeds"])
        all_HSMs.extend(_read_hsm_field(data, file))
        print(f"Loaded {file}: {len(data['PRs'])} entries")

    PRs = np.array(all_PRs)
    Gains = np.array(all_Gains)
    Weights_Seeds = np.array(all_Weights_Seeds)
    Initialisation_Seeds = np.array(all_Initialisation_Seeds)
    HSMs = np.array(all_HSMs)

    np.savez(
        "gains_PRs_results_combined.npz",
        PRs=PRs, Gains=Gains, Weights_Seeds=Weights_Seeds,
        Initialisation_Seeds=Initialisation_Seeds, HSMs=HSMs,
    )
    print("Saved combined results to 'gains_PRs_results_combined.npz'")

    sort_idx = np.lexsort((HSMs, Initialisation_Seeds, Weights_Seeds, Gains))

    PRs_sorted = PRs[sort_idx]
    Gains_sorted = Gains[sort_idx]
    Weights_Seeds_sorted = Weights_Seeds[sort_idx]
    Initialisation_Seeds_sorted = Initialisation_Seeds[sort_idx]
    HSMs_sorted = HSMs[sort_idx]

    return PRs_sorted, Gains_sorted, Weights_Seeds_sorted, Initialisation_Seeds_sorted, HSMs_sorted


# ---------------------------------------------------------------------------
# ADDED FROM: FORCE_RLS_implementation_for_RNN_git.ipynb
#
# These load/combine/sort FORCE hyperparameter-search results (Etas, Dts,
# Gains, mean/std test MAE, and individual training run outputs like
# J_initial/J_trained) -- a different kind of result than the PR/LLE
# loaders above, so these are genuinely new rather than duplicates.

def gains_combined_sorted_FORCE_optimized(foldername):
    """
    Load and combine FORCE hyperparameter-search results (Etas, Dts, Gains,
    mean/std test MAE, Weights_Seeds) from a list of .npz files, sort by
    (Gains, Weights_Seeds), and save the combined result.

    FIX applied: previously looped over a global `files` instead of its own
    `foldername` parameter (same bug class as the PR/LLE loaders above).

    Args:
        foldername: list of .npz file paths

    Returns:
        (Etas_sorted, Dts_sorted, Gains_sorted, mean_mae_sorted,
         std_mae_sorted, Weights_Seeds_sorted)
    """
    all_Etas = []
    all_Dts = []
    all_Gains = []
    all_mean_mae = []
    all_std_mae = []
    all_Weights_Seeds = []

    for file in foldername:
        data = np.load(file)
        all_Etas.extend(data["Etas"])
        all_Dts.extend(data["Dts"])
        all_Gains.extend(data["Gains"])
        all_mean_mae.extend(data["mean_mae"])
        all_std_mae.extend(data["std_mae"])
        all_Weights_Seeds.extend(data["Weights_Seeds"])
        print(f"Loaded {file}: {len(data['Etas'])} entries")

    Etas = np.array(all_Etas)
    Dts = np.array(all_Dts)
    Gains = np.array(all_Gains)
    mean_mae = np.array(all_mean_mae)
    std_mae = np.array(all_std_mae)
    Weights_Seeds = np.array(all_Weights_Seeds)

    np.savez(
        "FORCE_optimized_results_combined.npz",
        Etas=Etas, Dts=Dts, Gains=Gains, mean_mae=mean_mae, std_mae=std_mae,
        Weights_Seeds=Weights_Seeds,
    )
    print("Saved combined results to 'FORCE_optimized_results_combined.npz'")

    sort_idx = np.lexsort((Weights_Seeds, Gains))

    Etas_sorted = Etas[sort_idx]
    Dts_sorted = Dts[sort_idx]
    Gains_sorted = Gains[sort_idx]
    mean_mae_sorted = mean_mae[sort_idx]
    std_mae_sorted = std_mae[sort_idx]
    Weights_Seeds_sorted = Weights_Seeds[sort_idx]

    return Etas_sorted, Dts_sorted, Gains_sorted, mean_mae_sorted, std_mae_sorted, Weights_Seeds_sorted


def manually_combine_individual_results(foldername):
    """
    Manually combine individual result_*.npz files (use this if a cluster
    run didn't auto-combine its per-task results). Looks for
    "result_*.npz" inside `foldername`, reads each one's eta/Dt/gain/HSM/
    weights_seed/mean_test_mae/std_test_mae, sorts by (Gain, HSM, seed),
    and saves + returns the combined result.

    Args:
        foldername: directory containing the individual result_*.npz files

    Returns:
        dict with keys 'Etas', 'Dts', 'Gains', 'HSMs', 'Weights_Seeds',
        'mean_maes', 'std_maes' -- or None if no files were found.
    """
    pattern = os.path.join(foldername, "result_*.npz")
    result_files = glob.glob(pattern)

    print(f"Found {len(result_files)} result files in {foldername}")

    if len(result_files) == 0:
        print("No result files found!")
        return None

    all_etas = []
    all_Dts = []
    all_gains = []
    all_HSMs = []
    all_seeds = []
    all_mean_maes = []
    all_std_maes = []

    for filepath in result_files:
        try:
            data = np.load(filepath)
            all_etas.append(float(data['eta']))
            all_Dts.append(float(data['Dt']))
            all_gains.append(float(data['gain']))
            all_HSMs.append(float(data['HSM']))
            all_seeds.append(int(data['weights_seed']))
            all_mean_maes.append(float(data['mean_test_mae']))
            all_std_maes.append(float(data['std_test_mae']))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    all_etas = np.array(all_etas)
    all_Dts = np.array(all_Dts)
    all_gains = np.array(all_gains)
    all_HSMs = np.array(all_HSMs)
    all_seeds = np.array(all_seeds)
    all_mean_maes = np.array(all_mean_maes)
    all_std_maes = np.array(all_std_maes)

    # Sort by Gain -> HSM -> Seed
    sort_idx = np.lexsort((all_seeds, all_HSMs, all_gains))

    results = {
        'Etas': all_etas[sort_idx],
        'Dts': all_Dts[sort_idx],
        'Gains': all_gains[sort_idx],
        'HSMs': all_HSMs[sort_idx],
        'Weights_Seeds': all_seeds[sort_idx],
        'mean_maes': all_mean_maes[sort_idx],
        'std_maes': all_std_maes[sort_idx],
    }

    output_path = os.path.join(foldername, "results_combined_sorted.npz")
    np.savez(output_path, **results)
    print(f"Saved combined results to {output_path}")

    return results


def load_combined_FORCE_results(foldername):
    """
    Load the already-combined and sorted results_combined_sorted.npz file
    from `foldername` (as produced by manually_combine_individual_results
    above).

    FIX applied: previously raised `FileError` when the file was missing,
    but `FileError` isn't a real Python exception -- that branch would
    itself have raised NameError instead of a clear "file not found"
    message. Changed to the built-in FileNotFoundError.

    Args:
        foldername: folder containing results_combined_sorted.npz

    Returns:
        dict with keys 'Etas', 'Dts', 'Gains', 'HSMs', 'Weights_Seeds',
        'mean_maes', 'std_maes'
    """
    filepath = os.path.join(foldername, "results_combined_sorted.npz")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Combined results file not found: {filepath}")

    data = np.load(filepath)

    results = {
        'Etas': data['Etas'],
        'Dts': data['Dts'],
        'Gains': data['Gains'],
        'HSMs': data['HSMs'],
        'Weights_Seeds': data['Weights_Seeds'],
        'mean_maes': data['mean_maes'],
        'std_maes': data['std_maes'],
    }

    print(f"Loaded {len(results['Etas'])} entries from {filepath}")
    print(f"  Gains range: {results['Gains'].min():.1f} - {results['Gains'].max():.1f}")
    print(f"  HSMs range: {results['HSMs'].min():.1f} - {results['HSMs'].max():.1f}")

    return results


def merge_summary_files(folder_path, filename, name_Rsquareds_type, output_path=None, output_filename=None):
    """
    Merge multiple summary .npz files (each holding an R^2-type metric per
    (gain, HSM, weight_seed) configuration), reconstruct the parameter
    combinations, remove duplicate configurations, sort, and save the merged
    result.

    Args:
        folder_path: folder containing the summary files
        filename: glob pattern for the summary files, e.g.
                  "first_period_test_Rsquared_summary_*.npz"
        name_Rsquareds_type: the .npz key holding the R^2 values in each
                              file, e.g. 'first_period_test_Rsquareds'
        output_path: directory to save the merged file to (defaults to a
                     hardcoded path from the original notebook if not given
                     -- pass this explicitly rather than relying on that
                     default)
        output_filename: name for the merged output file (defaults to a
                          timestamp-based name if not given)

    Returns:
        str: path to the merged summary file
    """
    if output_path is None:
        output_path = (
            r"C:\Users\Lenovo\Documents\B. Master Neurophysics\Stage\Presentations and reports and code"
            r"\code\FORCE\FORCE results\Changing gain and HSM periodic target\Summary files VScode\merged files"
        )

    pattern = os.path.join(folder_path, filename)
    summary_files = glob.glob(pattern)

    if len(summary_files) == 0:
        raise ValueError(f"No summary files found in {folder_path}")

    print(f"Found {len(summary_files)} summary files to merge:")
    for f in summary_files:
        print(f"  - {os.path.basename(f)}")

    all_Rsquareds = []
    all_gains = []
    all_HSMs = []
    all_weights_seeds = []

    # eta/Dt are assumed constant across all files being merged
    etas_value = None
    Dts_value = None

    for filepath in summary_files:
        data = np.load(filepath)

        file_gains = data['gains']
        file_HSMs = data['HSMs']
        file_weights_seeds = data['weights_seeds']
        file_Rsquareds = data[name_Rsquareds_type]

        if etas_value is None:
            etas_value = data['etas']
            Dts_value = data['Dts']

        # Reconstruct the parameter combinations. This assumes
        # file_Rsquareds is already flattened in the same nested order as
        # this loop (gain-outer, weight_seed-middle, HSM-inner) -- i.e.
        # matching product(gains, weight_seeds, HSMs) from whatever
        # produced the original per-run files.
        for gain in file_gains:
            for weight_seed in file_weights_seeds:
                for HSM in file_HSMs:
                    all_gains.append(gain)
                    all_weights_seeds.append(weight_seed)
                    all_HSMs.append(HSM)

        all_Rsquareds.extend(file_Rsquareds)

    all_Rsquareds = np.array(all_Rsquareds)
    all_gains = np.array(all_gains)
    all_HSMs = np.array(all_HSMs)
    all_weights_seeds = np.array(all_weights_seeds)

    print(f"\nTotal configurations before deduplication: {len(all_Rsquareds)}")

    # Remove duplicates based on (gain, HSM, weight_seed), keeping the first
    # occurrence of each unique configuration
    unique_configs = {}
    for i in range(len(all_gains)):
        config_key = (round(all_gains[i], 6), round(all_HSMs[i], 6), int(all_weights_seeds[i]))
        if config_key not in unique_configs:
            unique_configs[config_key] = i

    unique_indices = sorted(unique_configs.values())

    duplicate_count = len(all_Rsquareds) - len(unique_indices)
    if duplicate_count > 0:
        print(f"  Removed {duplicate_count} duplicate configurations")

    all_Rsquareds = all_Rsquareds[unique_indices]
    all_gains = all_gains[unique_indices]
    all_HSMs = all_HSMs[unique_indices]
    all_weights_seeds = all_weights_seeds[unique_indices]

    print(f"Total configurations after deduplication: {len(all_Rsquareds)}")

    sort_indices = np.lexsort((all_weights_seeds, all_HSMs, all_gains))

    all_Rsquareds_sorted = all_Rsquareds[sort_indices]
    all_gains_sorted = all_gains[sort_indices]
    all_HSMs_sorted = all_HSMs[sort_indices]
    all_weights_seeds_sorted = all_weights_seeds[sort_indices]

    print("\nMerged data summary:")
    print(f"  Total unique configurations: {len(all_Rsquareds_sorted)}")
    print(f"  Unique gains: {len(np.unique(all_gains_sorted))} values from "
          f"{np.min(all_gains_sorted):.2f} to {np.max(all_gains_sorted):.2f}")
    print(f"  Unique HSMs: {len(np.unique(all_HSMs_sorted))} values from "
          f"{np.min(all_HSMs_sorted):.2f} to {np.max(all_HSMs_sorted):.2f}")
    print(f"  Unique etas: {etas_value}")
    print(f"  Unique Dts: {Dts_value}")
    print(f"  Unique weight_seeds: {np.unique(all_weights_seeds_sorted)}")

    if output_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{name_Rsquareds_type}_merged_{timestamp}.npz"

    output_filepath = os.path.join(output_path, output_filename)

    np.savez(
        output_filepath,
        **{name_Rsquareds_type: all_Rsquareds_sorted},
        gains=all_gains_sorted,
        HSMs=all_HSMs_sorted,
        etas=etas_value,
        Dts=Dts_value,
        weights_seeds=all_weights_seeds_sorted,
    )

    print(f"\nSaved merged summary to: {output_filepath}")

    print("\nFirst 5 configurations (sorted by gain, then HSM, then weight_seed):")
    for i in range(min(5, len(all_gains_sorted))):
        print(f"  G={all_gains_sorted[i]:.3f}, HSM={all_HSMs_sorted[i]:.3f}, "
              f"W={all_weights_seeds_sorted[i]}, R2={all_Rsquareds_sorted[i]:.6f}")

    print("\nLast 5 configurations:")
    for i in range(max(0, len(all_gains_sorted) - 5), len(all_gains_sorted)):
        print(f"  G={all_gains_sorted[i]:.3f}, HSM={all_HSMs_sorted[i]:.3f}, "
              f"W={all_weights_seeds_sorted[i]}, R2={all_Rsquareds_sorted[i]:.6f}")

    return output_filepath


def create_averages_performance_files(folder_path, HSMs, gains, Dt=0.2, eta=1,
                                       n_weight_seeds=np.array([9]),
                                       output_path=None, output_filename=None):
    """
    Read individual idx*.npz training-run files from `folder_path` and merge
    them into two summary files: average R^2 during training, and average
    R^2 after training (testing).

    NOTE on `output_filename`: this parameter exists but the two summary
    files this function writes always use an auto-generated,
    HSMs/timestamp-based name -- `output_filename` was accepted but never
    actually used in the original notebook code, and still isn't here
    (since it's ambiguous how one filename should apply to two output
    files). Decide how you want that handled before relying on this
    parameter.

    Args:
        folder_path: folder containing the individual idx*.npz files
        HSMs: array of HSM values used in the experiment (used in the
              summary and in the output filename)
        gains: array of gain values used in the experiment
        Dt, eta: fixed hyperparameters, saved alongside the results
        n_weight_seeds: array of weight seeds used
        output_path: directory to save the two summary files to (defaults
                     to a hardcoded path from the original notebook if not
                     given -- pass this explicitly rather than relying on
                     that default)
        output_filename: currently unused, see note above

    Returns:
        (training_summary_filepath, testing_summary_filepath)
    """
    if output_path is None:
        output_path = (
            r"C:\Users\Lenovo\Documents\B. Master Neurophysics\Stage\Presentations and reports and code"
            r"\code\FORCE\FORCE results\Changing gain and HSM periodic target\Summary files VScode\summaries_averages"
        )

    files = glob.glob(os.path.join(folder_path, "idx*.npz"))

    if len(files) == 0:
        raise ValueError(f"No idx*.npz files found in {folder_path}")

    files.sort(key=lambda f: int(os.path.basename(f).split('_')[0][3:]))

    print(f"Found {len(files)} individual result files")

    averages_train_Rsquared = []
    averages_test_Rsquared = []

    for i, filepath in enumerate(files):
        data = np.load(filepath)

        Rsquareds_train = data['Rsquareds_train']
        if len(Rsquareds_train) > 0:
            averages_train_Rsquared.append(np.mean(Rsquareds_train))
        else:
            print(f"  Warning [{i}]: No complete periods in train data for {os.path.basename(filepath)}")
            averages_train_Rsquared.append(np.nan)

        Rsquareds_test = data['Rsquareds_test_all_periods']
        if len(Rsquareds_test) > 0:
            averages_test_Rsquared.append(np.mean(Rsquareds_test))
        else:
            print(f"  Warning [{i}]: No complete periods in test data for {os.path.basename(filepath)}")
            averages_test_Rsquared.append(np.nan)

    print(f"\nCollected {len(averages_train_Rsquared)} training averages")
    print(f"Collected {len(averages_test_Rsquared)} testing averages")

    expected_count = len(gains) * len(n_weight_seeds) * len(HSMs)
    print(f"Expected: {expected_count} = {len(gains)} gains x {len(n_weight_seeds)} seeds x {len(HSMs)} HSMs")

    if len(averages_train_Rsquared) != expected_count:
        print(f"WARNING: Mismatch between number of files ({len(files)}) and expected combinations ({expected_count})")

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    summary_averages_during_training_filepath = os.path.join(
        output_path, f"summary_averages_during_training_filepath_HSMs={HSMs}_{timestamp}.npz"
    )
    np.savez(
        summary_averages_during_training_filepath,
        averages_training_Rsquareds=np.array(averages_train_Rsquared),
        gains=gains, HSMs=HSMs, etas=eta, Dts=Dt, weights_seeds=n_weight_seeds,
    )
    print(f"\nSaved training summary to: {os.path.basename(summary_averages_during_training_filepath)}")

    summary_averages_after_training_filepath = os.path.join(
        output_path, f"summary_averages_after_training_filepath_HSMs={HSMs}_{timestamp}.npz"
    )
    np.savez(
        summary_averages_after_training_filepath,
        averages_testing_Rsquareds=np.array(averages_test_Rsquared),
        gains=gains, HSMs=HSMs, etas=eta, Dts=Dt, weights_seeds=n_weight_seeds,
    )
    print(f"Saved testing summary to: {os.path.basename(summary_averages_after_training_filepath)}")

    return summary_averages_during_training_filepath, summary_averages_after_training_filepath
