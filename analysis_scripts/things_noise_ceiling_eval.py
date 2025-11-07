"""
================================================================================
THINGS Noise Ceiling Evaluation
================================================================================

Purpose:
    Computes the human noise ceiling for the THINGS odd-one-out dataset using
    leave-one-out cross-validation. The noise ceiling represents the upper bound
    on model performance, as it quantifies human inter-rater agreement when
    multiple participants judge the same triplets.

Context in Pipeline:
    This script is a PREPROCESSING step that analyzes the THINGS dataset to:
    1) Identify unique triplets and count responses across test sets
    2) Calculate leave-one-out noise ceiling (expected agreement with majority)
    3) Generate analysis metrics (participants, triplets, response counts)
    4) Save processed data for downstream model evaluation

Key Concepts:
    - Noise Ceiling: Maximum achievable performance given human variability
    - Leave-one-out estimation: For each triplet, remove one response and 
      predict it from the remaining responses' majority vote
    - Triplet normalization: Different orderings of same triplet are unified
      by sorting concept indices before aggregation

Inputs:
    - THINGS behavioral data CSV with columns: image1, image2, image3, choice, 
      subject_id, noise_ceiling (test set indicator)
      Download from: https://osf.io/f5rn6/ (data/triplet_dataset/)
    - Preprocessed triplet tensors for test sets 1-3 (created by 
      odd_one_out_text_preprocess.py)
    - Answer position tensors indicating chosen odd-one-out (0, 1, or 2)

Outputs:
    - Data/Things_data_preprocessed/unique_triplets_and_counts/
        * test_X_unique_triplets.pt: Unique triplets per test set
        * test_X_answers_counts.pt: Response counts [c0, c1, c2] per triplet
        * all_tests_unique_triplets.pt: Combined unique triplets
        * all_tests_answers_counts.pt: Combined response counts
        * analysis_noise_ceiling.csv: Summary statistics per test set
    
    Note: Test set 2 is the PRIMARY validation set used in the paper for all
          reported noise ceiling values and model performance comparisons.

Methods:
    - group_indices_by_unique_triplets(): Normalizes triplet orderings
    - count_choices_for_triplets(): Aggregates responses per unique triplet
    - leave_one_out_accuracy_per_triplet(): LOO accuracy for single triplet
    - leave_one_out_noise_ceiling(): Overall LOO noise ceiling across triplets

Usage:
    python analysis_scripts/things_noise_ceiling_eval.py
    
    Requires configuration via: scripts_configurations/things_noise_ceiling_eval.yaml

Dependencies:
    - odd_one_out_text_preprocess.py must be run first to create preprocessed
      triplet tensors
    - THINGS behavioral data CSV must be available

Related Scripts:
    - odd_one_out_text_preprocess.py: Creates initial triplet tensors from CSV
    - evaluate_THINGS_OOO_accuracy_CV.py: Uses noise ceiling data for validation

References:
    Noise ceiling methodology follows standard psychophysics practices for
    estimating human inter-rater reliability on similarity judgment tasks.

================================================================================
"""
import os
import sys
from pathlib import Path

# Set up paths to allow relative imports
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PARENT_DIR)  # Work with relative paths as if script is one level up
sys.path.append(str(PARENT_DIR))


import pandas as pd
import torch
import hydra
from omegaconf import DictConfig


# =============================================================================
# Function Definitions
# =============================================================================
def group_indices_by_unique_triplets(tensor, answers_positions):
    """
    Groups the indices of each unique triplet in the input tensor and updates 
    the answer positions so that different permutations of the same triplet are 
    treated identically.

    Args:
        tensor (torch.Tensor): An (N, 3) tensor of triplets.
        answers_positions (torch.Tensor): A 1D tensor of length N containing the 
                                            chosen position (0, 1, or 2) for each triplet.

    Returns:
        tuple: (unique_rows, inverse_indices, answers_positions_updated)
            - unique_rows (torch.Tensor): The unique triplets after row-wise sorting.
            - inverse_indices (torch.Tensor): For each original triplet row, an index 
                                                into unique_rows.
            - answers_positions_updated (torch.Tensor): The updated answer positions 
                                                          after sorting each row.
    """
    if tensor.ndim != 2 or tensor.size(1) != 3:
        raise ValueError("Input tensor must have shape (N, 3)")
    if answers_positions.ndim != 1 or answers_positions.size(0) != tensor.size(0):
        raise ValueError("answers_positions must be 1D and match the number of rows.")

    # 1) Sort each row so that different orderings become identical.
    sorted_t, sort_idx = torch.sort(tensor, dim=1)

    # 2) Invert the permutation so that we can update answers_positions.
    N = sorted_t.size(0)
    invert_idx = torch.empty_like(sort_idx)
    new_positions = torch.arange(3).expand(N, 3)  # shape: (N, 3)
    invert_idx.scatter_(1, sort_idx, new_positions)
    answers_positions_updated = invert_idx[torch.arange(N), answers_positions]

    # 3) Get unique triplets and the inverse indices mapping.
    unique_rows, inverse_indices = torch.unique(sorted_t, dim=0, return_inverse=True)

    return unique_rows, inverse_indices, answers_positions_updated


def count_choices_for_triplets(tensor, choices):
    """
    For each unique triplet (after sorting rows so that permutations match),
    counts the number of times each position (0, 1, 2) was chosen.

    Args:
        tensor (torch.Tensor): An (N, 3) tensor containing triplets.
        choices (torch.Tensor): A 1D tensor of length N with chosen indices (in the original ordering).

    Returns:
        tuple: (counts, unique_rows)
            - counts (torch.Tensor): A (U, 3) tensor, where U is the number of unique triplets.
                                     Each row holds counts for choices [0, 1, 2].
            - unique_rows (torch.Tensor): The unique, sorted triplets.
    """
    if tensor.ndim != 2 or tensor.size(1) != 3:
        raise ValueError("Input tensor must have shape (N, 3)")
    if choices.ndim != 1 or choices.size(0) != tensor.size(0):
        raise ValueError("Choices tensor must be 1D and match the number of rows.")
    
    # Sort and group triplets, updating choices accordingly.
    unique_rows, inverse_indices, choices_updated = group_indices_by_unique_triplets(tensor, choices)

    # One-hot encode the updated choices.
    N = choices.size(0)
    one_hot_choices = torch.zeros((N, 3), dtype=torch.int64)
    one_hot_choices[torch.arange(N), choices_updated] = 1

    # Sum the one-hot vectors for each unique triplet.
    counts = torch.zeros((unique_rows.size(0), 3), dtype=torch.int64)
    counts.index_add_(0, inverse_indices, one_hot_choices)

    return counts, unique_rows


def compute_noise_ceiling(counts: torch.Tensor) -> float:
    """
    Computes the (simple) noise ceiling as the average fraction of responses 
    that agree with the majority vote for each triplet.

    Args:
        counts (torch.Tensor): A (N, 3) tensor where each row contains the counts for the three choices.

    Returns:
        float: The noise ceiling (upper bound performance) as a fraction.
    """
    total_counts = counts.sum(dim=1).float()
    max_counts, _ = counts.max(dim=1)
    triplet_accuracies = max_counts.float() / total_counts
    return triplet_accuracies.mean().item()


def leave_one_out_accuracy_per_triplet(counts):
    """
    Computes the leave-one-out accuracy for a single triplet. For each option 
    that received votes, one vote is left out and the probability that this 
    vote agrees with the majority (among the remaining votes) is computed.
    
    Args:
        counts (torch.Tensor): A tensor of shape (3,) with counts [c0, c1, c2].
    
    Returns:
        float: The leave-one-out accuracy for this triplet.
    """
    T = counts.sum().item()
    if T < 2:
        return float('nan')  # Not enough responses to perform leave-one-out

    acc = 0.0
    for i in range(3):
        if counts[i].item() == 0:
            continue  # No vote for this option; skip.
        updated_counts = counts.clone().float()
        updated_counts[i] -= 1  # Remove one vote.
        max_val = updated_counts.max().item()
        winners = (updated_counts == max_val).nonzero(as_tuple=False).flatten().tolist()
        prob = 1.0 / len(winners) if i in winners else 0.0
        acc += (counts[i].item() / T) * prob

    return acc


def leave_one_out_noise_ceiling(counts_tensor):
    """
    Computes the overall leave-one-out noise ceiling (mean accuracy) from aggregated counts.
    
    Args:
        counts_tensor (torch.Tensor): A tensor of shape (N, 3) for N triplets.
    
    Returns:
        float: The overall leave-one-out noise ceiling (averaged over triplets).
    """
    accuracies = []
    for i in range(counts_tensor.size(0)):
        acc = leave_one_out_accuracy_per_triplet(counts_tensor[i])
        if not torch.isnan(torch.tensor(acc)):
            accuracies.append(acc)
    return sum(accuracies) / len(accuracies) if accuracies else float('nan')


def leave_one_out_noise_ceiling_with_std(counts_tensor):
    """
    Computes the overall leave-one-out noise ceiling (mean accuracy) and its standard deviation.
    
    Args:
        counts_tensor (torch.Tensor): A tensor of shape (N, 3), where each row contains counts for a triplet.
    
    Returns:
        tuple: (mean_accuracy, std_accuracy) averaged over all triplets.
    """
    accuracies = []
    for i in range(counts_tensor.size(0)):
        acc = leave_one_out_accuracy_per_triplet(counts_tensor[i])
        if not torch.isnan(torch.tensor(acc)):
            accuracies.append(acc)
    if not accuracies:
        return float('nan'), float('nan')
    accuracies_tensor = torch.tensor(accuracies)
    return accuracies_tensor.mean().item(), accuracies_tensor.std(unbiased=True).item()


@hydra.main(version_base=None, config_path="../scripts_configurations", config_name="things_noise_ceiling_eval")
def main(cfg: DictConfig):
    """Main noise ceiling evaluation function using Hydra configuration."""
    
    print("="*80)
    print("THINGS Noise Ceiling Evaluation")
    print("="*80)
    
    # --- Testing leave-one-out functions on a small sample ---
    print("\n1. Testing LOO functions on sample data...")
    sample_counts = torch.tensor([
        [10, 2, 3],   # Total = 15
        [5, 5, 0],    # Ties may occur
        [7, 3, 0]
    ], dtype=torch.float)
    
    los = leave_one_out_noise_ceiling(sample_counts)
    mean_los, std_los = leave_one_out_noise_ceiling_with_std(sample_counts)
    print(f"   Sample Leave-one-out Noise Ceiling: {los:.4f}")
    print(f"   Sample LOO Mean: {mean_los:.4f}, Std: {std_los:.4f}")
    
    # --- Testing count_choices_for_triplets ---
    print("\n2. Testing count_choices_for_triplets...")
    x = torch.tensor([[1, 2, 3], [3, 2, 1], [6, 4, 5]])
    g = torch.tensor([0, 2, 0])  # Choices for each triplet
    counts_example, unique_triplets_example = count_choices_for_triplets(x, g)
    print(f"   Example Counts:\n{counts_example}")
    
    # --- Loading test set tensors from configuration ---
    print("\n3. Loading test set tensors...")
    test_1_triplets_tensor = torch.load(cfg.paths.triplets.test1)
    test_2_triplets_tensor = torch.load(cfg.paths.triplets.test2)
    test_3_triplets_tensor = torch.load(cfg.paths.triplets.test3)
    all_tests_together_tensor = torch.cat(
        (test_1_triplets_tensor, test_2_triplets_tensor, test_3_triplets_tensor), dim=0
    )
    
    test_1_answers_positions_tensor = torch.load(cfg.paths.answers.test1)
    test_2_answers_positions_tensor = torch.load(cfg.paths.answers.test2)
    test_3_answers_positions_tensor = torch.load(cfg.paths.answers.test3)
    all_tests_answers_positions_tensor = torch.cat(
        (test_1_answers_positions_tensor, test_2_answers_positions_tensor, test_3_answers_positions_tensor), dim=0
    )
    print(f"   Loaded {len(all_tests_together_tensor):,} total judgments")
    
    # --- Get answer counts for each unique triplet in each test set ---
    print("\n4. Computing unique triplets and answer counts...")
    test_1_answers_counts, test_1_unique_triplets = count_choices_for_triplets(
        test_1_triplets_tensor, test_1_answers_positions_tensor
    )
    test_2_answers_counts, test_2_unique_triplets = count_choices_for_triplets(
        test_2_triplets_tensor, test_2_answers_positions_tensor
    )
    test_3_answers_counts, test_3_unique_triplets = count_choices_for_triplets(
        test_3_triplets_tensor, test_3_answers_positions_tensor
    )
    all_tests_answers_counts, all_tests_unique_triplets = count_choices_for_triplets(
        all_tests_together_tensor, all_tests_answers_positions_tensor
    )
    
    # --- Compute noise ceilings ---
    print("\n5. Computing noise ceilings...")
    noise_ceiling_test_1 = leave_one_out_noise_ceiling(test_1_answers_counts)
    noise_ceiling_test_2 = leave_one_out_noise_ceiling(test_2_answers_counts)
    noise_ceiling_test_3 = leave_one_out_noise_ceiling(test_3_answers_counts)
    noise_ceiling_all_tests = leave_one_out_noise_ceiling(all_tests_answers_counts)
    
    print("\n" + "="*80)
    print("NOISE CEILING RESULTS (Leave-One-Out)")
    print("="*80)
    print(f"   Test 1: {noise_ceiling_test_1*100:.2f}%")
    print(f"   Test 2: {noise_ceiling_test_2*100:.2f}% ⭐ PRIMARY (used in paper)")
    print(f"   Test 3: {noise_ceiling_test_3*100:.2f}%")
    print(f"   All tests combined: {noise_ceiling_all_tests*100:.2f}%")
    print("="*80)
    print("\nNote: Test set 2 values are used for all model comparisons in the paper.")
    print("="*80)
    
    # --- Analysis Metrics ---
    print("\n6. Computing detailed analysis metrics...")
    df = pd.read_csv(cfg.paths.behavior_csv, sep="\t")
    
    # Prepare containers for metrics (one row per test set)
    metrics = {}
    for test_num, answers_counts, unique_triplets in [
        (2, test_2_answers_counts, test_2_unique_triplets),
        (3, test_3_answers_counts, test_3_unique_triplets),
    ]:
        N_unique = unique_triplets.size(0)
        total_answers_per_triplet = answers_counts.sum(dim=1).float()
        avg_answers = total_answers_per_triplet.mean().item()
        
        if 'noise_ceiling' in df.columns:
            n_participants = len(df[df['noise_ceiling'] == test_num]['subject_id'].unique())
        else:
            n_participants = float('nan')
        
        mean_los, std_los = leave_one_out_noise_ceiling_with_std(answers_counts)
        metrics[test_num] = {
            "N_unique_triplets": N_unique,
            "average_answers_per_triplet": avg_answers,
            "N_participants": n_participants,
            "leave_one_out_noise_ceiling_mean": mean_los,
            "leave_one_out_noise_ceiling_std": std_los
        }
    
    # Additionally, compute metrics for all tests together
    total_answers_per_triplet_all = all_tests_answers_counts.sum(dim=1).float()
    avg_answers_all = total_answers_per_triplet_all.mean().item()
    if 'subject_id' in df.columns:
        n_participants_all = len(df[df['noise_ceiling'] != 0]['subject_id'].unique())
    else:
        n_participants_all = float('nan')
    mean_los_all, std_los_all = leave_one_out_noise_ceiling_with_std(all_tests_answers_counts)
    metrics["AllTests"] = {
        "N_unique_triplets": all_tests_unique_triplets.size(0),
        "average_answers_per_triplet": avg_answers_all,
        "N_participants": n_participants_all,
        "leave_one_out_noise_ceiling_mean": mean_los_all,
        "leave_one_out_noise_ceiling_std": std_los_all
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    
    # Save the metrics to CSV
    df_metrics = pd.DataFrame.from_dict(metrics, orient="index")
    df_metrics.to_csv(cfg.paths.summary_csv, index=True)
    print(f"\n7. Saved analysis metrics to: {cfg.paths.summary_csv}")
    
    # Save unique triplets and counts as torch tensors
    print("\n8. Saving unique triplets and answer counts...")
    output_mapping = {
        'test_1': (test_1_unique_triplets, test_1_answers_counts, cfg.output_files.test1_triplets, cfg.output_files.test1_counts),
        'test_2': (test_2_unique_triplets, test_2_answers_counts, cfg.output_files.test2_triplets, cfg.output_files.test2_counts),
        'test_3': (test_3_unique_triplets, test_3_answers_counts, cfg.output_files.test3_triplets, cfg.output_files.test3_counts),
        'all_tests': (all_tests_unique_triplets, all_tests_answers_counts, cfg.output_files.all_triplets, cfg.output_files.all_counts),
    }
    
    for test_name, (triplets, counts, triplets_file, counts_file) in output_mapping.items():
        triplets_path = os.path.join(cfg.paths.output_dir, triplets_file)
        counts_path = os.path.join(cfg.paths.output_dir, counts_file)
        torch.save(triplets, triplets_path)
        torch.save(counts, counts_path)
        print(f"   Saved {test_name}: {len(triplets):,} unique triplets")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"All outputs saved to: {cfg.paths.output_dir}")
    print("="*80)


# =============================================================================
# Example Usage and Analysis
# =============================================================================
if __name__ == "__main__":
    main()