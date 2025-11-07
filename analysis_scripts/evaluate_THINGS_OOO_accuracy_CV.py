"""
Evaluate Neural Network Models on THINGS Odd-One-Out Dataset

Key Features:
    - Cross-validation evaluation across disjoint image sets
    - Multiple regularization methods (L2, eye-distance/scalar-matrix shrinkage)
    - Flexible transformation constraints (full matrix, rectangular, diagonal, zero-shot)
    - Automatic result caching to avoid redundant computations
    - Prediction saving for downstream analyses

Workflow:
    1. Load human odd-one-out judgments from THINGS dataset
    2. Create/load cross-validation folds with disjoint image sets
    3. For each model:
       - Extract pre-computed image features
       - Fit linear transformation W to predict human choices
       - Evaluate prediction accuracy on held-out images
    4. Save results (accuracies, NLLs, predictions)

Usage:
    python analyses_scripts/evaluate_THINGS_OOO_accuracy_CV.py
    
    Configuration is controlled via evaluate_THINGS_OOO_accuracy_CV.yaml

IMPORTANT - Output Directory Structure:
    The results_out path is used for ALL regularization types evaluated together.
    
    Default configuration evaluates BOTH L2 and eye_distance simultaneously:
        reg_func_types: ["L2", "eye_distance"]
        results_out: "Results/regularization_methods_compare_all_models_eye_distance"
    
    This creates separate files for EACH regularization in the SAME directory:
        Results/regularization_methods_compare_all_models_eye_distance/full/full_W/
            ├── best_test_acc_L2_full_full_W.csv              # L2 results
            ├── best_test_acc_eye_distance_full_full_W.csv    # eye_distance results
            └── best_test_acc_for_each_model_full_full_W.csv  # Combined comparison
    
    Directory naming convention:
        - Multiple regularizations (comparison): Use primary/default in suffix (_eye_distance)
        - Single regularization: Use that regularization in suffix (_L2, _L1)
    
    The figure scripts (e.g., create_Figure_2.py) use the same results_out path as 
    results_base_dir, then select which regularization to display via the 
    regularization_type parameter.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PARENT_DIR)  # We want the script to work with relative paths as if script is one level up
sys.path.append(str(PARENT_DIR))
import json
from typing import Dict, List, Tuple, Optional, Any
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm
from omegaconf import ListConfig
from tools.Cog_model import CogModel
from tools.triplets_data_set import OddOneOutDataset
from tools.utils import  create_partitioned_datasets_THINGS

# =============================================================================
# UTILITY HELPERS
# =============================================================================
# These functions handle basic file I/O and directory management.
# They ensure proper data loading, atomic file writes, and directory creation.
# =============================================================================

def _ensure_dirs(paths: Dict[str, str]):
    """
    Ensure that necessary output directories exist.
    
    Creates directories for results and predictions if they don't already exist.
    Uses parents=True to create intermediate directories as needed.
    
    Args:
        paths: Dictionary containing path configurations. Checks for 'results_out' 
               and 'predictions_out' keys.
    """
    for key in ["results_out", "predictions_out"]:
        if key in paths:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)


def _load_tensor(path: str) -> torch.Tensor:
    """
    Load a PyTorch tensor from a file.
    
    Args:
        path: Path to .pt file containing saved tensor
        
    Returns:
        Loaded PyTorch tensor
    """
    return torch.load(path)


def _save_json_atomic(obj: Dict, path: Path):
    """
    Save a dictionary to a JSON file atomically.
    
    Uses a temporary file and atomic rename to prevent corruption if the
    process is interrupted during writing. This ensures results are never
    partially written.
    
    Args:
        obj: Dictionary to save
        path: Target file path for JSON output
    """
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def _append_csv(row: Dict, path: Path):
    """
    Append a single row to a CSV file.
    
    Creates the file with headers if it doesn't exist, otherwise appends
    to existing file without duplicating headers.
    
    Args:
        row: Dictionary representing a single row of data
        path: Path to CSV file
    """
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _load_datasets(paths: Dict[str, str]) -> Tuple[torch.Tensor, torch.Tensor, OddOneOutDataset]:
    """
    Load training triplets and test dataset.
    
    The THINGS dataset contains human odd-one-out judgments. Each triplet consists
    of three images, and participants selected which image was the "odd one out".
    
    Args:
        paths: Dictionary with keys:
            - 'train_triplets': Path to training triplet indices
            - 'train_answers': Path to training answer positions (1, 2, or 3)
            - 'test_triplets': Path to test/noise ceiling triplet indices
            - 'test_answers': Path to test answer positions
    
    Returns:
        Tuple of:
            - triplets_tensor: Training triplet indices (N_train x 3)
            - triplets_answers_tensor: Training answer positions (N_train,)
            - test_dataset: OddOneOutDataset object for noise ceiling evaluation
    """
    triplets_tensor = _load_tensor(paths["train_triplets"])
    triplets_answers_tensor = _load_tensor(paths["train_answers"])
    
    test_triplets = _load_tensor(paths["test_triplets"])
    test_answers = _load_tensor(paths["test_answers"])
    test_dataset = OddOneOutDataset(
        triplets=test_triplets,
        choice_position=test_answers,
        device='cuda'
    )
    
    return triplets_tensor, triplets_answers_tensor, test_dataset


def _load_or_create_folds(
    triplets_tensor: torch.Tensor,
    triplets_answers_tensor: torch.Tensor,
    paths: Dict[str, str],
    k_folds: int,
    n_concepts: int = 1854,
    random_seed: int = 42
) -> List[Tuple[OddOneOutDataset, OddOneOutDataset]]:
    """
    Load existing fold partitions or create new ones.
    
    This function implements cross-validation over DISJOINT IMAGE SETS, not data splits.
    Each fold uses a different subset of the 1,854 THINGS images for testing, ensuring
    that models generalize to novel images (as described in the paper).
    
    The partitioning ensures:
        1. Each image appears in exactly one test set across the 3 folds
        2. Train and test sets are completely disjoint (no shared images)
        3. All 1,854 images are covered exactly once across test sets
    
    Args:
        triplets_tensor: All training triplets (N_triplets x 3 image indices)
        triplets_answers_tensor: Corresponding odd-one-out choices (N_triplets,)
        paths: Dictionary containing 'partitions_dir' key for save/load location
        k_folds: Number of cross-validation folds (typically 3)
        n_concepts: Total number of images/concepts in THINGS (default: 1854)
        random_seed: Random seed for reproducible partitioning
    
    Returns:
        List of (train_dataset, test_dataset) tuples, one per fold.
        Each dataset is an OddOneOutDataset containing triplets drawn from
        the respective image partition.
    
    File Structure:
        partitions_dir/
            fold_1_train_partitioning_indexes.pt    # Triplet indices for training
            fold_1_val_partitioning_indexes.pt      # Triplet indices for validation
            fold_1_test_partitioning_indexes.pt     # Triplet indices for testing
            ... (repeated for each fold)
    
    Note:
        If partition files exist, they are loaded and validated. Otherwise, new
        partitions are created using create_partitioned_datasets_THINGS().
    """
    # Check for existence of all required files
    required_files = []
    for fold_idx in range(1, k_folds + 1):
        required_files.extend([
            Path(paths["partitions_dir"]) / f"fold_{fold_idx}_train_partitioning_indexes.pt",
            Path(paths["partitions_dir"]) / f"fold_{fold_idx}_val_partitioning_indexes.pt",
            Path(paths["partitions_dir"]) / f"fold_{fold_idx}_test_partitioning_indexes.pt"
        ])
    
    if all(p.exists() for p in required_files):
        # Load existing partitions from .pt files
        print(f"Loading existing fold partitions from {paths['partitions_dir']}")
        folds_datasets = []
        
        for fold_idx in range(1, k_folds + 1):
            # Load train, validation and test indices
            train_indices = torch.load(Path(paths["partitions_dir"]) / f"fold_{fold_idx}_train_partitioning_indexes.pt")
            val_indices = torch.load(Path(paths["partitions_dir"]) / f"fold_{fold_idx}_val_partitioning_indexes.pt")
            test_indices = torch.load(Path(paths["partitions_dir"]) / f"fold_{fold_idx}_test_partitioning_indexes.pt")
            
            # Verify indices are mutually exclusive
            train_val_indices = torch.cat([train_indices, val_indices])
            train_val_set = set(train_val_indices.cpu().numpy())
            test_set = set(test_indices.cpu().numpy())
            
            if train_val_set.intersection(test_set):
                raise ValueError(f"Fold {fold_idx}: Train/val and test sets are not mutually exclusive!")
            
            # Verify no duplicates within train+val
            if len(train_val_set) != len(train_val_indices):
                raise ValueError(f"Fold {fold_idx}: Duplicate indices found in train+val set!")
            
            # Verify no duplicates within test
            if len(test_set) != len(test_indices):
                raise ValueError(f"Fold {fold_idx}: Duplicate indices found in test set!")
            
            print(f"Fold {fold_idx} verification passed:")
            print(f"  Train size: {len(train_indices)}")
            print(f"  Validation size: {len(val_indices)}")
            print(f"  Test size: {len(test_indices)}")
            
            # Create datasets
            train_set = OddOneOutDataset(
                triplets=triplets_tensor[train_val_indices],
                choice_position=triplets_answers_tensor[train_val_indices],
                device='cuda'
            )
            
            test_set = OddOneOutDataset(
                triplets=triplets_tensor[test_indices],
                choice_position=triplets_answers_tensor[test_indices],
                device='cuda'
            )
            
            folds_datasets.append((train_set, test_set))
            
            # Additional verification of indices
            train_indices_flat = train_set.triplets.flatten().unique().cpu().numpy()
            test_indices_flat = test_set.triplets.flatten().unique().cpu().numpy()
            
            # Check for intersection
            if set(train_indices_flat).intersection(set(test_indices_flat)):
                raise ValueError(f"Fold {fold_idx}: Found overlapping indices between train and test sets!")
            
            # Check completeness (should cover 0 to 1853)
            all_indices = set(train_indices_flat).union(set(test_indices_flat))
            expected_indices = set(range(1854))  # 0 to 1853
            if all_indices != expected_indices:
                missing = expected_indices - all_indices
                extra = all_indices - expected_indices
                raise ValueError(
                    f"Fold {fold_idx}: Indices don't cover the full range 0-1853!\n"
                    f"Missing indices: {sorted(missing)}\n"
                    f"Extra indices: {sorted(extra)}"
                )
            
            print(f"Fold {fold_idx} indices verification passed:")
            print(f"  Train unique indices: {len(train_indices_flat)}")
            print(f"  Test unique indices: {len(test_indices_flat)}")
            print(f"  Total unique indices: {len(all_indices)}")
        
        return folds_datasets
    else:
        # Create new partitions using the utility function
        print(f"Creating new fold partitions and saving to {paths['partitions_dir']}")
        
        # Ensure the partitions directory exists
        Path(paths["partitions_dir"]).mkdir(parents=True, exist_ok=True)
        
        # Use the create_partitioned_datasets_THINGS function
        folds_datasets = create_partitioned_datasets_THINGS(
            full_triplet_set=triplets_tensor,
            full_odd_one_out_index=triplets_answers_tensor,
            K_folds=k_folds,
            N_concepts=n_concepts,
            random_seed=random_seed,
            with_validation=True,  # We need validation to merge with train
            save_path=paths["partitions_dir"]
        )
        
        # Convert the (train, val, test) tuples to (train+val, test) tuples
        merged_folds = []
        for fold_idx, (train_set, val_set, test_set) in enumerate(folds_datasets, 1):
            # Merge train and val sets
            merged_train_triplets = torch.cat([train_set.triplets, val_set.triplets])
            merged_train_answers = torch.cat([train_set.choice_position, val_set.choice_position])
            
            # Verify indices are mutually exclusive
            train_val_indices = torch.arange(len(merged_train_triplets))
            test_indices = torch.arange(len(test_set.triplets))
            
            train_val_set = set(train_val_indices.cpu().numpy())
            test_set_indices = set(test_indices.cpu().numpy())
            
            if train_val_set.intersection(test_set_indices):
                raise ValueError(f"Fold {fold_idx}: Train/val and test sets are not mutually exclusive!")
            
            # Verify no duplicates within train+val
            if len(train_val_set) != len(train_val_indices):
                raise ValueError(f"Fold {fold_idx}: Duplicate indices found in train+val set!")
            
            # Verify no duplicates within test
            if len(test_set_indices) != len(test_indices):
                raise ValueError(f"Fold {fold_idx}: Duplicate indices found in test set!")
            
            print(f"Fold {fold_idx} verification passed:")
            print(f"  Train+val size: {len(merged_train_triplets)}")
            print(f"  Test size: {len(test_set.triplets)}")
            
            merged_train_set = OddOneOutDataset(
                triplets=merged_train_triplets,
                choice_position=merged_train_answers,
                device='cuda'
            )
            
            merged_folds.append((merged_train_set, test_set))
        
        return merged_folds


# =============================================================================
# RESULTS VERIFICATION HELPERS
# =============================================================================
# These functions check if results already exist to avoid redundant computation.
# They verify both result files (JSON/CSV) and prediction files if requested.
# This enables incremental runs when adding new models or changing configurations.
# =============================================================================

def _check_zero_shot_results_exist(
    model_name: str,
    save_dir: Path,
    feature_type: str,
    save_predictions: bool = False,
    predictions_dir: Optional[Path] = None,
    k_folds: int = 5
) -> bool:
    """
    Check if zero-shot results already exist for a model.
    
    Zero-shot evaluation uses the model's original representations without
    any learned transformation (W = Identity matrix).
    
    Args:
        model_name: Name of the neural network model
        save_dir: Directory where results are saved
        feature_type: Type of features used ('full' or 'PCA_500')
        save_predictions: Whether predictions should have been saved
        predictions_dir: Directory where predictions should be stored
        k_folds: Number of cross-validation folds
    
    Returns:
        True if all required result files exist and are valid, False otherwise
    """
    # Check if JSON results file exists
    json_path = save_dir / f"zero_shot_results_{feature_type}.json"
    if not json_path.exists():
        return False
    
    # Load and check if model results exist in JSON
    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        if model_name not in results:
            return False
        
        # Check if all expected metrics are present
        expected_metrics = {"train_acc", "test_acc", "train_NLL", "test_NLL"}
        if not expected_metrics.issubset(set(results[model_name].keys())):
            return False
        
    except (json.JSONDecodeError, KeyError):
        return False
    
    # Check prediction files if they should exist
    if save_predictions and predictions_dir:
        pred_folder = Path(predictions_dir) / model_name / "zero_shot"
        if not pred_folder.exists():
            return False
        
        for fold_idx in range(k_folds):
            pred_file = pred_folder / f"preds_{fold_idx}.pt"
            if not pred_file.exists():
                return False
    
    return True


def _check_regular_results_exist(
    model_name: str,
    save_dir: Path,
    feature_type: str,
    constraint_name: str,
    reg_funcs: List[str],
    reg_cons: List[float],
    save_predictions: bool = False,
    predictions_dir: Optional[Path] = None,
    k_folds: int = 5
) -> bool:
    """
    Check if regular evaluation results already exist for a model.
    
    Regular evaluation learns a linear transformation W to align model
    representations to human judgments, with various regularization methods
    and hyperparameters.
    
    Args:
        model_name: Name of the neural network model
        save_dir: Directory where results are saved
        feature_type: Type of features used ('full' or 'PCA_500')
        constraint_name: Name of the transformation constraint (e.g., 'full_W', 'diagonal')
        reg_funcs: List of regularization functions ('L2', 'eye_distance')
        reg_cons: List of regularization constants to check
        save_predictions: Whether predictions should have been saved
        predictions_dir: Directory where predictions should be stored
        k_folds: Number of cross-validation folds
    
    Returns:
        True if all required result files exist and are valid, False otherwise
    """
    # Check JSON results files for each regularization function
    for reg_func in reg_funcs:
        json_path = save_dir / f"mean_acc_train_val_{reg_func}_{constraint_name}_{feature_type}.json"
        if not json_path.exists():
            return False
        
        # Load and check if model results exist in JSON
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
            
            if model_name not in results:
                return False
            
            # Check if all regularization constants are present
            for reg_con in reg_cons:
                reg_con_str = str(reg_con)
                if reg_con_str not in results[model_name]:
                    return False
                
                # Check if all expected metrics are present
                expected_metrics = {"train_acc", "test_acc", "train_NLL", "test_NLL"}
                if not expected_metrics.issubset(set(results[model_name][reg_con_str].keys())):
                    return False
            
        except (json.JSONDecodeError, KeyError):
            return False
        
        # Check prediction files if they should exist
        if save_predictions and predictions_dir:
            for reg_con in reg_cons:
                pred_folder = Path(predictions_dir) / model_name / reg_func / str(reg_con)
                if not pred_folder.exists():
                    return False
                
                for fold_idx in range(k_folds):
                    pred_file = pred_folder / f"preds_{fold_idx}.pt"
                    if not pred_file.exists():
                        return False
    
    return True


def _check_model_results_exist(
    model_name: str,
    constraint_config: Dict[str, Any],
    save_dir: Path,
    feature_type: str,
    reg_funcs: Optional[List[str]],
    reg_cons: Optional[List[float]],
    predictions_dir: Optional[Path] = None,
    k_folds: int = 5
) -> bool:
    """
    Check if results already exist for a specific model and configuration.
    
    This is the main entry point for result verification. It delegates to
    _check_zero_shot_results_exist or _check_regular_results_exist based
    on the constraint type.
    
    Args:
        model_name: Name of the neural network model
        constraint_config: Configuration dictionary with 'constraint' and 
                          'save_predictions' keys
        save_dir: Directory where results are saved
        feature_type: Type of features used ('full' or 'PCA_500')
        reg_funcs: List of regularization functions (required for non-zero-shot)
        reg_cons: List of regularization constants (required for non-zero-shot)
        predictions_dir: Directory where predictions should be stored
        k_folds: Number of cross-validation folds
    
    Returns:
        True if all required files exist, False otherwise
    """
    constraint = constraint_config.get("constraint")
    save_predictions = constraint_config.get("save_predictions", False)
    
    # Use standardized constraint formatting
    constraint_name, standardized_constraint = fix_constraint_type_format(constraint)
    
    if standardized_constraint == "zero_shot":
        return _check_zero_shot_results_exist(
            model_name, save_dir, feature_type, save_predictions, predictions_dir, k_folds
        )
    else:
        assert reg_funcs is not None and reg_cons is not None, \
            "reg_funcs and reg_cons must be provided for non-zero-shot evaluations."
        return _check_regular_results_exist(
            model_name, save_dir, feature_type, constraint_name, reg_funcs, reg_cons,
            save_predictions, predictions_dir, k_folds
        )


# =============================================================================
# CROSS-VALIDATION EVALUATION
# =============================================================================
# Core functions for evaluating models on odd-one-out triplets.
# These implement the linear probing procedure: fit transformation matrix W
# on training triplets, evaluate on test triplets from disjoint image sets.
# =============================================================================

def evaluate_model_cv(
    model_name: str,
    model_features: torch.Tensor,
    folds: List[Tuple[OddOneOutDataset, OddOneOutDataset]],
    reg_cons: List[float],
    reg_func: str,
    constraint: Optional[str],
    device: str,
    save_predictions: bool = True,
    predictions_dir: Optional[Path] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a single model across all regularization constants using cross-validation.
    
    This function implements the core evaluation loop:
    1. For each regularization constant λ:
       2. For each cross-validation fold:
          3. Initialize CogModel with specified constraint on W
          4. Fit W to minimize negative log-likelihood on training triplets
          5. Evaluate accuracy and NLL on test triplets (disjoint images)
          6. Optionally save probabilistic predictions for downstream analysis
    
    Args:
        model_name: Name of the neural network model
        model_features: Pre-extracted features (N_images x feature_dim)
        folds: List of (train_dataset, test_dataset) tuples from disjoint image sets
        reg_cons: List of regularization constants to evaluate
        reg_func: Regularization function type ('L2' or 'eye_distance')
                  - 'L2': Standard Frobenius norm ||W||²_F
                  - 'eye_distance': Scalar-matrix shrinkage ||W - γI||²_F (Eq. 4 in paper)
        constraint: Transformation constraint type:
                   - None: Full p×p matrix (maximum flexibility)
                   - 'diagonal': Diagonal matrix only
                   - ('Rectangular', k): Only top k principal components
                   - 'zero_shot': Identity matrix (no learning)
        device: 'cuda' or 'cpu'
        save_predictions: Whether to save probabilistic predictions
        predictions_dir: Directory for saving predictions
    
    Returns:
        Dictionary mapping regularization constant (as string) to metrics:
        {
            '0.001': {
                'train_acc': 0.65,     # Mean accuracy on training folds
                'test_acc': 0.58,      # Mean accuracy on test folds
                'train_NLL': 0.92,     # Mean negative log-likelihood (train)
                'test_NLL': 1.05       # Mean negative log-likelihood (test)
            },
            ...
        }
    
    Note:
        Accuracy is averaged across folds. Each fold uses disjoint images, so
        test accuracy measures generalization to novel stimuli.
    """
    results = {}
    
    reg_pbar = tqdm(reg_cons, desc=f"{reg_func} λ", position=2, leave=False)
    for reg_con in reg_pbar:
        reg_pbar.set_description(f"{reg_func} λ={reg_con}")
        train_accs = []
        train_nlls = []
        test_accs = []
        test_nlls = []
        
        for fold_idx, (train_set, test_set) in enumerate(tqdm(folds, desc="Folds", position=3, leave=False)):
            # Initialize model
            cog_model = CogModel(
                model_name=model_name,
                images_features_dim=model_features.shape[1],
                device=device,
                constraints=constraint
            )
            
            # Train model (unless zero-shot)
            if constraint != 'zero_shot':
                cog_model.fit_W_matrix(
                    train_set=train_set,
                    images_features=model_features,
                    reg_func=reg_func,
                    reg_con=reg_con,
                    verbose=False
                )
            
            # Evaluate on train set
            train_acc, train_nll = cog_model.test_model(
                test_set=train_set,
                images_features=model_features,
                batch_size=len(train_set)
            )
            train_accs.append(train_acc.item())
            train_nlls.append(train_nll.item())
            
            # Evaluate on test set
            test_acc, test_nll = cog_model.test_model(
                test_set=test_set,
                images_features=model_features,
                batch_size=len(test_set)
            )
            test_accs.append(test_acc.item())
            test_nlls.append(test_nll.item())
            print(f"Fold {fold_idx} test accuracy: {test_acc.item()}")
            # Save predictions if requested
            if save_predictions and predictions_dir:
                pred_folder = predictions_dir / model_name / reg_func / str(reg_con)
                pred_folder.mkdir(parents=True, exist_ok=True)
                
                preds = cog_model.pred(
                    triplets=test_set.triplets,
                    images_features=model_features
                )
                torch.save(preds, pred_folder / f"preds_{fold_idx}.pt")
            
            # Clean up
            del cog_model
            torch.cuda.empty_cache()
        
        # Store mean results
        results[str(reg_con)] = {
            "train_acc": float(np.mean(train_accs)),
            "test_acc": float(np.mean(test_accs)),
            "train_NLL": float(np.mean(train_nlls)),
            "test_NLL": float(np.mean(test_nlls))
        }
    
    return results


def evaluate_zero_shot(
    model_name: str,
    model_features: torch.Tensor,
    folds: List[Tuple[OddOneOutDataset, OddOneOutDataset]],
    device: str,
    save_predictions: bool = True,
    predictions_dir: Optional[Path] = None
) -> Dict[str, float]:
    """
    Evaluate zero-shot performance (W = Identity, no learning).
    
    Zero-shot evaluation tests how well the model's original representations
    predict human judgments without any alignment. This provides a baseline
    for comparison with flexible evaluation methods.
    
    Args:
        model_name: Name of the neural network model
        model_features: Pre-extracted features (N_images x feature_dim)
        folds: List of (train_dataset, test_dataset) tuples
        device: 'cuda' or 'cpu'
        save_predictions: Whether to save probabilistic predictions
        predictions_dir: Directory for saving predictions
    
    Returns:
        Dictionary with averaged metrics:
        {
            'train_acc': 0.52,
            'test_acc': 0.51,
            'train_NLL': 1.08,
            'test_NLL': 1.09
        }
    
    Note:
        The transformation matrix W is verified to be the identity matrix.
    """
    train_accs = []
    train_nlls = []
    test_accs = []
    test_nlls = []
    
    for fold_idx, (train_set, test_set) in enumerate(tqdm(folds, desc="Folds", position=2, leave=False)):
        cog_model = CogModel(
            model_name=model_name,
            images_features_dim=model_features.shape[1],
            device=device,
            constraints='zero_shot'
        )
        
        # Verify zero-shot constraint
        weight = cog_model.W_mat.weight
        assert isinstance(weight, torch.Tensor), "W_mat.weight must be a Tensor"
        assert torch.allclose(
            weight, 
            torch.eye(model_features.shape[1]).to(device)
        )
        
        # Evaluate
        train_acc, train_nll = cog_model.test_model(
            test_set=train_set,
            images_features=model_features,
            batch_size=len(train_set)
        )
        train_accs.append(train_acc.item())
        train_nlls.append(train_nll.item())
        
        test_acc, test_nll = cog_model.test_model(
            test_set=test_set,
            images_features=model_features,
            batch_size=len(test_set)
        )
        test_accs.append(test_acc.item())
        test_nlls.append(test_nll.item())
        
        # Save predictions
        if save_predictions and predictions_dir:
            pred_folder = predictions_dir / model_name / "zero_shot"
            pred_folder.mkdir(parents=True, exist_ok=True)
            
            preds = cog_model.pred(
                triplets=test_set.triplets,
                images_features=model_features
            )
            torch.save(preds, pred_folder / f"preds_{fold_idx}.pt")
        
        del cog_model
        torch.cuda.empty_cache()
    
    return {
        "train_acc": float(np.mean(train_accs)),
        "test_acc": float(np.mean(test_accs)),
        "train_NLL": float(np.mean(train_nlls)),
        "test_NLL": float(np.mean(test_nlls))
    }


# =============================================================================
# RESULTS AGGREGATION
# =============================================================================
# Functions for collecting and organizing results across models and regularization
# settings. These create summary dataframes for easy comparison and plotting.
# =============================================================================

def create_best_results_dataframes(
    results_dict: Dict[str, Dict[str, Dict[str, float]]],
    reg_func_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create dataframes for best test accuracy and NLL for each model.
    
    For each model, identifies the regularization constant that yields the best
    performance according to two criteria:
    1. Highest test accuracy
    2. Lowest test NLL (negative log-likelihood)
    
    Args:
        results_dict: Nested dictionary structure:
                     {model_name: {reg_con: {metric: value}}}
        reg_func_type: Type of regularization ('L2' or 'eye_distance')
    
    Returns:
        Tuple of two DataFrames:
        - df_best_acc: Best results according to test accuracy
        - df_best_nll: Best results according to test NLL
        
        Each DataFrame has columns:
        - model_name: Name of the model
        - reg_con: Best regularization constant
        - test_acc: Test accuracy at best regularization
        - train_acc: Train accuracy at best regularization
        - train_NLL: Train NLL at best regularization
        - test_NLL: Test NLL at best regularization
        - Regularization_type: Type of regularization used
    """
    rows_acc = []
    rows_nll = []
    
    for model_name, reg_results in results_dict.items():
        # Find best by test accuracy
        best_acc_reg = max(reg_results.items(), key=lambda x: x[1]["test_acc"])
        rows_acc.append({
            "model_name": model_name,
            "reg_con": best_acc_reg[0],
            "test_acc": best_acc_reg[1]["test_acc"],
            "train_acc": best_acc_reg[1]["train_acc"],
            "train_NLL": best_acc_reg[1]["train_NLL"],
            "test_NLL": best_acc_reg[1]["test_NLL"],
            "Regularization_type": reg_func_type
        })
        
        # Find best by test NLL
        best_nll_reg = min(reg_results.items(), key=lambda x: x[1]["test_NLL"])
        rows_nll.append({
            "model_name": model_name,
            "reg_con": best_nll_reg[0],
            "test_acc": best_nll_reg[1]["test_acc"],
            "train_acc": best_nll_reg[1]["train_acc"],
            "train_NLL": best_nll_reg[1]["train_NLL"],
            "test_NLL": best_nll_reg[1]["test_NLL"],
            "Regularization_type": reg_func_type
        })
    
    return pd.DataFrame(rows_acc), pd.DataFrame(rows_nll)


def combine_regularization_results(
    results_eye: Dict[str, Dict[str, Dict[str, float]]],
    results_l2: Dict[str, Dict[str, Dict[str, float]]],
    zero_shot_results: Optional[Dict[str, Dict[str, float]]] = None
) -> pd.DataFrame:
    """
    Combine results from different regularization methods into a single dataframe.
    
    This creates a unified view of model performance across different regularization
    approaches, facilitating comparison between L2, eye-distance, and zero-shot.
    
    Args:
        results_eye: Results using eye-distance (scalar-matrix shrinkage) regularization
        results_l2: Results using standard L2 (Frobenius norm) regularization
        zero_shot_results: Optional results from zero-shot evaluation (W = Identity)
    
    Returns:
        Combined DataFrame with all results. Each model appears multiple times
        (once per regularization type), with the best hyperparameter for each.
        
    Note:
        This allows direct comparison of how different regularization methods
        affect model-behavior alignment (as discussed in paper Section 2.1).
    """
    # Get best results for each regularization type
    df_acc_eye, df_nll_eye = create_best_results_dataframes(results_eye, "eye_distance")
    df_acc_l2, df_nll_l2 = create_best_results_dataframes(results_l2, "L2")
    
    # Combine accuracy-based selections
    combined_df = pd.concat([df_acc_eye, df_acc_l2], axis=0, ignore_index=True)
    
    # Add zero-shot results if provided
    if zero_shot_results:
        zero_shot_rows = []
        for model_name, metrics in zero_shot_results.items():
            zero_shot_rows.append({
                "model_name": model_name,
                "reg_con": "zero_shot",
                "test_acc": metrics["test_acc"],
                "train_acc": metrics["train_acc"],
                "train_NLL": metrics["train_NLL"],
                "test_NLL": metrics["test_NLL"],
                "Regularization_type": "zero_shot"
            })
        zero_shot_df = pd.DataFrame(zero_shot_rows)
        combined_df = pd.concat([combined_df, zero_shot_df], axis=0, ignore_index=True)
    
    return combined_df

def fix_constraint_type_format(constraint: Any) -> Tuple[str, Any]:
    """
    Standardize constraint format and generate constraint name.
    
    Constraints control the structure of the transformation matrix W:
    - None → full_W: Unconstrained p×p matrix (maximum flexibility)
    - "zero_shot": Identity matrix (no learning)
    - "diagonal": Diagonal matrix only (scales each feature independently)
    - ["Rectangular", k]: Rectangular matrix (projects to k dimensions)
    
    This function handles various input formats (string, tuple, list, ListConfig)
    and converts them to a standardized tuple format for consistent processing.
    
    Args:
        constraint: Constraint specification in any supported format
        
    Returns:
        Tuple of (constraint_name, standardized_constraint)
        - constraint_name: String representation for file naming (e.g., "Rectangular_50")
        - standardized_constraint: Tuple format for consistent processing
        
    Examples:
        >>> fix_constraint_type_format(None)
        ('full_W', None)
        
        >>> fix_constraint_type_format(['Rectangular', 50])
        ('Rectangular_50', ('Rectangular', 50))
        
        >>> fix_constraint_type_format('diagonal')
        ('diagonal', 'diagonal')
    """
    if constraint is None:
        return "full_W", None
    
    if constraint == "zero_shot":
        return "zero_shot", "zero_shot"
    
    if isinstance(constraint, str):
        return constraint, constraint
    
    # Handle tuple format (already correct)
    if isinstance(constraint, tuple):
        if len(constraint) == 2:
            constraint_name = f"{constraint[0]}_{constraint[1]}"
            return constraint_name, constraint
        else:
            return str(constraint), constraint
    
    # Handle list or ListConfig format - convert to tuple
    if isinstance(constraint, (list, ListConfig)):
        if len(constraint) == 2:
            # Convert to tuple for consistency
            constraint_tuple = (constraint[0], constraint[1])
            constraint_name = f"{constraint[0]}_{constraint[1]}"
            return constraint_name, constraint_tuple
        else:
            # Handle other list lengths
            constraint_name = "_".join(str(x) for x in constraint)
            constraint_tuple = tuple(constraint)
            return constraint_name, constraint_tuple
    
    # Fallback for other types
    return str(constraint), constraint

# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================
# Orchestrates the complete evaluation workflow for a single constraint type.
# Handles result caching, parallel evaluation across models, and summary generation.
# =============================================================================

def run_evaluation_pipeline(
    models: List[str],
    reg_cons: List[float],
    reg_funcs: List[str],
    constraint_config: Dict[str, Any],
    folds: List[Tuple[OddOneOutDataset, OddOneOutDataset]],
    features_dir: Path,
    results_dir: Path,
    device: str = "cuda",
    feature_type: str = "full"
) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline for a single constraint configuration.
    
    This function orchestrates the entire evaluation workflow:
    1. Check for existing results to avoid redundant computation
    2. Load pre-extracted model features
    3. Evaluate each model with cross-validation
    4. Save results incrementally (per model) to prevent data loss
    5. Generate summary dataframes for easy comparison
    
    The pipeline supports two evaluation modes:
    - Zero-shot: W = Identity (no learning)
    - Regular: Learn W with specified regularization and constraints
    
    Args:
        models: List of model names to evaluate (e.g., ['ResNet50', 'ViT_L/16'])
        reg_cons: Regularization constants to try (e.g., [1e-6, 1e-5, ..., 10])
        reg_funcs: Regularization functions (e.g., ['L2', 'eye_distance'])
        constraint_config: Dictionary with:
            - 'constraint': Transformation constraint (None, 'diagonal', etc.)
            - 'save_predictions': Whether to save probabilistic predictions
        folds: Cross-validation folds from _load_or_create_folds()
        features_dir: Directory containing pre-extracted model features (*.pt files)
        results_dir: Root directory for saving results
        device: 'cuda' or 'cpu'
        feature_type: 'full' or 'PCA_500' (dimensionality reduction)
    
    Returns:
        Dictionary with evaluation results:
        {
            'eye_distance': {model_name: {reg_con: metrics}},
            'L2': {model_name: {reg_con: metrics}},
            'zero_shot': {model_name: metrics}  # if constraint == 'zero_shot'
        }
    
    Output Files:
        For regular evaluation (non-zero-shot):
            - mean_acc_train_val_{reg_func}_{constraint}_{feature_type}.json
            - best_test_acc_{reg_func}_{feature_type}_{constraint}.csv
            - best_test_nll_{reg_func}_{feature_type}_{constraint}.csv
            - best_test_acc_for_each_model_{feature_type}_{constraint}.csv
            - predictions/{model_name}/{reg_func}/{reg_con}/preds_{fold}.pt (optional)
        
        For zero-shot evaluation:
            - zero_shot_results_{feature_type}.json
            - zero_shot_results_{feature_type}.csv
            - predictions/{model_name}/zero_shot/preds_{fold}.pt (optional)
    
    Note:
        Results are cached and automatically skipped if they already exist.
        This enables incremental runs when adding new models or if interrupted.
    """
    # Extract constraint and save_predictions setting
    constraint = constraint_config.get("constraint")
    save_predictions = constraint_config.get("save_predictions", False)
    
    # Use standardized constraint formatting
    constraint_name, standardized_constraint = fix_constraint_type_format(constraint)
    
    save_dir = Path(results_dir) / feature_type / constraint_name
    save_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = save_dir / "predictions" if save_predictions else None
    
    # Initialize results storage
    all_results = {
        "eye_distance": {},
        "L2": {},
        "zero_shot": {}
    }
    
    # Handle zero-shot evaluation
    if standardized_constraint == "zero_shot":
        print("Evaluating zero-shot models...")
        
        # Check for existing results and filter models
        models_to_run = []
        skipped_models = []
        
        for model_name in models:
            if _check_model_results_exist(
                model_name, constraint_config, Path(save_dir), feature_type, 
                reg_funcs, reg_cons,  # Add these missing arguments
                predictions_dir=Path(predictions_dir) if predictions_dir else None, 
                k_folds=len(folds)):
                # Try to load existing results
                json_path = Path(save_dir) / f"zero_shot_results_{feature_type}.json"
                try:
                    with open(json_path, 'r') as f:
                        existing_results = json.load(f)
                    all_results["zero_shot"][model_name] = existing_results[model_name]
                    print(f"  ✓ Skipping {model_name} - results already exist")
                    skipped_models.append(model_name)
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    print(f"  ⚠ Warning: Results verification passed but loading failed for {model_name}: {e}")
                    print(f"    Will re-run evaluation for {model_name}")
                    models_to_run.append(model_name)
            else:
                models_to_run.append(model_name)
        
        print(f"Running evaluation for {len(models_to_run)}/{len(models)} models")
        print(f"Skipped {len(skipped_models)} models with existing results")
        
        if len(models_to_run) == 0:
            print("  → All models already have complete results!")
        
        model_pbar = tqdm(models_to_run, desc="Models", position=0)
        for model_name in model_pbar:
            model_pbar.set_description(f"Models: {model_name}")
            model_features = _load_tensor(str(features_dir / f"{model_name}.pt"))
            results = evaluate_zero_shot(
                model_name, model_features, folds, device,
                save_predictions=save_predictions, predictions_dir=predictions_dir
            )
            all_results["zero_shot"][model_name] = results
        
        # Save zero-shot results
        zero_shot_df = pd.DataFrame(all_results["zero_shot"]).T
        zero_shot_df.index.name = "model_name"
        zero_shot_df.reset_index(inplace=True)
        zero_shot_df.to_csv(save_dir / f"zero_shot_results_{feature_type}.csv", index=False)
        
        with open(save_dir / f"zero_shot_results_{feature_type}.json", "w") as f:
            json.dump(all_results["zero_shot"], f, indent=2)
        
        return all_results
    
    # Regular evaluation with regularization
    # Check for existing results and filter models
    models_to_run = []
    skipped_models = []
    
    for model_name in models:
        if _check_model_results_exist(
            model_name, constraint_config, Path(save_dir), feature_type, 
            reg_funcs, reg_cons, 
            predictions_dir=Path(predictions_dir) if predictions_dir else None, 
            k_folds=len(folds)
        ):
            # Try to load existing results for each regularization function
            try:
                for reg_func in reg_funcs:
                    json_path = Path(save_dir) / f"mean_acc_train_val_{reg_func}_{constraint_name}_{feature_type}.json"
                    with open(json_path, 'r') as f:
                        existing_results = json.load(f)
                    all_results[reg_func][model_name] = existing_results[model_name]
                print(f"  ✓ Skipping {model_name} - results already exist")
                skipped_models.append(model_name)
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"  ⚠ Warning: Results verification passed but loading failed for {model_name}: {e}")
                print(f"    Will re-run evaluation for {model_name}")
                models_to_run.append(model_name)
        else:
            models_to_run.append(model_name)
    
    print(f"Running evaluation for {len(models_to_run)}/{len(models)} models")
    print(f"Skipped {len(skipped_models)} models with existing results")
    
    if len(models_to_run) == 0:
        print("  → All models already have complete results!")
    
    model_pbar = tqdm(models_to_run, desc="Models", position=0)
    for model_name in model_pbar:
        model_pbar.set_description(f"Models: {model_name}")
        model_features = _load_tensor(str(features_dir / f"{model_name}.pt"))
        
        for reg_func in tqdm(reg_funcs, desc="Reg Types", position=1, leave=False):
            results = evaluate_model_cv(
                model_name, model_features, folds, reg_cons, reg_func,
                standardized_constraint, device, save_predictions=save_predictions, predictions_dir=predictions_dir
            )
            all_results[reg_func][model_name] = results
            
            # Save intermediate results
            json_path = save_dir / f"mean_acc_train_val_{reg_func}_{constraint_name}_{feature_type}.json"
            _save_json_atomic(all_results[reg_func], json_path)
    
    # Create summary dataframes
    for reg_func in reg_funcs:
        df_acc, df_nll = create_best_results_dataframes(all_results[reg_func], reg_func)
        df_acc.to_csv(save_dir / f"best_test_acc_{reg_func}_{feature_type}_{constraint_name}.csv", index=False)
        df_nll.to_csv(save_dir / f"best_test_nll_{reg_func}_{feature_type}_{constraint_name}.csv", index=False)
    
    # Create combined results
    combined_df = combine_regularization_results(
        all_results["eye_distance"],
        all_results["L2"]
    )
    combined_df["feature_name"] = feature_type
    combined_df["constraint_name"] = constraint_name
    combined_df.to_csv(save_dir / f"best_test_acc_for_each_model_{feature_type}_{constraint_name}.csv", index=False)
    
    return all_results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
# Hydra-based main function that loads configuration and orchestrates the
# complete evaluation across all models, constraints, and feature types.
# =============================================================================

@hydra.main(version_base=None, config_path="../scripts_configurations", config_name="evaluate_THINGS_OOO_accuracy_CV.yaml")
def main(cfg: DictConfig):
    """
    Main function for evaluating model performance on THINGS odd-one-out dataset.
    
    This is the top-level entry point that orchestrates the entire evaluation pipeline.
    It loads human behavioral data, sets up cross-validation folds, and evaluates
    multiple neural network models under various configurations.
    
    The function is configured via Hydra using evaluate_THINGS_OOO_accuracy_CV.yaml,
    which specifies:
    - Which models to evaluate (from shared_args.yaml)
    - Regularization methods and constants
    - Transformation constraints (full_W, diagonal, rectangular, zero_shot)
    - Cross-validation settings
    - Paths to data and outputs
    
    Workflow:
        1. Load THINGS odd-one-out triplets and human judgments
        2. Create/load cross-validation folds with disjoint image sets
        3. For each feature type (full features or PCA):
           4. For each transformation constraint:
              5. For each model:
                 6. Load pre-extracted features
                 7. Fit linear transformation W
                 8. Evaluate on held-out images
                 9. Save results and predictions
              10. Generate summary statistics
    
    Configuration Keys (from YAML):
        models: List of model names (from shared_args.yaml)
        additional_models: Optional extended model set
        used_additional_models: Boolean to include additional models
        
        paths:
            train_triplets: Path to training triplet indices
            train_answers: Path to training answers
            test_triplets: Path to test/noise ceiling triplets
            test_answers: Path to test answers
            features_dir: Directory with pre-extracted features
            partitions_dir: Directory for CV fold definitions
            results_out: Output directory for results
        
        constraints: Dictionary of constraint configurations, each with:
            constraint: Transformation type (None/'diagonal'/['Rectangular', k]/'zero_shot')
            save_predictions: Whether to save probabilistic predictions
        
        feature_types: List of feature types to evaluate ['full', 'PCA_500']
        k_folds: Number of cross-validation folds (typically 3)
        N_concepts: Number of images in THINGS (1854)
        random_seed: Seed for reproducible fold creation
        
        reg_cons: List of regularization constants
        reg_func_types: List of regularization functions ['L2', 'eye_distance']
    
    Output Structure:
        Results/
            {results_out}/
                {feature_type}/          # 'full' or 'PCA_500'
                    {constraint_name}/    # 'full_W', 'diagonal', 'Rectangular_50', etc.
                        mean_acc_train_val_{reg_func}_{constraint}_{feature_type}.json
                        best_test_acc_{reg_func}_{feature_type}_{constraint}.csv
                        best_test_nll_{reg_func}_{feature_type}_{constraint}.csv
                        best_test_acc_for_each_model_{feature_type}_{constraint}.csv
                        predictions/      # If save_predictions=True
                            {model_name}/
                                {reg_func}/
                                    {reg_con}/
                                        preds_{fold_idx}.pt
    
    Example Usage:
        # Run with default configuration
        python evaluate_THINGS_OOO_accuracy_CV.py
        
        # Override specific settings
        python evaluate_THINGS_OOO_accuracy_CV.py k_folds=5 device=cpu
        
        # Use additional models
        python evaluate_THINGS_OOO_accuracy_CV.py used_additional_models=True
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = cfg.models if not cfg.used_additional_models else cfg.models + cfg.additional_models
    # Load datasets
    triplets_tensor, triplets_answers_tensor, test_dataset = _load_datasets(cfg.paths)
    
    # Load or create folds
    folds = _load_or_create_folds(
        triplets_tensor, triplets_answers_tensor,
        cfg.paths, cfg.k_folds,
        cfg.N_concepts, cfg.random_seed
    )
    
    # Process each feature type
    for feature_type in cfg.feature_types:
        print(f"\nProcessing feature type: {feature_type}")
        
        # Setup feature directory
        if feature_type == "PCA_500":
            features_dir = Path(cfg.paths.features_dir) / "PCA_500"
        else:
            features_dir = Path(cfg.paths.features_dir) / "full"
        
        # Process each constraint
        for constraint_name, constraint_config in tqdm(cfg.constraints.items(), desc="Constraints", position=0):
            print(f"\n{'='*60}")
            print(f"Evaluating constraint: {constraint_name}")
            print(f"  Constraint type: {constraint_config.get('constraint')}")
            print(f"  Save predictions: {constraint_config.get('save_predictions', False)}")
            print(f"  Feature type: {feature_type}")
            print(f"{'='*60}")
            
            results = run_evaluation_pipeline(
                models=models,
                reg_cons=cfg.reg_cons,
                reg_funcs=cfg.reg_func_types,
                constraint_config=constraint_config,
                folds=folds,
                features_dir=features_dir,
                results_dir=Path(cfg.paths.results_out),
                device=device,
                feature_type=feature_type
            )
    
    print("\nEvaluation pipeline completed successfully.")


if __name__ == "__main__":
    main() 