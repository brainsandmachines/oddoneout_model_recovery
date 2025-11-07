from __future__ import annotations
from pathlib import Path
import os
import sys
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PARENT_DIR)  # We want the script to work with relative paths as if script is one level up
sys.path.append(str(PARENT_DIR))
import json
import re
from typing import Dict, List, Tuple, Optional
import hydra
from omegaconf import DictConfig, ListConfig
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm

from tools.Cog_model import CogModel
from tools.triplets_data_set import OddOneOutDataset
from tools.model_io import save_cog_model

# -----------------------------------------------------------------------------
# --- Utility helpers ---------------------------------------------------------
# -----------------------------------------------------------------------------

def maybe_parse_tuple(s):
    """
    Parse string representations of tuples into actual tuple objects.
    
    Detects and parses strings like '(label,number)' or 'label_number' 
    (e.g., "Rectangular_30") into tuples ('label', number).
    Leaves everything else unchanged.
    
    Examples:
        "(Rectangular,30)" -> ('Rectangular', 30)
        "Rectangular_30" -> ('Rectangular', 30)
        "rectangular_50" -> ('rectangular', 50)
        "full_W" -> 'full_W'
    """
    if not isinstance(s, str):
        return s  # already not a string, skip

    # Pattern 1: Parenthesized format "(something,number)"
    pattern_parens = r"^\(\s*['\"]?([\w\s-]+)['\"]?\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)$"
    match = re.match(pattern_parens, s)
    if match:
        label = match.group(1).strip()
        num_str = match.group(2)
        num = int(num_str) if num_str.isdigit() else float(num_str)
        return (label, num)
    
    # Pattern 2: Underscore format "Rectangular_30", "rectangular_50", etc.
    pattern_underscore = r"^([Rr]ectangular)_([0-9]+(?:\.[0-9]+)?)$"
    match = re.match(pattern_underscore, s, re.IGNORECASE)
    if match:
        label = match.group(1)  # Preserve original case
        num_str = match.group(2)
        num = int(num_str) if num_str.isdigit() else float(num_str)
        return (label, num)
    
    # No match, return as-is
    return s


def _ensure_dirs(paths: Dict[str, str]):
    """
    Ensure that necessary output directories exist, creating them if they don't.
    
    Parameters:
        paths (Dict[str, str]): Dictionary containing paths to directories that should exist,
                               typically containing at least 'models_out' and 'analysis_out'
    """
    Path(paths["models_out"]).mkdir(parents=True, exist_ok=True)
    Path(paths["analysis_out"]).mkdir(parents=True, exist_ok=True)


def _load_tensor(path: str) -> torch.Tensor:
    """
    Load a PyTorch tensor from a file.
    
    Parameters:
        path (str): Path to the file containing a saved tensor
        
    Returns:
        torch.Tensor: The loaded tensor
    """
    return torch.load(path)


def _build_datasets(paths: Dict[str, str]) -> Tuple[OddOneOutDataset, OddOneOutDataset]:
    """
    Build training and testing datasets from stored triplets and answers.
    
    Parameters:
        paths (Dict[str, str]): Dictionary containing paths to the required data files,
                               must include 'train_triplets', 'train_answers', 
                               'test_triplets', and 'test_answers'
                               
    Returns:
        Tuple[OddOneOutDataset, OddOneOutDataset]: A tuple containing (train_dataset, test_dataset)
    """
    train_triplets = _load_tensor(paths["train_triplets"])
    train_answers = _load_tensor(paths["train_answers"])
    test_triplets = _load_tensor(paths["test_triplets"])
    test_answers = _load_tensor(paths["test_answers"])
    return (
        OddOneOutDataset(train_triplets, train_answers),
        OddOneOutDataset(test_triplets, test_answers),
    )


def _save_json_atomic(obj: Dict, path: Path):
    """
    Save a dictionary to a JSON file atomically to prevent data corruption.
    
    This function first writes to a temporary file and then renames it to the final path,
    which is an atomic operation in most filesystems, ensuring that the file is either
    completely written or not changed at all.
    
    Parameters:
        obj (Dict): The dictionary object to save
        path (Path): The target file path
    """
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f)
    tmp.replace(path)


def _append_csv(row: Dict, path: Path):
    """
    Append a single row to a CSV file, creating the file with headers if it doesn't exist.
    
    Parameters:
        row (Dict): Dictionary representing a single row of data
        path (Path): Path to the CSV file
    """
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)

def _check_if_results_are_full(results: Dict[str, Dict[str, Dict]],models: List[str],reg_cons: List[float])->bool:
    """
    Check if results dictionary contains entries for all models and regularization constants.
    
    This function verifies that the results dictionary has complete entries for all specified
    models and regularization constants, which is useful for determining if cross-validation
    needs to be run or can be skipped because results already exist.
    
    Parameters:
        results (Dict[str, Dict[str, Dict]]): Dictionary containing cross-validation results
        models (List[str]): List of model names to check
        reg_cons (List[float]): List of regularization constants to check
        
    Returns:
        bool: True if results are complete for all models and regularization constants, False otherwise
    """
    # Check if all models exist in results
    if not all(model in results for model in models):
        return False
    
    # For each model, check if all regularization constants exist
    for model in models:
        # Convert reg_cons to strings for dictionary key comparison
        reg_cons_str = [str(reg) for reg in reg_cons]
        # Check if all regularization constants exist for this model
        if not all(str(reg) in results[model] for reg in reg_cons_str):
            return False
    
    # If we get here, all models and regularization constants are present
    return True
# -----------------------------------------------------------------------------
# --- Phase 1: Cross‑validation ----------------------------------------------
# -----------------------------------------------------------------------------

def run_cross_validation(
    models: List[str],
    reg_cons: List[float],
    train_ds: OddOneOutDataset,
    reg_func:str,
    k_fold: int,
    device: str,
    paths: Dict[str, str],
    constraints: Optional[str] = None,
) -> Dict[str, Dict[str, Dict]]:
    """
    Perform K-fold cross-validation for a set of models and regularization constants.
    
    This function evaluates each model with each regularization constant using K-fold
    cross-validation. Results are saved to both JSON and CSV files for later analysis.
    The function can resume from previous runs if results already exist.
    
    Parameters:
        models (List[str]): List of model names to evaluate
        reg_cons (List[float]): List of regularization constants to evaluate
        train_ds (OddOneOutDataset): Training dataset
        k_fold (int): Number of cross-validation folds
        device (str): Device to use for computation ('cuda' or 'cpu')
        paths (Dict[str, str]): Dictionary of paths, including 'features_dir', 'analysis_out'
        constraints (Optional[str], optional): Constraints on model parameter matrix (e.g., 'diagonal'). 
                                             Defaults to None.
        reg_func (Optional[str], optional): Regularization function type. Defaults to "eye_distance".
    
    Returns:
        Dict[str, Dict[str, Dict]]: Nested dictionary containing cross-validation results for each
                                  model and regularization constant
    """
    cv_json = Path(paths["analysis_out"]) / f"data_generating_models_evaluations_full_{reg_func}.json"
    cv_csv = Path(paths["analysis_out"]) / f"data_generating_models_evaluations_full_{reg_func}.csv"
    results = json.loads(cv_json.read_text()) if cv_json.exists() else {}
    #check if the results are already exist and full
    if _check_if_results_are_full(results,models,reg_cons):
        print("Results are already exist and full")
        return results
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    folds = list(kf.split(torch.arange(len(train_ds))))

    for model in tqdm(models, desc="CV‑Models", position=0):
        features = _load_tensor(Path(paths["features_dir"]) / f"{model}.pt")
        results.setdefault(model, {})
        # Update the outer progress bar with current model name
        tqdm.write(f"Processing model: {model}")
        for reg in tqdm(reg_cons, desc=f"λ (Model: {model})", position=1, leave=False):
            if str(reg) in results[model]:
                continue  # already done (resumable)
            stats = {k: [] for k in ("tr_acc", "va_acc", "tr_nll", "va_nll")}
            for tr_idx, va_idx in tqdm(folds, desc=f"folds (λ={reg})", position=2, leave=False):
                tr_ds = train_ds.subset(tr_idx, device)
                va_ds = train_ds.subset(va_idx, device)
                cog = CogModel(model, images_features_dim=features.shape[1], device=device,constraints=constraints)
                cog.fit_W_matrix(tr_ds, features, reg_func=reg_func, reg_con=reg, verbose=False)
                va_acc, va_nll = cog.test_model(va_ds, features, batch_size=len(va_ds))
                tr_acc, tr_nll = cog.test_model(tr_ds, features, batch_size=len(tr_ds))
                stats["va_acc"].append(va_acc.item())
                stats["tr_acc"].append(tr_acc.item())
                stats["va_nll"].append(va_nll.item())
                stats["tr_nll"].append(tr_nll.item())
                del cog
                torch.cuda.empty_cache()
            res = {
                "mean_train_acc": float(np.mean(stats["tr_acc"])),
                "std_train_acc": float(np.std(stats["tr_acc"])),
                "mean_val_acc": float(np.mean(stats["va_acc"])),
                "std_val_acc": float(np.std(stats["va_acc"])),
                "mean_train_nll": float(np.mean(stats["tr_nll"])),
                "std_train_nll": float(np.std(stats["tr_nll"])),
                "mean_val_nll": float(np.mean(stats["va_nll"])),
                "std_val_nll": float(np.std(stats["va_nll"])),
                "embedding_size": features.shape[1],
                "reg_func": reg_func
            }
            results[model][str(reg)] = res
            _save_json_atomic(results, cv_json)
            _append_csv({"model_name": model, "reg_con": reg, "reg_func": reg_func, **res}, cv_csv)
    return results


# -----------------------------------------------------------------------------
# --- Phase 2: Model‑selection ------------------------------------------------
# -----------------------------------------------------------------------------

def select_best_regularization(results: Dict[str, Dict[str, Dict]]) -> Dict[str, float]:
    """
    Select the best regularization constant for each model based on validation accuracy.
    
    This function analyzes cross-validation results to determine which regularization constant
    produces the highest validation accuracy for each model, excluding the zero regularization case.
    
    Parameters:
        results (Dict[str, Dict[str, Dict]]): Nested dictionary containing cross-validation results
                                            from run_cross_validation
    
    Returns:
        Dict[str, float]: Dictionary mapping each model name to its best regularization constant
        
    Notes:
        - Regularization constant 0.0 is excluded from selection
        - Selection is based on the 'mean_val_acc' metric
    """
    best = {}
    for model, reg_dict in results.items():
        candidates = {float(r): v for r, v in reg_dict.items() if float(r) != 0.0}
        best_reg = max(candidates, key=lambda r: candidates[r]["mean_val_acc"])
        best[model] = best_reg
    return best


# -----------------------------------------------------------------------------
# --- Phase 3: Training & Calibration ----------------------------------------
# -----------------------------------------------------------------------------

def train_and_calibrate(
    models: List[str],
    best_reg: Dict[str, float],
    reg_func: str, 
    train_ds: OddOneOutDataset,
    test_ds: OddOneOutDataset,
    device: str,
    paths: Dict[str, str],
    noise_ceiling: float,
    calibrate_only: bool = False,
    constraints: Optional[str] = None,
    
):
    """
    Train (optional) and/or calibrate temperature for each model.
    
    This function performs the final training and temperature calibration for models using
    the best regularization constants identified from cross-validation. Results are saved
    to both JSON and CSV files for later analysis.
    
    Parameters:
        models (List[str]): List of model names to train and calibrate
        best_reg (Dict[str, float]): Dictionary mapping each model to its best regularization constant
        train_ds (OddOneOutDataset): Training dataset
        test_ds (OddOneOutDataset): Test dataset for calibration and evaluation
        device (str): Device to use for computation ('cuda' or 'cpu')
        paths (Dict[str, str]): Dictionary of paths including 'features_dir', 'models_out', 'analysis_out'
        noise_ceiling (float): Target noise ceiling value for temperature calibration
        calibrate_only (bool, optional): If True, skip W-matrix fitting and just calibrate temperature.
                                       Defaults to False.
        constraints (Optional[str], optional): Constraints on model parameter matrix (e.g., 'diagonal').
                                             Defaults to None.
        reg_func (Optional[str], optional): Regularization function to use for W-matrix fitting.
                                          Defaults to "eye_distance".
    """
    out_json = Path(paths["analysis_out"]) / f"data_generating_models_evaluations_final_{reg_func}.json"
    out_csv = Path(paths["analysis_out"]) / f"data_generating_models_results_final_{reg_func}.csv"
    finals = json.loads(out_json.read_text()) if out_json.exists() else {}

    for model in tqdm(models, desc="Calib‑Models", position=0):
        if model in finals and calibrate_only:
            continue  # already calibrated

        feat_path = Path(paths["features_dir"]) / f"{model}.pt"
        features = _load_tensor(feat_path)

        # -----------------------------------------------------------------
        # Load existing model or initialise a fresh one
        # -----------------------------------------------------------------
        model_path = Path(paths["models_out"]) / f"{model}.pt"
        if calibrate_only and model_path.exists():
            cog = torch.load(model_path).to(device)  # relies on save_cog_model using torch.save
        else:
            cog = CogModel(model, images_features_dim=features.shape[1], device=device,constraints=constraints)
            if not calibrate_only and model != "VICE":
                cog.fit_W_matrix(train_ds, features, reg_func=reg_func, reg_con=best_reg[model])

        # -----------------------------------------------------------------
        # Temperature calibration
        # -----------------------------------------------------------------
        cog.fit_model_temp(
            test_ds,
            features,
            num_epochs=100,
            batch_size=len(test_ds),
            fit_to_noise_ceiling=True,
            target_noise_ceiling=noise_ceiling,
            verbose=False,
        )

        # -----------------------------------------------------------------
        # Evaluation (skip train set when calibrate‑only)
        # -----------------------------------------------------------------
        if calibrate_only:
            tr_acc = tr_nll = float("nan")
        else:
            tr_acc, tr_nll = cog.test_model(train_ds, features, batch_size=len(train_ds))

        te_acc, te_nll = cog.test_model(test_ds, features, batch_size=len(test_ds))

        # Persist model & metrics
        save_cog_model(cog, paths["models_out"])
        res = {
            "reg_con": best_reg.get(model, 0.0),
            "reg_func": reg_func,
            "train_acc": None if calibrate_only else tr_acc.item(),
            "train_nll": None if calibrate_only else tr_nll.item(),
            "test_acc": te_acc.item(),
            "test_nll": te_nll.item(),
            "temp": cog.temp.item(),
            "embedding_size": features.shape[1],
            "calibrate_only": calibrate_only,
        }
        finals[model] = res
        _save_json_atomic(finals, out_json)
        _append_csv({"model_name": model, "reg_func": reg_func, **res}, out_csv)

# -----------------------------------------------------------------------------
# --- Main --------------------------------------------------------------------
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../scripts_configurations", config_name="create_data_generating_models.yaml")
def main(cfg: DictConfig):
    """
    Main function for creating and calibrating data-generating models.
    
    This function orchestrates the process of creating data-generating models for the model
    recovery experiments. It follows a three-phase approach:
    1. Cross-validation to identify optimal regularization constants
    2. Model selection using validation performance
    3. Final training and temperature calibration
    
    The function handles different constraints on the model parameters (diagonal, rectangular, etc.)
    and can optionally skip training to just perform calibration.
    
    Parameters:
        cfg (DictConfig): Hydra configuration object containing experiment parameters
        
    Notes:
        - Configuration is loaded from a YAML file via Hydra
        - Directory paths are set up based on configuration and constraint types
        - For 'zero_shot' constraints or when calibrate_only=True, cross-validation is skipped
        - Results are saved in both model files (.pt) and analysis files (JSON, CSV)
        - Different constraint types are processed sequentially
    """
    paths = cfg.paths
    models_out_path = cfg.paths["models_out"] if cfg.reg_func == "eye_distance" else cfg.paths["models_out"][:-1]+f"_{cfg.reg_func}/"
    analysis_out_path = cfg.paths["analysis_out"] if cfg.reg_func == "eye_distance" else cfg.paths["analysis_out"][:-1]+f"_{cfg.reg_func}/"
    models_features_path = cfg.paths["features_dir"]
    if cfg.PCA_500_Features == True:
        models_features_path = os.path.join(models_features_path,"PCA_500")
        analysis_out_path = os.path.join(analysis_out_path,"PCA_500")
        models_out_path = os.path.join(models_out_path,"PCA_500")
    else:
        models_features_path = os.path.join(models_features_path,"full")
        analysis_out_path = os.path.join(analysis_out_path,"full")
        models_out_path = os.path.join(models_out_path,"full")
    paths["features_dir"] = models_features_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Determine models list based on include_new_models flag in config
    if cfg.include_new_models:
        models_list = cfg.models + cfg.new_models
    else:
        models_list = cfg.models
        
    for constraint in cfg.constraints_list:
        print(f"Running with constraint: {constraint}")
        print(f"Running with models: {models_list}")
        print(f"Running with reg_cons: {cfg.reg_cons}")
        print(f"Running with reg_func: {cfg.reg_func}")
        print(f"Running with PCA reduced features to 500 : {cfg.PCA_500_Features}")
        
        # To handle the rectangular constraint, we need to break to parse the constraint correctly from string to tuples.
        # Handle any '(something,number)' string to a tuple of (string,number)
        constraint = maybe_parse_tuple(constraint)
        if isinstance(constraint, tuple):
            paths["models_out"] = os.path.join(models_out_path,f"models_{constraint[0]}_{constraint[1]}")
            paths["analysis_out"] = os.path.join(analysis_out_path,f"analysis_{constraint[0]}_{constraint[1]}")
        else:
            paths["models_out"] = os.path.join(models_out_path,f"models_{constraint}")
            paths["analysis_out"] = os.path.join(analysis_out_path,f"analysis_{constraint}")
        _ensure_dirs(paths)
        train_ds, test_ds = _build_datasets(paths)
        
        if cfg.calibrate_only or constraint == "zero_shot":
            dummy_best = {m: 0.0 for m in models_list}
            train_and_calibrate(
                models_list,
                dummy_best,
                cfg.reg_func,
                train_ds,
                test_ds,
                device,
                paths,
                cfg.noise_ceiling,
                calibrate_only=True,
                constraints=constraint,
            )
        else:
            cv_results = run_cross_validation(
                models_list,
                cfg.reg_cons,
                train_ds,
                cfg.reg_func,
                cfg.k_fold,
                device,
                paths,
                constraints=constraint if constraint != "full_W" else None,
            )
            best_reg = select_best_regularization(cv_results)
            train_and_calibrate(
                models_list,
                best_reg,
                cfg.reg_func,
                train_ds,
                test_ds,
                device,
                paths,
                cfg.noise_ceiling,
                calibrate_only=False,
                constraints=constraint if constraint != "full_W" else None,
            )

        print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
    
