"""
===============================================================================
Figure 5 Analysis: Transformation Flexibility vs. Model Recovery Accuracy
===============================================================================

This script analyzes the relationship between transformation flexibility and
model recovery accuracy for Figure 5 of the paper. It evaluates how different
constraint types (zero-shot, diagonal, rectangular, full matrix) affect the
ability to correctly identify which neural network model generated behavioral
data, as a function of training set size.

Scientific Question:
-------------------
Can we reliably identify the true data-generating model when fitting various
candidate models with different transformation flexibility levels? How does
this recovery accuracy depend on the amount of training data?

Analysis Pipeline:
-----------------
For each constraint type and training set size:
1. Load cross-validation folds from preprocessed THINGS dataset
2. Sample N triplets from training set (N from n_list)
3. For each model:
   - Train transformation matrix W with multiple regularization constants
   - Select best regularization via validation set accuracy
   - Evaluate final accuracy on held-out test set
4. Query simulation database for model recovery accuracy
5. Aggregate results across all constraint types
6. Save comprehensive results to CSV files

Key Metrics:
-----------
- Test Accuracy: Performance on held-out triplets
- Model Recovery Accuracy: Percentage of correct model identifications
- Best/Worst Models: Models with highest/lowest test accuracy

Constraint Types (flexibility from rigid to flexible):
-----------------------------------------------------
- zero_shot: Identity matrix (no learning, W = I)
- diagonal: Element-wise scaling only
- Rectangular_k: Low-rank projection to k dimensions
- full_W_L1: Full matrix with L1 regularization
- full_W: Full unconstrained matrix (maximum flexibility)

Output Structure:
----------------
Results/flexibility_accuracy/
├── {constraint_type}/
│   └── train_set_size_{n}/
│       ├── best_acc_models.csv      # Best accuracy per model
│       ├── mean_acc_models.csv      # Mean accuracy across folds
│       └── folds_acc_models.csv     # Per-fold accuracies
└── final_results_df.csv             # Aggregated results

Configuration:
-------------
Controlled via scripts_configurations/figure_5_analysis.yaml
Key parameters:
- constraint_types_to_analyze: Which constraints to process
- n_list: Training set sizes to evaluate
- reg_cons: Regularization constants to test
- features_dim: "full" or "PCA_500" (dimensionality)

Usage:
------
python analysis_scripts/figure_5_analysis.py

# To analyze specific constraints only:
python analysis_scripts/figure_5_analysis.py constraint_types_to_analyze="['full_W','zero_shot']"

# To use different training set sizes:
python analysis_scripts/figure_5_analysis.py n_list="[400,25600,204800]"

Authors: Itamar Avitan & Tal Golan
Institution: Department of Cognitive and Brain Sciences, 
             Ben-Gurion University of the Negev
Paper: "Model-Behavior Alignment under Flexible Evaluation" (NeurIPS 2025)
===============================================================================
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from typing import List,Dict,Tuple,Optional,Union
from figures_scripts.create_Figures_1_and_3 import DBResultsAnalysis
from tools.Cog_model import CogModel
from tools.utils import OddOneOutDataset
import gc
import hydra
from omegaconf import DictConfig
from evaluate_THINGS_OOO_accuracy_CV import _load_or_create_folds
import re
from typing import Dict,List,Tuple


# =============================================================================
# Checkpoint Management and Validation Functions
# =============================================================================

def load_final_results_to_dict(final_results_path: str) -> Optional[Dict[str, Dict[int, Dict[str, Union[float, str]]]]]:
    """
    Load previously computed results from CSV to resume interrupted analysis.
    
    This function enables checkpoint recovery by loading the final results
    DataFrame and reconstructing the nested dictionary structure used during
    computation. Allows the analysis to skip already-completed configurations.
    
    Parameters
    ----------
    final_results_path : str
        Path to the final_results_df.csv file containing aggregated results
        
    Returns
    -------
    Dict[str, Dict[int, Dict[str, Union[float, str]]]] or None
        Three-level nested dictionary structure:
        - Level 1: constraint type (e.g., 'full_W', 'diagonal')
        - Level 2: training set size (e.g., 400, 25600)
        - Level 3: metrics dict with keys:
            * 'test_acc': Mean test accuracy across models
            * 'test_acc_std': Standard deviation of test accuracy
            * 'model_recovery_acc': Model recovery accuracy from simulations
            * 'max_acc_model': Name of best-performing model
            * 'max_acc_model_acc': Accuracy of best model
            * 'min_acc_model': Name of worst-performing model
            * 'min_acc_model_acc': Accuracy of worst model
        Returns None if file doesn't exist or loading fails
        
    Example
    -------
    >>> results = load_final_results_to_dict("Results/flexibility_accuracy/final_results_df.csv")
    >>> if results:
    >>>     full_w_400 = results['full_W'][400]
    >>>     print(f"Test accuracy: {full_w_400['test_acc']:.2f}%")
    """
    
    if not os.path.exists(final_results_path):
        print(f"Final results file not found at {final_results_path}")
        return None
    
    try:
        # Load the DataFrame
        final_df = pd.read_csv(final_results_path)
        
        # Initialize the dictionary structure
        final_res_mean_acc_dict_full: Dict[str, Dict[int, Dict[str, Union[float, str]]]] = {}
        
        # Get unique constraint types
        constraint_types = final_df['constraint'].unique()
        for constraint_type in constraint_types:
            n_list = final_df[final_df['constraint'] == constraint_type]['train_set_size'].unique()
            final_res_mean_acc_dict_full[constraint_type] = {}
            for n in n_list:
                final_res_mean_acc_dict_full[constraint_type][int(n)] = {
                                                                "test_acc": float(final_df[(final_df['constraint'] == constraint_type) & (final_df['train_set_size'] == n)]['test_acc'].values[0]),
                                                                "test_acc_std": float(final_df[(final_df['constraint'] == constraint_type) & (final_df['train_set_size'] == n)]['test_acc_std'].values[0]),
                                                                "model_recovery_acc": float(final_df[(final_df['constraint'] == constraint_type) & (final_df['train_set_size'] == n)]['model_recovery_acc'].values[0]),
                                                                "max_acc_model": str(final_df[(final_df['constraint'] == constraint_type) & (final_df['train_set_size'] == n)]['max_acc_model'].values[0]),
                                                                "max_acc_model_acc": float(final_df[(final_df['constraint'] == constraint_type) & (final_df['train_set_size'] == n)]['max_acc_model_acc'].values[0]),
                                                                "min_acc_model": str(final_df[(final_df['constraint'] == constraint_type) & (final_df['train_set_size'] == n)]['min_acc_model'].values[0]),
                                                                "min_acc_model_acc": float(final_df[(final_df['constraint'] == constraint_type) & (final_df['train_set_size'] == n)]['min_acc_model_acc'].values[0])
                    
                }

        return final_res_mean_acc_dict_full
    except Exception as e:
        print(f"Error loading final results: {e}")
        return None
def validate_checkpoint_and_get_missing_configs(cfg: DictConfig) -> Tuple[List[Tuple[str, int]], Optional[Dict[str, Dict[int, Dict[str, Union[float, str]]]]]]:
    """
    Validate existing results and identify missing configurations that need computation.
    
    This function performs comprehensive validation of existing analysis outputs
    by checking for:
    1. Directory existence for each (constraint, n) combination
    2. Presence of all required CSV files (best_acc, mean_acc, folds_acc)
    3. Correct column structure in each CSV
    4. Complete data for all expected models and regularization constants
    
    The function enables smart resumption of interrupted analyses by identifying
    exactly which configurations are incomplete or missing.
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing:
        - constraint_types_to_analyze: List of constraints to check
        - n_list: Training set sizes to validate
        - models: Expected model names
        - reg_cons: Expected regularization constants
        - save_path: Output directory to validate
        
    Returns
    -------
    missing_configs : List[Tuple[str, int]]
        List of (constraint_name, training_size) tuples that need computation.
        Example: [('full_W', 400), ('diagonal', 25600)]
        
    loaded_results_dict : Dict[str, Dict[int, Dict[str, Union[float, str]]]] or None
        Previously computed results loaded from final_results_df.csv, or None
        if the file doesn't exist. Has same structure as load_final_results_to_dict()
        
    Validation Checks:
    -----------------
    For each configuration, validates:
    1. **Directory exists**: save_path/{constraint}/train_set_size_{n}/
    2. **Files exist**: best_acc_models.csv, mean_acc_models.csv, folds_acc_models.csv
    3. **best_acc_models.csv**:
       - Columns: model_name, best_accuracy, best_reg_con
       - All models present
    4. **mean_acc_models.csv**:
       - Columns: model_name, reg_con, mean_accuracy
       - All models present
    5. **folds_acc_models.csv**:
       - Columns: model_name, reg_con, fold_1_accuracy, fold_2_accuracy, fold_3_accuracy
       - All (model × reg_con) combinations present
       - Exception: zero_shot only needs reg_con=0
       
    Side Effects:
    ------------
    Prints validation status for each configuration:
    - "✓ Complete results found for {constraint} with n={n}"
    - "Missing directory: {path}" 
    - "Missing files in {path}: [files]"
    - "Invalid columns in {file} at {path}"
    - "Missing models/reg_cons in {file} at {path}"
    
    Example:
    -------
    >>> missing, loaded = validate_checkpoint_and_get_missing_configs(cfg)
    >>> print(f"Need to compute {len(missing)} configurations")
    >>> if loaded:
    >>>     print(f"Loaded results for {len(loaded)} constraint types")
    """
    missing_configs = []
    expected_models = set(cfg.models)
    expected_reg_cons = set(cfg.reg_cons)
    
    for constraints in cfg.constraint_types_to_analyze:
        constraints = maybe_parse_tuple(constraints)
        if type(constraints) == tuple:
            constraints_name = f"{constraints[0]}_{constraints[1]}"
        elif constraints is None:
            constraints_name = "full_W"
        else:
            constraints_name = constraints
            
        for n in cfg.n_list:
            save_results_dir = f"{cfg.save_path}/{constraints_name}/train_set_size_{n}"
            
            # Check if directory exists
            if not os.path.exists(save_results_dir):
                print(f"Missing directory: {save_results_dir}")
                missing_configs.append((constraints_name, n))
                continue
            
            # Define expected files
            expected_files = [
                "best_acc_models.csv",
                "mean_acc_models.csv", 
                "folds_acc_models.csv"
            ]
            
            # Check if all expected files exist
            missing_files = []
            for file_name in expected_files:
                file_path = os.path.join(save_results_dir, file_name)
                if not os.path.exists(file_path):
                    missing_files.append(file_name)
            
            if missing_files:
                print(f"Missing files in {save_results_dir}: {missing_files}")
                missing_configs.append((constraints_name, n))
                continue
            
            # Validate CSV completeness
            is_complete = True
            
            # Check best_acc_models.csv
            try:
                best_acc_df = pd.read_csv(os.path.join(save_results_dir, "best_acc_models.csv"))
                if set(best_acc_df.columns) != {"model_name", "best_accuracy", "best_reg_con"}:
                    print(f"Invalid columns in best_acc_models.csv at {save_results_dir}")
                    is_complete = False
                elif set(best_acc_df["model_name"]) != expected_models:
                    print(f"Missing models in best_acc_models.csv at {save_results_dir}")
                    is_complete = False
            except Exception as e:
                print(f"Error reading best_acc_models.csv at {save_results_dir}: {e}")
                is_complete = False
            
            # Check mean_acc_models.csv
            try:
                mean_acc_df = pd.read_csv(os.path.join(save_results_dir, "mean_acc_models.csv"))
                expected_columns = {"model_name", "reg_con", "mean_accuracy"}
                if set(mean_acc_df.columns) != expected_columns:
                    print(f"Invalid columns in mean_acc_models.csv at {save_results_dir}")
                    is_complete = False
                elif set(mean_acc_df["model_name"]) != expected_models:
                    print(f"Missing models in mean_acc_models.csv at {save_results_dir}")
                    is_complete = False
            except Exception as e:
                print(f"Error reading mean_acc_models.csv at {save_results_dir}: {e}")
                is_complete = False
            
            # Check folds_acc_models.csv
            try:
                folds_acc_df = pd.read_csv(os.path.join(save_results_dir, "folds_acc_models.csv"))
                expected_columns = {"model_name", "reg_con", "fold_1_accuracy", "fold_2_accuracy", "fold_3_accuracy"}
                if set(folds_acc_df.columns) != expected_columns:
                    print(f"Invalid columns in folds_acc_models.csv at {save_results_dir}")
                    is_complete = False
                else:
                    # Check if we have all combinations of models and reg_cons
                    if constraints == "zero_shot":
                        expected_rows = len(expected_models)  # only one reg_con (0) for zero_shot
                    else:
                        expected_rows = len(expected_models) * len(expected_reg_cons)
                    
                    if len(folds_acc_df) != expected_rows:
                        print(f"Incomplete data in folds_acc_models.csv at {save_results_dir}: {len(folds_acc_df)}/{expected_rows} rows")
                        is_complete = False
                    elif set(folds_acc_df["model_name"]) != expected_models:
                        print(f"Missing models in folds_acc_models.csv at {save_results_dir}")
                        is_complete = False
                    elif set(folds_acc_df["reg_con"]) != expected_reg_cons and constraints != "zero_shot":
                        print(f"Missing reg_cons in folds_acc_models.csv at {save_results_dir}")
                        is_complete = False
            except Exception as e:
                print(f"Error reading folds_acc_models.csv at {save_results_dir}: {e}")
                is_complete = False
            
            if not is_complete:
                missing_configs.append((constraints_name, n))
            else:
                print(f"✓ Complete results found for {constraints_name} with n={n}")
    #load the final results df if exists
    final_results_path = os.path.join(cfg.save_path, "final_results_df.csv")  
    final_results_dict = load_final_results_to_dict(final_results_path)
    return missing_configs,final_results_dict


# =============================================================================
# Utility Functions
# =============================================================================

def maybe_parse_tuple(s):
    """
    Parse string representations of tuples into actual tuple objects.
    
    This function handles constraint specifications like "(Rectangular,30)"
    or "Rectangular_30" that may be stored as strings in configuration files.
    It converts them to proper Python tuples with the numeric component cast
    to int or float.
    
    Parameters
    ----------
    s : str or any
        Input to parse. If not a string, returns unchanged.
        Expected formats:
        - "(label,number)" - e.g., "(Rectangular,30)"
        - "label_number" - e.g., "Rectangular_30", "rectangular_50"
        where label is alphanumeric and number is an integer or float.
        
    Returns
    -------
    tuple or original input
        - If input matches tuple pattern: (str, int/float) tuple
        - Otherwise: returns input unchanged
        
    Examples
    --------
    >>> maybe_parse_tuple("(Rectangular,30)")
    ('Rectangular', 30)
    
    >>> maybe_parse_tuple("Rectangular_30")
    ('Rectangular', 30)
    
    >>> maybe_parse_tuple("rectangular_50")
    ('rectangular', 50)
    
    >>> maybe_parse_tuple("(PCA,500)")
    ('PCA', 500)
    
    >>> maybe_parse_tuple("full_W")
    'full_W'
    
    >>> maybe_parse_tuple(None)
    None
    
    Notes
    -----
    - The pattern allows optional quotes around the label for parenthesized format
    - Whitespace around components is stripped
    - Integers are returned as int, floats as float
    - Used for parsing constraint types from YAML configuration
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
    # Match case-insensitive "rectangular" or "Rectangular" followed by underscore and number
    pattern_underscore = r"^([Rr]ectangular)_([0-9]+(?:\.[0-9]+)?)$"
    match = re.match(pattern_underscore, s, re.IGNORECASE)
    if match:
        label = match.group(1)  # Preserve original case
        num_str = match.group(2)
        num = int(num_str) if num_str.isdigit() else float(num_str)
        return (label, num)
    
    # No match, return as-is
    return s




def evaluate_models_on_test_set(
            trained_models_dict: Dict[str,CogModel], #either model name with reg_cons dict or model name with model
            test_set: OddOneOutDataset,
            model_features_dict:Dict[str,str] #model name with model features path
            )->Dict[str,float]:
    #check if the trained_models_dict is a dict of dicts or a dict of models
    res_dict = {}
    for model_name,model in trained_models_dict.items():
        model_features = torch.load(model_features_dict[model_name])
        model.eval()
        with torch.no_grad():
            accuracy,_ = model.test_model(test_set=test_set,images_features=model_features)
            res_dict[model_name] = accuracy.detach().cpu().item()
        del model_features,model
        gc.collect()
        torch.cuda.empty_cache()
    return res_dict


def trained_models(
    model_names:List[str],
    reg_con:Union[int,float],
    train_set:OddOneOutDataset,
    model_features_dict:Dict[str,str],
    constraints: Optional[Union[str,Tuple[str,int],None]] = None,
    regularization: Optional[str] = None
    ):
    trained_models_dict = {}
    for model_name in model_names:
        model_features = torch.load(model_features_dict[model_name])
        model = CogModel(model_name=model_name,images_features_dim=model_features.shape[1],constraints=constraints)
        if constraints != "zero_shot":
            losses = model.fit_W_matrix(train_set=train_set,reg_con=reg_con,images_features=model_features,verbose=False,reg_func=regularization if regularization is not None else "eye_distance")  #If not specified use the fit function default
        else:
            assert torch.allclose(model.W_mat.weight,torch.eye(model.W_mat.weight.shape[0]).to(model.W_mat.weight.device))
        trained_models_dict[model_name] = model
        del model_features,model
        gc.collect()
        torch.cuda.empty_cache()
    return trained_models_dict

def find_models_best_accuracy_over_reg_cons(
            model_names:List[str],
            reg_cons:List[float],
            train_test_sets:List[Tuple[OddOneOutDataset,OddOneOutDataset]],
            model_features_dict:Dict[str,str],
            constraints: Optional[Union[Tuple[str,int],str]] = None,
            regularization: Optional[str] = None
                ):
    # build models_folds_accuracies_over_reg_cons dict structure
    models_folds_accuracies_over_reg_cons = {
        model_name: {
            reg_con: [] for reg_con in (reg_cons if constraints != "zero_shot" else [0])
        } 
        for model_name in model_names
    }
    models_mean_accuracy_over_reg_cons = {model_name :{} for model_name in model_names }
    models_best_accuracy_over_reg_cons = {model_name :(-np.inf,np.nan) for model_name in model_names }
    for reg_con in reg_cons if constraints != "zero_shot" else [0]:
        for i,(train_set,test_set) in enumerate(train_test_sets):
            trained_models_dict = trained_models(model_names=model_names,reg_con=reg_con,train_set=train_set,model_features_dict=model_features_dict,constraints=constraints,regularization=regularization)
            res_dict = evaluate_models_on_test_set(trained_models_dict=trained_models_dict,test_set=test_set,model_features_dict=model_features_dict)
            for model_name,accuracy in res_dict.items():
                models_folds_accuracies_over_reg_cons[model_name][reg_con].append(accuracy)
        for model_name in model_names:
            mean_accuracy = np.mean(models_folds_accuracies_over_reg_cons[model_name][reg_con]).item()
            models_mean_accuracy_over_reg_cons[model_name][reg_con] = mean_accuracy
            # Save the best accuracy and regularization constant for the model
            if mean_accuracy > models_best_accuracy_over_reg_cons[model_name][0]:
                models_best_accuracy_over_reg_cons[model_name] = (mean_accuracy,reg_con)
    
    return models_best_accuracy_over_reg_cons,models_mean_accuracy_over_reg_cons,models_folds_accuracies_over_reg_cons
    
    
def n_size_odd_one_out_sampled_set(set:OddOneOutDataset,n:int,seed:Optional[int]=42)->OddOneOutDataset:
    torch.manual_seed(seed)
    random_indices = torch.randperm(len(set))[:n]
    n_size_set = set.subset(random_indices)
    return n_size_set
    
    
def evaluate_models_on_n_size_set(
    model_names:List[str],
    n:Union[int,str],
    full_folds_sets:List[Tuple[OddOneOutDataset,OddOneOutDataset]],
    model_features_dict:Dict[str,str],
    reg_cons:List[Union[int,float]],
    constraints: Optional[Union[Tuple[str,int],str]] = None,
    regularization: Optional[str] = None
    ):
    n_size_sets = []
    if n == "full":
        n_size_sets = full_folds_sets
    else:
        # At this point, n must be an int
        assert isinstance(n, int), f"Expected n to be int, got {type(n)}"
        for train_set,test_set in full_folds_sets:
            n_size_train_set = n_size_odd_one_out_sampled_set(set=train_set,n=n,seed=42)
            n_size_sets.append((n_size_train_set,test_set))
        
    best_acc_models,mean_acc_models,folds_acc_models = find_models_best_accuracy_over_reg_cons(
                                                                                model_names=model_names,
                                                                                reg_cons=reg_cons,
                                                                                train_test_sets=n_size_sets,
                                                                                model_features_dict=model_features_dict,
                                                                                constraints=constraints,
                                                                                regularization=regularization
                                                                            )
    return best_acc_models,mean_acc_models,folds_acc_models
    
def n_specific_results_dfs(
    models_names:List[str],
    reg_cons:List[float],
    save_results_dir:str,
    best_acc_models:Optional[Dict[str,Tuple[float,float]]] = None,
    mean_acc_models:Optional[Dict[str,Dict[float,float]]] = None,
    folds_acc_models:Optional[Dict[str,Dict[float,List[float]]]] = None,
    save_results:bool = True,
    return_dfs_dict:bool = False
    ):
    os.makedirs(save_results_dir,exist_ok=True)
    #build results df
    #df_best_acc_models columns are model name, best accuracy, best reg con
    #df_mean_acc_models columns are model name, reg con, mean accuracy
    #df_folds_acc_models columns are model name, reg con, fold 1 accuracy, fold 2 accuracy, fold 3 accuracy
    df_best_acc_models = pd.DataFrame(columns=["model_name","best_accuracy","best_reg_con"])
    df_mean_acc_models = pd.DataFrame(columns=["model_name","reg_con","mean_accuracy"])
    df_folds_acc_models = pd.DataFrame(columns=["model_name","reg_con","fold_1_accuracy","fold_2_accuracy","fold_3_accuracy"])
    
    for model_name in models_names:
        if best_acc_models is not None:
            df_best_acc_models = pd.concat([df_best_acc_models, pd.DataFrame([{
                "model_name": model_name,
                "best_accuracy": best_acc_models[model_name][0],
                "best_reg_con": best_acc_models[model_name][1]
            }])], ignore_index=True)
        for reg_con in reg_cons:
            if mean_acc_models is not None:
                df_mean_acc_models = pd.concat([df_mean_acc_models, pd.DataFrame([{
                    "model_name": model_name,
                    "reg_con": reg_con,
                    "mean_accuracy": mean_acc_models[model_name][reg_con]
                }])], ignore_index=True)
            if folds_acc_models is not None:
                df_folds_acc_models = pd.concat([df_folds_acc_models, pd.DataFrame([{
                    "model_name": model_name,
                    "reg_con": reg_con,
                    "fold_1_accuracy": folds_acc_models[model_name][reg_con][0],
                    "fold_2_accuracy": folds_acc_models[model_name][reg_con][1],
                    "fold_3_accuracy": folds_acc_models[model_name][reg_con][2]
                }])], ignore_index=True)    
    # Save DataFrames if requested
    dfs_dict = {}
    if best_acc_models is not None:
        if save_results:
            df_best_acc_models.to_csv(f"{save_results_dir}/best_acc_models.csv",index=False)
            dfs_dict["best_acc_models"] = df_best_acc_models
    if mean_acc_models is not None:
        if save_results:
            df_mean_acc_models.to_csv(f"{save_results_dir}/mean_acc_models.csv",index=False)
            dfs_dict["mean_acc_models"] = df_mean_acc_models
    if folds_acc_models is not None:
        if save_results:
            df_folds_acc_models.to_csv(f"{save_results_dir}/folds_acc_models.csv",index=False)
            dfs_dict["folds_acc_models"] = df_folds_acc_models
    if return_dfs_dict:
        return dfs_dict

def save_final_results_df(final_res_mean_acc_dict_full: Dict[str, Dict[int, Dict[str, Union[float, str]]]], save_path: str):
    # Structure of the DataFrame: model_name, constraint, train_set_size, test_acc, model_recovery_acc
    df = pd.DataFrame(columns=["constraint","train_set_size","test_acc","model_recovery_acc","max_acc_model","max_acc_model_acc","min_acc_model","min_acc_model_acc"])
    for constraint,constraint_dict in final_res_mean_acc_dict_full.items():
        for train_set_size,train_set_size_dict in constraint_dict.items():
            df = pd.concat([df, pd.DataFrame([{
                "constraint": constraint,
                "train_set_size": train_set_size if train_set_size != "full" else 1638400,
                "test_acc": train_set_size_dict["test_acc"],
                "test_acc_std": train_set_size_dict["test_acc_std"],
                "model_recovery_acc": train_set_size_dict["model_recovery_acc"],
                "max_acc_model": train_set_size_dict["max_acc_model"],
                "max_acc_model_acc": train_set_size_dict["max_acc_model_acc"],
                "min_acc_model": train_set_size_dict["min_acc_model"],
                "min_acc_model_acc": train_set_size_dict["min_acc_model_acc"]
            }])], ignore_index=True)
    df.to_csv(f"{save_path}/final_results_df.csv",index=False)
    
    
    
    
@hydra.main(version_base=None, config_path='../scripts_configurations',config_name="figure_5_analysis")
def main(cfg: DictConfig):
    models_names = cfg.models if not cfg.include_additional_models else cfg.models + cfg.new_models
    features_dirs =  os.path.join(cfg.features_dir,cfg.features_dim)
    reg_cons = cfg.reg_cons
    #define paths
    triplets_tensor_path = cfg.paths.triplets_tensor_path
    triplets_answers_positions_tensor_path = cfg.paths.triplets_answers_positions_tensor_path
    #load triplets tensors
    triplets_tensor = torch.load(triplets_tensor_path)
    triplets_answers_positions_tensor = torch.load(triplets_answers_positions_tensor_path)
    #load or create folds
    folds = _load_or_create_folds(
        triplets_tensor=triplets_tensor,
        triplets_answers_tensor=triplets_answers_positions_tensor,
        paths=cfg.paths,
        k_folds=cfg.k_folds,
        n_concepts=1854, #This is constant for the THINGS dataset
        random_seed=42 #this is constant for reproducibility
    )
    constraint = None
    regularization_type = None
    model_features_dict = {model_name:f"{features_dirs}/{model_name}.pt" for model_name in models_names}
    db_paths_dict = dict(cfg.db_paths_dict[cfg.features_dim])
    n_list = cfg.n_list
    
    # Validate checkpoints and get missing configurations
    missing_configs, loaded_results_dict = validate_checkpoint_and_get_missing_configs(cfg)
    
    save_path = cfg.save_path
    
    # Initialize the results dictionary properly
    if loaded_results_dict is None:
        # No previous results, start fresh
        final_res_mean_acc_dict_full: Dict[str, Dict[int, Dict[str, Union[float, str]]]] = {}
        print("No previous results found. Starting fresh computation.")
    else:
        # Use loaded results as base
        final_res_mean_acc_dict_full = loaded_results_dict
        print(f"Loaded previous results for {len(loaded_results_dict)} constraint type(s).")
    
    if not missing_configs:
        print("All results are complete! No computation needed.")
        return
    
    # Determine which constraints need to be processed
    print(f"Found {len(missing_configs)} configurations that need computation:")
    for constraints_name, n in missing_configs:
        print(f"  - {constraints_name} with n={n}")
    
    # Extract unique constraint types from missing configs, preserving the order from config
    # Filter constraint_types_to_analyze to only include those that have missing configs
    missing_constraint_names = set([constraint_name for constraint_name, n_val in missing_configs])
    constraints_to_process = [c for c in cfg.constraint_types_to_analyze if c in missing_constraint_names]
    
    # If some missing constraints are not in constraint_types_to_analyze, add them at the end
    for constraint_name in missing_constraint_names:
        if constraint_name not in cfg.constraint_types_to_analyze:
            print(f"Warning: '{constraint_name}' found in missing configs but not in constraint_types_to_analyze. Adding it.")
            constraints_to_process.append(constraint_name)
    
    print(f"Processing constraints in order: {constraints_to_process}")
    
    # Convert missing_configs to a set for quick lookup
    missing_configs_set = set(missing_configs)
        
    for constraint in tqdm(constraints_to_process, desc="Constraints", total=len(constraints_to_process), position=0):
        constraint = maybe_parse_tuple(constraint)
        regularization_type = None  # Reset for each constraint
        
        if isinstance(constraint, tuple):
            constraints_name = f"{constraint[0]}_{constraint[1]}"
        elif constraint is None:
            constraints_name = "full_W"
        elif constraint == "full_W_L1":
            constraint = None  # Set to None for full_W
            constraints_name = "full_W_L1"
            regularization_type = "L1"
        else:
            constraints_name = constraint
        
        # Initialize constraint dict if not already present
        if constraints_name not in final_res_mean_acc_dict_full:
            final_res_mean_acc_dict_full[constraints_name] = {}
        
        # Get only the n values that need computation for this constraint
        n_values_to_process = [n for n in n_list if (constraints_name, n) in missing_configs_set]
        
        if not n_values_to_process:
            print(f"No missing configurations for {constraints_name}, skipping...")
            continue
            
        print(f"Processing {len(n_values_to_process)} training set size(s) for {constraints_name}: {n_values_to_process}")
        
        for n in tqdm(n_values_to_process, desc="Train set size", total=len(n_values_to_process), position=1):
            model_recovery_analysis = DBResultsAnalysis(
                db_path=db_paths_dict[constraints_name],
                data_generating_models_list=models_names,
                reference_models_list=models_names,
                simulations_list=None,
            )
            model_recovery_acc = model_recovery_analysis.get_total_model_recovery_accuracy(n_triplets=n)       

            
            best_acc_models,mean_acc_models,folds_acc_models = evaluate_models_on_n_size_set(
                model_names=models_names,
                n=n,
                full_folds_sets=folds,
                model_features_dict=model_features_dict,
                reg_cons=reg_cons,
                constraints=constraint, 
                regularization = regularization_type
            )
            
            
            max_acc_model = max(best_acc_models.keys(), key=lambda k: best_acc_models[k][0])
            min_acc_model = min(best_acc_models.keys(), key=lambda k: best_acc_models[k][0])
    
            final_res_mean_acc_dict_full[constraints_name][n] = {
                                                                "test_acc": float(np.mean([val[0] for key,val in best_acc_models.items()])),
                                                                "test_acc_std": float(np.std([val[0] for key,val in best_acc_models.items()])),
                                                                "model_recovery_acc": float(model_recovery_acc),
                                                                "max_acc_model": str(max_acc_model),
                                                                "max_acc_model_acc": float(best_acc_models[max_acc_model][0]),
                                                                "min_acc_model": str(min_acc_model),
                                                                "min_acc_model_acc": float(best_acc_models[min_acc_model][0])
                                                                }
 
            save_results_dir = f"{save_path}/{constraints_name}/train_set_size_{n}"
            n_specific_results_dfs(
            models_names=models_names,
            reg_cons=reg_cons if constraint != "zero_shot" else [0],
            save_results_dir=save_results_dir,
            best_acc_models=best_acc_models,
            mean_acc_models=mean_acc_models,
            folds_acc_models=folds_acc_models,
            save_results=True,
            return_dfs_dict=False
            )
            
            
        save_final_results_df(final_res_mean_acc_dict_full=final_res_mean_acc_dict_full,save_path=save_path)


if __name__ == "__main__":
    main()

