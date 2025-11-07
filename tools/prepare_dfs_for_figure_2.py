import torch
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import os
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from typing import Dict, List, Tuple, Optional



def process_predictions_to_success_and_failures(
    model_a_predictions:torch.Tensor,
    model_b_predictions:torch.Tensor,
    correct_predictions:torch.Tensor)->tuple[float,float,float,float]:
    """
    Process the predictions of two models and the correct predictions to get the number of wins for each model, the number of ties, and the total number of non-tie cases.
    """
    assert model_a_predictions.shape == model_b_predictions.shape == correct_predictions.shape
    #make sure that every one on the same device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_a_predictions = model_a_predictions.to(device)
    model_b_predictions = model_b_predictions.to(device)
    correct_predictions = correct_predictions.to(device)
    correct1 = (model_a_predictions == correct_predictions)
    correct2 = (model_b_predictions == correct_predictions)
    wins_model1 = correct1 & ~correct2
    wins_model2 = correct2 & ~correct1
    n_model1_wins = wins_model1.sum().item()
    n_model2_wins = wins_model2.sum().item()
    total_non_ties = n_model1_wins + n_model2_wins
    ties = (correct1 == correct2).sum().item()
    return n_model1_wins,total_non_ties,ties,n_model2_wins

def sign_test(
    n_model1_wins:float,
    total_non_ties:float,
    alternative:str="two-sided"
    ):
    """
    Compute the sign test p-value if there are any non-tie cases.
    """
    if total_non_ties > 0:
        p_value = binomtest(n_model1_wins, total_non_ties, 0.5, alternative=alternative).pvalue
    else:
        p_value = np.nan  # Not applicable if there are no informative instances
    return p_value

def multiple_sign_test(models_predictions_dict,correct_predictions,significance_level:float=0.05):
    results_list = []
    model_names = list(models_predictions_dict.keys())
    for model_A_name, model_B_name in combinations(model_names, 2):
        model_A_predictions = models_predictions_dict[model_A_name]
        model_B_predictions = models_predictions_dict[model_B_name]
        n_model1_wins,total_non_ties,ties,n_model2_wins = process_predictions_to_success_and_failures(model_A_predictions,model_B_predictions,correct_predictions)
        p_value = sign_test(n_model1_wins,total_non_ties)
        print(f"The p-value for the sign test between {model_A_name} and {model_B_name} is {p_value}")
        results_list.append({
            'model_a': model_A_name,
            'model_b': model_B_name,
            'n_a_wins': n_model1_wins,
            'n_b_wins': n_model2_wins,
            'total_non_ties': total_non_ties,
            'ties': ties,
            'p_value': p_value,
            'is_sig': np.nan,
            'p_value_corrected': np.nan,
            'effect_direction': np.nan
        })
    
    valid_results = [r for r in results_list if not np.isnan(r['p_value'])]
    p_values = np.array([r['p_value'] for r in valid_results])
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=significance_level, method='bonferroni')
    # Map the corrected p-values back to the valid_results
    for i, r in enumerate(valid_results):
        r['p_value_corrected'] = pvals_corrected[i]
        r['is_sig'] = bool(reject[i])  # Ensure boolean type
        r['effect_direction'] = 1 if r['n_a_wins'] > r['n_b_wins'] else -1
    
    return valid_results,results_list

def prepare_results_for_metro_plot(results_list:list[dict], formal_names_dict:dict)->pd.DataFrame:
    """
    Prepare the results for the metro plot.
    """
    # metroplot expects columns: level1, level2, effect_direction, is_sig
    # Here, we use model_a as level1 and model_b as level2.
    if not results_list:
        # Return empty DataFrame with correct columns if no results
        return pd.DataFrame(columns=['level1', 'level2', 'effect_direction', 'is_sig'])
    
    df = pd.DataFrame(results_list)[['model_a', 'model_b', 'effect_direction', 'is_sig']]
    
    # Convert model names to formal names
    df['level1'] = df['model_a'].apply(lambda x: go_formal(x, formal_names_dict))
    df['level2'] = df['model_b'].apply(lambda x: go_formal(x, formal_names_dict))
    
    # Drop the original columns
    df = df.drop(['model_a', 'model_b'], axis=1)
    
    # Ensure correct data types for metroplot
    df['is_sig'] = df['is_sig'].astype(bool)
    df['effect_direction'] = df['effect_direction'].astype(int)
    
    return df

def remove_under_score_from_model_name(model_name:str):
    return model_name.replace("_"," ")

def calculate_binomial_ci(successes, n, confidence=0.95):
    """
    Calculate binomial confidence intervals using scipy's binomtest
    """
    result = binomtest(successes, n)
    ci = result.proportion_ci(confidence_level=confidence)
    return (ci.low, ci.high)

def go_formal(name, formal_dict):
    """Convert model name to formal display name"""
    if name in ('Google_ViT_Large_224','Google_ViT_Large'):
        return 'ViT L/16'
    return formal_dict.get(name, name)

def load_correct_predictions(original_cwd: str) -> torch.Tensor:
    """Load correct predictions from the partitioned folds"""
    partitioned_indexes_paths = [
        os.path.join(original_cwd, f"Data/Things_data_preprocessed/partitions_folds/fold_{i}_test_partitioning_indexes.pt") 
        for i in range(1, 4)
    ]
    triplets_answers_positions_tensor_path = os.path.join(
        original_cwd,
        "Data/Things_data_preprocessed/Things_odd_one_out_answers_positions_train_set.pt"
    )
    
    # Load triplets answers tensor
    triplets_answers_positions_tensor = torch.load(triplets_answers_positions_tensor_path)
    
    # Load the three folds correct predictions
    triplets_correct_predictions = []
    for fold_indexes_path in partitioned_indexes_paths:
        partitioning_indexes = torch.load(fold_indexes_path)
        triplets_correct_predictions.append(triplets_answers_positions_tensor[partitioning_indexes])
    
    return torch.cat(triplets_correct_predictions, dim=0)

def load_model_predictions(
    model_name: str,
    predictions_base_dir: str,
    reg_func_type: str,
    reg_con: Optional[str]
) -> Optional[torch.Tensor]:
    """Load model predictions from the predictions folder"""
    try:
        if reg_func_type == "zero_shot":
            predictions_path = os.path.join(predictions_base_dir, model_name, reg_func_type)
        else:
            if reg_con is None:
                return None
            predictions_path = os.path.join(predictions_base_dir, model_name, reg_func_type, str(reg_con))
        
        if not os.path.exists(predictions_path):
            # Try alternative naming for reg_con (only for non-zero-shot)
            if reg_func_type != "zero_shot" and reg_con is not None:
                if str(reg_con) == "0":
                    predictions_path = os.path.join(predictions_base_dir, model_name, reg_func_type, "0.0")
                elif str(reg_con) == "0.0":
                    predictions_path = os.path.join(predictions_base_dir, model_name, reg_func_type, "0")
                else:
                    return None
            else:
                return None
        
        if not os.path.exists(predictions_path):
            return None
        
        # Load predictions from all folds
        predictions_files = os.listdir(predictions_path)
        predictions_files = sorted(predictions_files)
        
        predictions = []
        for i in range(3):  # 3 folds
            if i < len(predictions_files):
                file = predictions_files[i]
                model_preds = torch.load(os.path.join(predictions_path, file))
                predictions.append(model_preds)
            else:
                return None
        
        # Concatenate the predictions
        return torch.cat(predictions, dim=0)
    
    except Exception as e:
        print(f"Error loading predictions for {model_name}: {e}")
        return None

def create_accuracy_dataframe_with_ci(
    csv_path: str,
    models_order: List[str],
    formal_names_dict: Dict[str, str],
    confidence: float = 0.95
) -> pd.DataFrame:
    """Create accuracy dataframe with confidence intervals"""
    df = pd.read_csv(csv_path)
    
    # Ensure we have the models in the specified order
    df = df[df['model_name'].isin(models_order)]
    df = df.set_index('model_name').reindex(models_order).reset_index()
    
    # Calculate confidence intervals assuming 3-fold CV
    n_folds = 3
    results = []
    
    for _, row in df.iterrows():
        model_name = row['model_name']
        test_acc = row['test_acc']
        
        # Estimate number of test samples per fold (this is approximate)
        # For THINGS dataset, we have about 16740 triplets, divided into 3 folds
        n_samples_per_fold = 16740 // 3
        total_samples = n_samples_per_fold * n_folds
        
        # Calculate confidence interval
        successes = int(test_acc * total_samples)
        ci_low, ci_high = calculate_binomial_ci(successes, total_samples, confidence)
        
        results.append({
            'model_name': model_name,
            'test_acc': test_acc,
            'test_acc_low': test_acc - ci_low,
            'test_acc_high': ci_high - test_acc
        })
    
    return pd.DataFrame(results)

def prepare_figure_2_data(
    results_base_dir: str,
    constraint_types: List[str],
    regularization_types: List[str],
    models_order: List[str],
    formal_names_dict: Dict[str, str],
    original_cwd: str,
    feature_type: str = "full",
    output_base_name: str = "THINGS_accuracy",
    confidence_level: float = 0.95
) -> Tuple[Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]], List[str], List[str]]:
    """
    Prepare all data needed for Figure 2
    
    Returns:
        Tuple containing:
        - Dict mapping constraint_type to (accuracy_df, metro_df) tuples
        - List of successfully processed constraint types
        - List of failed constraint types
        metro_df is None if predictions are not available
    """
    results = {}
    successful_types = []
    failed_types = []
    
    # Load correct predictions once
    try:
        correct_predictions = load_correct_predictions(original_cwd)
        print(f"Loaded correct predictions: {correct_predictions.shape}")
    except Exception as e:
        print(f"Warning: Could not load correct predictions: {e}")
        correct_predictions = None
    
    for constraint_type in constraint_types:
        print(f"\nProcessing constraint type: {constraint_type}")
        
        # Load accuracy dataframe - handle zero_shot special case
        best_accuracy_df = None
        
        # Special handling for zero_shot constraint type
        if constraint_type == "zero_shot":
            best_reg_type = "zero_shot"  # Zero-shot has its own "regularization type"
            
            # Zero-shot uses a different naming convention
            csv_filename = f"zero_shot_results_{feature_type}.csv"
            csv_path = os.path.join(results_base_dir, constraint_type, csv_filename)
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    mean_acc = df['test_acc'].mean()
                    best_accuracy_df = create_accuracy_dataframe_with_ci(
                        csv_path, models_order, formal_names_dict, confidence_level
                    )
                    print(f"  Found zero-shot results (mean acc: {mean_acc:.4f})")
                except Exception as e:
                    print(f"  Error reading {csv_path}: {e}")
        else:
            # Regular constraint types REQUIRE a regularization type
            if not regularization_types:
                print(f"  Error: No regularization types provided for non-zero-shot constraint '{constraint_type}'")
                failed_types.append(constraint_type)
                continue
            
            best_reg_type: str = regularization_types[0]  # Type hint tells Pylance this is str, not None
            
            # Regular constraint types use the standard naming
            csv_filename = f"best_test_acc_{best_reg_type}_{feature_type}_{constraint_type}.csv"
            csv_path = os.path.join(results_base_dir, constraint_type, csv_filename)
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    mean_acc = df['test_acc'].mean()
                    best_accuracy_df = create_accuracy_dataframe_with_ci(
                        csv_path, models_order, formal_names_dict, confidence_level
                    )
                    print(f"  Found {best_reg_type} results (mean acc: {mean_acc:.4f})")
                except Exception as e:
                    print(f"  Error reading {csv_path}: {e}")
        
        if best_accuracy_df is None:
            print(f"  No valid CSV file found for {constraint_type}")
            failed_types.append(constraint_type)
            continue
        
        # Try to create metro plot data if predictions are available
        metro_df = None
        predictions_dir = os.path.join(results_base_dir, constraint_type, "predictions")
        
        if os.path.exists(predictions_dir) and correct_predictions is not None:
            print(f"  Attempting to load predictions for statistical analysis...")
            
            try:
                # Load best regularization constants - handle zero_shot special case
                models_predictions_dict = {}
                
                if constraint_type == "zero_shot":
                    # Zero-shot doesn't have reg_con, just load predictions directly
                    for model_name in models_order:
                        predictions = load_model_predictions(
                            model_name, predictions_dir, "zero_shot", None
                        )
                        if predictions is not None:
                            models_predictions_dict[model_name] = predictions
                            print(f"    Loaded predictions for {model_name}")
                        else:
                            print(f"    Could not load predictions for {model_name}")
                else:
                    # Regular constraint types use reg_con from CSV
                    csv_path = os.path.join(results_base_dir, constraint_type, 
                                          f"best_test_acc_{best_reg_type}_{feature_type}_{constraint_type}.csv")
                    best_reg_df = pd.read_csv(csv_path)
                    
                    # Create mapping of model to best reg_con
                    best_reg_cons = dict(zip(best_reg_df['model_name'], best_reg_df['reg_con']))
                    
                    # Load predictions for each model
                    for model_name in models_order:
                        if model_name in best_reg_cons:
                            reg_con = best_reg_cons[model_name]
                            predictions = load_model_predictions(
                                model_name, predictions_dir, best_reg_type, reg_con
                            )
                            if predictions is not None:
                                models_predictions_dict[model_name] = predictions
                                print(f"    Loaded predictions for {model_name}")
                            else:
                                print(f"    Could not load predictions for {model_name}")
                
                # Perform statistical analysis if we have enough predictions
                if len(models_predictions_dict) >= 2:
                    print(f"  Performing statistical analysis with {len(models_predictions_dict)} models...")
                    valid_results, _ = multiple_sign_test(
                        models_predictions_dict, correct_predictions
                    )
                    metro_df = prepare_results_for_metro_plot(valid_results, formal_names_dict)
                    print(f"  Created metro plot data with {len(metro_df)} comparisons")
                else:
                    print(f"  Not enough models with predictions ({len(models_predictions_dict)}) for statistical analysis")
                    
            except Exception as e:
                print(f"  Error during statistical analysis: {e}")
                metro_df = None
        else:
            print(f"  Predictions directory not found or correct predictions unavailable")
        
        results[constraint_type] = (best_accuracy_df, metro_df)
        successful_types.append(constraint_type)
        print(f"  Completed {constraint_type}: accuracy_df={len(best_accuracy_df)}, metro_df={'available' if metro_df is not None else 'unavailable'}")
    
    return results, successful_types, failed_types

def save_prepared_data(
    prepared_data: Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]],
    output_dir: str,
    feature_type: str = "full"
):
    """Save prepared data to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for constraint_type, (accuracy_df, metro_df) in prepared_data.items():
        # Save accuracy data
        accuracy_path = os.path.join(output_dir, f"{constraint_type}.csv")
        accuracy_df.to_csv(accuracy_path, index=False)
        print(f"Saved accuracy data: {accuracy_path}")
        
        # Save metro data if available
        if metro_df is not None:
            metro_path = os.path.join(output_dir, f"metro_{constraint_type}.csv")
            metro_df.to_csv(metro_path, index=False)
            print(f"Saved metro data: {metro_path}")
        else:
            print(f"No metro data to save for {constraint_type}")

def check_existing_results(output_dir: str, constraint_types: List[str]) -> Dict[str, Dict[str, bool]]:
    """Check which results already exist"""
    existing = {}
    for constraint_type in constraint_types:
        accuracy_path = os.path.join(output_dir, f"{constraint_type}.csv")
        metro_path = os.path.join(output_dir, f"metro_{constraint_type}.csv")
        
        existing[constraint_type] = {
            'accuracy': os.path.exists(accuracy_path),
            'metro': os.path.exists(metro_path)
        }
    
    return existing

