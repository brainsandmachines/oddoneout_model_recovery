import os
import sys
from pathlib import Path
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(
    PARENT_DIR
)  # We want the script to work with relative paths as if script is one level up
sys.path.append(str(PARENT_DIR))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tools.db_analysis import DBResultsAnalysis 
import hydra
from omegaconf import DictConfig
import logging

# Configure logging for clean, readable output
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

"""
Table_S3_regression.py - Statistical Analysis of Model Properties and Performance

This script performs regression analysis to understand the relationship between
neural network model properties and their performance in cognitive model recovery tasks.
It analyzes how architectural features of models affect their ability to predict
human similarity judgments and creates the regression analysis reported in Table S3.

The script provides:
1. A RegressionAnalysis class extending DBResultsAnalysis for statistical modeling
2. Functions for bootstrap resampling to assess coefficient stability
3. Utilities for creating final results tables with confidence intervals and p-values

Key predictors analyzed include:
- Effective dimensionality scores (ED_pre_dg, ED_pre_ref, ED_post_dg, ED_post_ref)
  from effective_dimensinlity_score.py, measuring how many dimensions are meaningfully
  used in feature representations
- Model architecture properties (n_params, embed_dim)
- Embedding distances before and after transformations

Main workflow:
- Load model properties and simulation results from database
- Run OLS regression to identify significant predictors of model performance
- Perform bootstrap analysis to generate stable confidence intervals
- Format and save results for inclusion in the paper
"""

# Update pandas display options to avoid warnings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
from statsmodels.stats.multitest import multipletests

class RegressionAnalysis(DBResultsAnalysis):
    def __init__(
        self, 
        models_data_table_path: str,
        db_path: str,
        data_generating_models_list: list, 
        simulations_list: list, 
        reference_models_list: list = None,
        N_triplets:int = 4200000,
        categorical_columns: list = None,
        annotate_misidentification: bool = True
        ):
        """
        Initialize RegressionAnalysis with database and model properties for statistical analysis.
        
        This class extends DBResultsAnalysis to provide regression-based analysis of model
        properties and their relationship to model recovery performance. It loads model
        properties from a table and combines them with performance metrics from the database.
        
        Parameters:
            models_data_table_path (str): Path to CSV file containing model properties
            db_path (str): Path to the SQLite database containing experiment results
            data_generating_models_list (list): List of data-generating model names to analyze
            simulations_list (list): List of simulation indices to include in analysis
            reference_models_list (list, optional): List of reference model names. If None,
                                                   uses data_generating_models_list
            N_triplets (int, default=4200000): Number of triplets to filter for in the analysis
            categorical_columns (list, optional): List of column names in the model properties 
                                                that should be treated as categorical variables
            annotate_misidentification (bool, default=True): Whether to add a 'misidentification'
                                                           column to the results dataframe
        
        Notes:
            - On initialization, loads model properties and performance data, then combines them
            - Scales numeric features and handles categorical variables appropriately
            - Adds additional annotations like misidentification flags and accuracy difference
              metrics to facilitate analysis
            - The full_results_df attribute contains the complete dataset for analysis
        """
        super().__init__(db_path, data_generating_models_list, simulations_list, reference_models_list)
        self.models_properties_df = pd.read_csv(models_data_table_path)
        self.N_triplets = N_triplets
        self.full_results_df = self._build_full_results_df()
        if annotate_misidentification:
            self._annotate_misidentification()
        self._annotate_accuracy_difference()
        self.categorical_columns = categorical_columns
        self.models_properties_df = self._scale_data(self.models_properties_df)
    
    def change_full_results_df(self, full_results_df: pd.DataFrame):
        """
        Replace the current results DataFrame with a new one.
        
        Used primarily during bootstrap analysis to swap in resampled data.
        
        Parameters:
            full_results_df (pd.DataFrame): New DataFrame to use for analysis
        """
        self.full_results_df = full_results_df
        
    def _build_full_results_df(self):
        """
        Build a complete DataFrame of simulation results for all data-generating models.
        
        Returns:
            pd.DataFrame: Concatenated DataFrame with results for all models
        """
        res = []
        for model in self.data_generating_models_list:
            model_results = self.get_model_simulations_results(
                data_generating_model_name=model, 
                n_triplets=self.N_triplets,
                return_df=True
            )
            # Ensure we have a DataFrame
            if isinstance(model_results, dict):
                model_results = pd.DataFrame(model_results)
            res.append(model_results)
        
        if not res:
            raise ValueError(f"No results found for any models. Check database and N_triplets={self.N_triplets}")
        
        return pd.concat(res, ignore_index=True)
    
    def _scale_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale all non-categorical columns in the DataFrame using StandardScaler.
        
        This standardizes features to have zero mean and unit variance, which is 
        important for fair comparison of regression coefficients.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing model properties
            
        Returns:
            pd.DataFrame: DataFrame with numeric columns standardized
        """
        num_cols = []
        # Get numeric columns
        
        for col in df.columns:
            if col not in self.categorical_columns if self.categorical_columns else True:
                num_cols.append(col)
        df[num_cols] = StandardScaler().fit_transform(df[num_cols])
        return df

    def _annotate_misidentification(self) -> None:
        """
        Adds a column 'misidentification' to self.full_results_df:
        
        - NaN if data_generating_model == reference_model
        - 1   if self_rank > rank (indicates misidentification)
        - 0   otherwise
        
        where self_rank is the rank in the row where reference_model == data_generating_model
        (matched by simulation_idx and data_generating_model).
        
        This helps track cases where a better-fitting model was incorrectly identified.
        """
        df = self.full_results_df.copy()

        # 1) extract the self‐rank for each (simulation_idx, data_generating_model)
        self_ranks = (
            df.loc[
                df['data_generating_model'] == df['reference_model'],
                ['simulation_idx', 'data_generating_model', 'rank']
            ]
            .set_index(['simulation_idx', 'data_generating_model'])['rank']
        )

        # 2) map it back onto every row
        idx = pd.MultiIndex.from_frame(df[['simulation_idx', 'data_generating_model']])
        df['self_rank'] = idx.map(self_ranks)

        # 3) compute the flag
        is_same = df['data_generating_model'] == df['reference_model']
        df['misidentification'] = np.where(
            is_same,
            np.nan,
            np.where(df['self_rank'] > df['rank'], 1, 0)
        )

        # 4) drop helper column and assign back
        df.drop(columns=['self_rank'], inplace=True)
        self.full_results_df = df
        
    def _annotate_accuracy_difference(self):
        """
        Adds a column 'accuracy_difference' to self.full_results_df.
        
        This represents the difference between a model's accuracy when predicting its own
        data (self-prediction) versus predicting data from another model. It's a key metric
        for assessing model flexibility and specificity.
        
        Calculation: self_accuracy - test_accuracy
        - Positive values: Model is better at predicting its own data than other models' data
        - Negative values: Model is better at predicting other models' data than its own
        """
        df = self.full_results_df.copy()
        self_accuracy = (
            df.loc[
                df['data_generating_model'] == df['reference_model'],
                ['simulation_idx', 'data_generating_model','test_accuracy']
            ]
            .set_index(['simulation_idx', 'data_generating_model'])['test_accuracy']
        )
        idx = pd.MultiIndex.from_frame(df[['simulation_idx', 'data_generating_model']])
        df['self_accuracy'] = idx.map(self_accuracy)
        df['accuracy_difference'] = df['self_accuracy'] - df['test_accuracy']  
        df.drop(columns=['self_accuracy'], inplace=True)
        self.full_results_df = df
        
    def _dummify(self, df: pd.DataFrame, *, wide: bool) -> pd.DataFrame:
        """
        One-hot encode all categorical/string/boolean predictors and
        return a numeric-only DataFrame needed for regression analysis.

        • Encodes dg-side columns (and ref-side if wide=False)
        • Uses drop_first=True for k-1 coding to avoid multicollinearity
        • Forces dtype=float so statsmodels never receives bool/object types
        
        Parameters:
            df (pd.DataFrame): DataFrame containing mixed data types
            wide (bool): Whether we're in wide mode (changes suffix handling)
            
        Returns:
            pd.DataFrame: DataFrame with categorical variables dummified as numeric columns
        """
        work = df.copy()

        # 1) columns explicitly declared as categorical
        declared = []
        if self.categorical_columns:
            suffixes = ["_dg"] if wide else ["_dg", "_ref"]
            for base in self.categorical_columns:
                for s in suffixes:
                    col = f"{base}{s}"
                    if col in work.columns:
                        declared.append(col)

        # 2) any remaining object / category / bool columns
        auto = [
            c for c in work.columns
            if work[c].dtype.name in ("object", "category", "bool")
            and c not in ("simulation_idx",
                        "data_generating_model",
                        "reference_model")
        ]

        cat_cols = list(dict.fromkeys(declared + auto))          # keep order
        if cat_cols:
            work = pd.get_dummies(
                work,
                columns=cat_cols,
                drop_first=True,
                prefix_sep="_",
                dtype=float          # <<<  ALL dummy cols become float 0.0 / 1.0
            )

        # 3) convert any leftover bool columns that weren't dummy-created
        bool_cols = work.select_dtypes("bool").columns
        if bool_cols.any():
            work[bool_cols] = work[bool_cols].astype(float)

        return work
        
    def _expand_predictor_cols(self, df, predictors):
        """
        Replace each base predictor name by the concrete column(s) present
        in `df` (handles one-hot expansions transparently).
        
        This is needed because categorical variables expand into multiple dummy columns,
        and we need to include all of them in the regression.
        
        Parameters:
            df (pd.DataFrame): The dataframe containing columns
            predictors (list): Base predictor names to expand
            
        Returns:
            list: Expanded list of column names for regression
            
        Raises:
            KeyError: If a predictor cannot be found in the dataframe
        """
        cols = []
        for base in predictors:
            if base in df.columns:
                cols.append(base)
            else:
                pref = f"{base}_"
                matches = [c for c in df.columns if c.startswith(pref)]
                if not matches:
                    raise KeyError(f"Predictor '{base}' (or its dummies) not found")
                cols.extend(matches)
        return cols
    
    def assemble_dataframe(self,
                           predictors_list: list[str],
                           target_variable: str,
                           wide: bool = False
                          ) -> pd.DataFrame:
        """
        Build a DataFrame of features + targets according to two modes:
        
        - wide=False (Case 1): one row per (dg_model, ref_model, simulation_idx).
          Columns: [<predictor>_dg…, <predictor>_ref…, target_variable].
        
        - wide=True (Case 2): one row per (dg_model, simulation_idx).
          Columns: [<predictor>_dg…, <target_variable>_<ref_model1>, ..., <target_variable>_<ref_modelN>].
          
        This function handles merging model properties with simulation results and
        formatting the data appropriately for regression analysis.
        
        Parameters:
            predictors_list (list[str]): List of predictor variable names
            target_variable (str): Name of the target variable
            wide (bool, default=False): Whether to use wide format (multiple columns per target)
            
        Returns:
            pd.DataFrame: DataFrame formatted for regression analysis
        """
        # shorthand
        full = self.full_results_df
        props = self.models_properties_df.rename(columns={'model_name': 'model_name'})
        props= props.dropna()  # drop rows with any NaNs in model properties
        # 1) merge in dg predictors
        dg_props = props[['model_name'] + predictors_list].rename(
            columns={col: f"{col}_dg" for col in predictors_list}
        )
        df = full.merge(dg_props,
                        how='left',
                        left_on='data_generating_model',
                        right_on='model_name'
                       ).drop(columns='model_name')
        
        # 2) depending on mode, either merge ref predictors or pivot targets wide
        if not wide:
            # Case 1: merge in reference‐model predictors
            ref_props = props[['model_name'] + predictors_list].rename(
                columns={col: f"{col}_ref" for col in predictors_list}
            )
            df = df.merge(ref_props,
                          how='left',
                          left_on='reference_model',
                          right_on='model_name'
                         ).drop(columns='model_name')
        else:
            # Case 2: pivot the target variable wide across reference models
            pivot = (df
                     .pivot_table(
                         index=['simulation_idx', 'data_generating_model'],
                         columns='reference_model',
                         values=target_variable
                     )
                     .add_prefix(f"{target_variable}_")
                     .reset_index()
                    )
            # merge back dg predictors
            out = (pivot
                   .merge(dg_props,
                          how='left',
                          left_on='data_generating_model',
                          right_on='model_name')
                   .drop(columns='model_name')
                  )
            # order columns: simulation, dg model, predictors, then all targets
            cols = (
                ['simulation_idx', 'data_generating_model']
                + [f"{p}_dg" for p in predictors_list]
                + [c for c in out.columns
                   if c.startswith(f"{target_variable}_")]
            )
        
        if self.categorical_columns:
            suffixes = ["_dg"] if wide else ["_dg", "_ref"]
            to_dummy = [f"{c}{s}" for c in self.categorical_columns for s in suffixes
                        if f"{c}{s}" in (out if wide else df).columns]
            if wide:
                out = self._dummify(out, wide=wide)
            else:
                df  = self._dummify(df,  wide=wide)
      #  ------------------------------------------------------------------

        return out if wide else df[
            ['simulation_idx', 'data_generating_model']  # first columns
            + ([] if wide else ['reference_model'])
            + [c for c in (out if wide else df).columns
            if c.startswith(tuple(predictors_list))]   # all (dummy) predictors
            + ([c for c in (out if wide else df).columns
                if c.startswith(f"{target_variable}_")]   # wide targets
            if wide else [target_variable])
        ]

    def ols_paired(self,
                predictors_list: list[str],
                target_variable: str,
                print_summary:bool = True,
                return_vif:bool = False #if true, return the vif dataframe
                ):
        """
        Perform ordinary least squares regression on the paired/long data format.
        
        This function:
        * Prepares data using assemble_dataframe
        * Automatically handles dummy-coding for categorical predictors
        * Optionally calculates variance inflation factors (VIF) to check multicollinearity
        * Standardizes the target variable
        * Fits and returns the OLS model
        
        Parameters:
            predictors_list (list[str]): List of predictor variable names
            target_variable (str): Name of the target variable
            print_summary (bool, default=True): Whether to print regression summary
            return_vif (bool, default=False): Whether to return variance inflation factors
            
        Returns:
            model: Fitted OLS model object
            vif_df (optional): DataFrame with variance inflation factors if return_vif=True
        """
        df = self.assemble_dataframe(predictors_list, target_variable, wide=False)
        #if target variable is accuracy_differences , remove same data generating ,reference model pairs
        if target_variable == 'accuracy_difference':
            df = df[df['data_generating_model'] != df['reference_model']]
        
        # ------------------------------------------------------------------
        # 5) build design matrix -------------------------------------------
        X_cols = self._expand_predictor_cols(df, predictors_list)
        X = sm.add_constant(df[X_cols], prepend=True).astype(float)
        y = pd.DataFrame(df[target_variable].astype(float))
        #run variance inflation factor to check for multicollinearity
        if return_vif:
            vif_data = []
            for i, col in enumerate(X.columns):
                vif = variance_inflation_factor(X.values, i)
                vif_data.append((col, vif))
            vif_df = pd.DataFrame(vif_data, columns=['feature', 'VIF'])
        #scale all the non categorical columns and the target variable
        # ------------------------------------------------------------------
        # 6) fit OLS --------------------------------------------------------
        y[target_variable] = StandardScaler().fit_transform(y)
        res = sm.OLS(y, X, missing="raise").fit()
        if print_summary:
            print(res.summary())
        if return_vif:
            return res,vif_df
        else:
            return res

def bootstrap_ols(
                models_names: list[str],
                predictors_list: list[str],
                target_variable: str,
                regression_analysis_args:dict,
                n_boot:int = 1000, 
                ci:int = 95,
                output_dir: str = "Results/regression_analysis",
                filter_reference_models:bool = True,
                exclude_self_pairs:bool = True,
                save_bootstrap_files:bool = True):
    """
    Perform bootstrap analysis of OLS regression to assess coefficient stability.
    
    This function conducts bootstrap resampling on the model dataset to estimate the
    sampling distribution of regression coefficients. It samples models with replacement,
    runs OLS regression on each bootstrap sample, and computes summary statistics and
    confidence intervals for the coefficients.
    
    Parameters:
        models_names (list[str]): List of model names to bootstrap from
        predictors_list (list[str]): List of predictor variable names to include in the regression
        target_variable (str): Name of the target variable to predict
        regression_analysis_args (dict): Arguments for initializing the RegressionAnalysis class
        n_boot (int, default=1000): Number of bootstrap iterations to perform
        ci (int, default=95): Confidence interval percentage (e.g., 95 for 95% CI)
        output_dir (str, default="Results/regression_analysis"): Directory to save bootstrap results
        filter_reference_models (bool, default=True): If True, filter reference models to match
                                                    the sampled data-generating models
        exclude_self_pairs (bool, default=True): If True, exclude cases where reference model
                                               is the same as data-generating model
        
    Returns:
        dict: Dictionary containing:
             - "bootstrap_estimates": DataFrame of coefficient estimates from all bootstrap iterations
             - "bootstrap_pvalues": DataFrame of p-values from all bootstrap iterations
             - "confidence_intervals": DataFrame with lower and upper confidence bounds for coefficients
             
    Notes:
        - Uses random seed 42 for reproducibility
        - Creates weighted bootstrap samples based on the frequency of models in each resample
        - Handles edge cases like empty bootstrap dataframes
        - Saves detailed results to CSV files for further analysis
        - The bootstrap distribution can be used to assess coefficient stability and
          generate more robust confidence intervals than those from a single regression
    """
    np.random.seed(42)
    # Create output directory
    target_dir = os.path.join(output_dir, target_variable)

    
    # Create a base regression analysis object to get the original data
    base_reg = RegressionAnalysis(**regression_analysis_args)
    original_df = base_reg.full_results_df.copy()
    
    coefs = []
    pvalues = []
    for i in tqdm(range(n_boot), desc="Bootstrapping"):
        # Sample with replacement from the models list
        sampled_indices = np.random.choice(len(models_names), size=len(models_names), replace=True)
        sampled_models = [models_names[i] for i in sampled_indices]
        
        # Count occurrences of each model in the sample
        model_counts = {}
        for model in sampled_models:
            model_counts[model] = model_counts.get(model, 0) + 1
        
        # Create a bootstrap dataframe with appropriate weighting
        bootstrap_dfs = []
        
        # For each pair of data-generating and reference models in our bootstrap
        for dg_model, dg_count in model_counts.items():
            # Define which reference models to include
            if filter_reference_models:
                ref_models = model_counts.keys()  # Only use models from our bootstrap
            else:
                ref_models = models_names  # Use all models as references
                
            for ref_model in ref_models:
                if ref_model == dg_model and exclude_self_pairs: # exclude self-pairs
                    continue
                # If filtering references, get the count for this reference model
                ref_count = model_counts.get(ref_model, 1) if filter_reference_models else 1
                
                # Filter rows for this dg_model and ref_model pair
                pair_rows = original_df[(original_df['data_generating_model'] == dg_model) & 
                                        (original_df['reference_model'] == ref_model)].copy()

                # Replicate these rows according to data-generating and reference model counts
                # The weight is the product of dg_count and ref_count, representing the 
                # number of times this pair appears in a bootstrap sample
                weight = dg_count * (ref_count if filter_reference_models else 1)
                # If weight > 1, duplicate these rows weight-1 times
                if weight > 1:
                    # First copy already there with original simulation_idx
                    for _ in range(weight - 1):
                        dup_rows = pair_rows.copy()
                        # Create new simulation_idx to avoid duplicate keys
                        max_sim_idx = max(original_df['simulation_idx'].max(), 
                                        0 if not bootstrap_dfs else 
                                        max([df['simulation_idx'].max() for df in bootstrap_dfs]))
                        dup_rows['simulation_idx'] = max_sim_idx + 1
                        bootstrap_dfs.append(dup_rows)
                
                # Add the original rows
                bootstrap_dfs.append(pair_rows)
        
        # Combine all the bootstrap dataframes
        bootstrap_df = pd.concat(bootstrap_dfs, ignore_index=True) if bootstrap_dfs else pd.DataFrame()
        
        # Skip this iteration if we got an empty dataframe
        if len(bootstrap_df) == 0:
            print(f"Warning: Empty bootstrap dataframe in iteration {i}. Skipping.")
            continue
        
        # Create the bootstrap regression analysis object
        regression_analysis_args['annotate_misidentification'] = False
        bootstrap_reg = RegressionAnalysis(**regression_analysis_args)
        bootstrap_reg.change_full_results_df(bootstrap_df)
        
        try:
            # Run the regression
            model = bootstrap_reg.ols_paired(predictors_list, target_variable, print_summary=False, return_vif=False)
            coefs.append(model.params)
            pvalues.append(model.pvalues)
        except Exception as e:
            print(f"Warning: Regression failed in iteration {i}: {str(e)}. Skipping.")
            continue
        
        # Clean up to avoid memory issues
        del bootstrap_reg, model, bootstrap_df
        
    coefs_df = pd.DataFrame(coefs)
    pvalues_df = pd.DataFrame(pvalues)
    ci_lower = coefs_df.quantile((100 - ci) / 2 / 100)
    ci_upper = coefs_df.quantile(1 - (100 - ci) / 2 / 100)
    
    # Calculate summary statistics
    summary_stats = pd.DataFrame({
        'mean': coefs_df.mean(),
        'std': coefs_df.std(),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value_mean': pvalues_df.mean(),
        'p_value_std': pvalues_df.std()
    })
    
    # Save results
    os.makedirs(target_dir, exist_ok=True)
    
    if save_bootstrap_files:
        coefs_df.to_csv(os.path.join(target_dir, 'bootstrap_coefficients.csv'))
        pvalues_df.to_csv(os.path.join(target_dir, 'bootstrap_pvalues.csv'))
        summary_stats.to_csv(os.path.join(target_dir, 'bootstrap_summary.csv'))

    return {
        "bootstrap_estimates": coefs_df,
        "bootstrap_pvalues": pvalues_df,
        "confidence_intervals": pd.concat([ci_lower, ci_upper], axis=1, keys=['lower', 'upper'])
    }

def final_results_table(bootstrap_results, paired_model):
    """
    Create a final results table from bootstrap and OLS regression results.
    
    This function combines the point estimates from OLS regression with confidence
    intervals and p-values derived from bootstrap analysis. It also calculates
    p-values for left and right tails and applies multiple comparison correction.
    
    Parameters:
        bootstrap_results (dict): Results from bootstrap_ols function containing:
            - bootstrap_estimates: DataFrame of bootstrap coefficient estimates
            - confidence_intervals: DataFrame with lower and upper bounds
        paired_model: Fitted statsmodels OLS model object
            
    Returns:
        pd.DataFrame: Final results table with columns:
            - predictor: Name of the predictor variable
            - coefficient: Point estimate from OLS regression
            - bootstrap_p_value: Two-tailed p-value from bootstrap analysis
            - corrected_p_value: Bonferroni-corrected p-value
            - ci_lower: Lower confidence interval bound
            - ci_upper: Upper confidence interval bound
            - reject: Boolean indicating statistical significance after correction
            - bootstrap_coef_mean: Mean of bootstrap coefficient estimates
            - bootstrap_coef_std: Standard deviation of bootstrap estimates
            - bootstrap_p_value_left: Left-tailed p-value
            - bootstrap_p_value_right: Right-tailed p-value
    """
    # Create a final results table from the bootstrap and OLS results.  
    coefs_df = bootstrap_results['bootstrap_estimates']
    ci = bootstrap_results['confidence_intervals']
    paired_coefs = paired_model.params
    final_res = {
        "predictor": [],
        "coefficient": [],
        "bootstrap_p_value": [], 
        "corrected_p_value": None,
        "ci_lower": [],
        "ci_upper": [],
        "reject": None,             
        "bootstrap_coef_mean": [],
        "bootstrap_coef_std": [],
        "bootstrap_p_value_left": [],
        "bootstrap_p_value_right": [],
    }
    # Plot individual coefficient distributions
    for coef_name, paired_value in paired_coefs.items():
        # Plot coefficient distribution
        plt.figure(figsize=(10,  6))
        bootstrap_data = coefs_df[coef_name]
        # Calculate bootstrap-based p-values
        M = len(bootstrap_data)
        B_left = (bootstrap_data <= 0).sum()
        B_right = (bootstrap_data >= 0).sum()
        left_tail_p_value = (B_left + 1) / (M + 1)
        right_tail_p_value = (B_right + 1) / (M + 1)
        two_tailed_p_value = min(min(left_tail_p_value, right_tail_p_value) * 2, 1)
        safe_coef_name = coef_name.replace('/', '_').replace(' ', '_')
        # Fit normal distribution
        mu = bootstrap_data.mean()
        sigma = bootstrap_data.std()
        final_res["predictor"].append(safe_coef_name)
        final_res["coefficient"].append(paired_value)
        final_res["bootstrap_p_value"].append(two_tailed_p_value)
        final_res["ci_lower"].append(ci.loc[coef_name, 'lower'])
        final_res["ci_upper"].append(ci.loc[coef_name, 'upper'])
        final_res["bootstrap_p_value_left"].append(left_tail_p_value)
        final_res["bootstrap_p_value_right"].append(right_tail_p_value)
        final_res["bootstrap_coef_mean"].append(mu)
        final_res["bootstrap_coef_std"].append(sigma)
    rejected, pvals_corrected,_,_ = multipletests(final_res["bootstrap_p_value"],method="bonferroni")
    final_res["corrected_p_value"] = pvals_corrected
    final_res["reject"] = rejected
    final_res_df = pd.DataFrame(final_res)
    return final_res_df

@hydra.main(version_base=None, config_path="../scripts_configurations", config_name="Table_S3_regression")
def main(cfg: DictConfig):
    """
    Main execution function for the regression analysis that generates Table S3.
    
    This function performs a comprehensive statistical analysis to understand which
    neural network model properties predict performance in cognitive model recovery tasks.
    
    WORKFLOW:
    ---------
    STEP 1: Load Configuration
            - Extract model list, database path, and parameters from Hydra config
            - Validate that all specified models exist in both database and properties table
    
    STEP 2: Initialize Regression Analysis
            - Create RegressionAnalysis object by merging database results with model properties
            - Perform data preprocessing (scaling, dummy coding of categorical variables)
    
    STEP 3: Define Predictors and Target
            - Predictors: n_params, embed_dim, alignment_induced_shift, pre_gride, post_gride
            - Target: accuracy_difference (self-prediction vs. cross-prediction accuracy)
    
    STEP 4: Run OLS Regression
            - Fit ordinary least squares regression with standardized features
            - Calculate Variance Inflation Factors (VIF) to check multicollinearity
            - Print regression summary and VIF diagnostics
    
    STEP 5: Bootstrap Analysis
            - Perform 10,000 bootstrap resamples at the model level
            - Generate stable confidence intervals for regression coefficients
            - Compute bootstrap-based p-values with multiple comparison correction
    
    STEP 6: Create Final Results Table
            - Combine OLS estimates with bootstrap CIs and p-values
            - Apply Bonferroni correction for multiple testing
            - Save comprehensive results table to CSV (Table S3)
    
    CONFIGURATION:
    --------------
    All parameters loaded from: scripts_configurations/Table_S3_regression.yaml
    
    Key config parameters:
        - models_list: List of 20 model names to analyze
        - db_path: Path to SQLite database with simulation results
        - models_data_table_path: Path to CSV with model properties
        - N_simulations: Number of independent simulations per model (default: 30)
        - N_triplets: Training set size to analyze (default: 4200000)
    
    OUTPUTS:
    --------
    Console:
        1. OLS regression summary (coefficients, R², p-values)
        2. VIF table (multicollinearity diagnostics)
        3. Bootstrap progress bar (10,000 iterations)
    
    Files saved to: Results/regression_analysis/accuracy_difference/
        1. bootstrap_coefficients.csv - All bootstrap coefficient estimates
        2. bootstrap_pvalues.csv - All bootstrap p-values
        3. bootstrap_summary.csv - Summary statistics (mean, std, CI)
        4. regression_analysis_gride.csv - **MAIN TABLE S3 RESULTS**
    
    USAGE EXAMPLES:
    ---------------
    # Default configuration (all parameters from YAML)
    python Table_S3_regression.py
    
    # Override training set size
    python Table_S3_regression.py N_triplets=1638400
    
    # Use fewer simulations for faster testing
    python Table_S3_regression.py N_simulations=10
    
    # Custom model subset
    python Table_S3_regression.py models_list='["ResNet50", "Google_ViT_Large"]'
    
    DEPENDENCIES:
    -------------
    - tools.db_analysis.DBResultsAnalysis: For querying simulation database
    - statsmodels: OLS regression and statistical tests
    - sklearn: StandardScaler for feature normalization
    - pandas, numpy: Data manipulation
    - tqdm: Progress bars

    """
    
    # =========================================================================
    # STEP 1: Load Configuration from Hydra
    # =========================================================================
    logger.info("=" * 80)
    logger.info("TABLE S3 - REGRESSION ANALYSIS OF MODEL PROPERTIES")
    logger.info("=" * 80)
    
    # Load basic configuration
    db_path = cfg.db_path
    models_data_table_path = cfg.models_data_table_path
    N_simulations = cfg.N_simulations
    N_triplets = cfg.get('N_triplets', 4200000)  # Default to maximum if not specified
    predictors = list(cfg.predictors)  # Load predictors from YAML configuration
    
    # Bootstrap parameters from YAML
    n_boot = cfg.bootstrap.get('n_boot', 10000)
    bootstrap_ci = cfg.bootstrap.get('ci', 95)
    filter_reference_models = cfg.bootstrap.get('filter_reference_models', True)
    exclude_self_pairs = cfg.bootstrap.get('exclude_self_pairs', True)
    
    # Output configuration from YAML
    output_base_dir = cfg.output.get('base_dir', 'Results/regression_analysis')
    results_filename = cfg.output.get('results_filename', 'regression_analysis_gride.csv')
    save_bootstrap_files = cfg.output.get('save_bootstrap_files', True)
    
    # Auto-discover models from database or use manual list
    if cfg.get('models_list') is not None and cfg.models_list is not None:
        # Manual models list provided in YAML
        models_list = list(cfg.models_list)
        logger.info(f"\n📋 Using manual model list from configuration: {len(models_list)} models")
    else:
        # Auto-discover models from database
        import sqlite3
        logger.info(f"\n🔍 Auto-discovering models from database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("""
            SELECT DISTINCT data_generating_model 
            FROM jobs 
            WHERE status = 'done'
            ORDER BY data_generating_model
        """)
        models_list = [row[0] for row in cursor.fetchall()]
        conn.close()
        logger.info(f"   ✓ Found {len(models_list)} models with completed simulations")
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   • Number of models: {len(models_list)}")
    logger.info(f"   • Database: {db_path}")
    logger.info(f"   • Model properties: {models_data_table_path}")
    logger.info(f"   • Simulations per model: {N_simulations}")
    logger.info(f"   • Training triplets analyzed: {N_triplets:,}")
    logger.info(f"   • Bootstrap iterations: {n_boot:,}")
    logger.info(f"   • Output file: {output_base_dir}/{results_filename}")
    logger.info(f"\n📊 Models included in analysis:")
    for i, model in enumerate(models_list, 1):
        logger.info(f"   {i:2d}. {model}")
    
    # =========================================================================
    # STEP 2: Initialize Regression Analysis
    # =========================================================================
    logger.info(f"\n⚙️  Initializing regression analysis...")
    
    reg_args = {
        "models_data_table_path": models_data_table_path,
        "db_path": db_path,
        "data_generating_models_list": models_list,
        "simulations_list": list(range(N_simulations)),
        "reference_models_list": models_list,
        "N_triplets": N_triplets,
        "categorical_columns": ['model_name']
    }
    
    regression_analysis = RegressionAnalysis(**reg_args)
    logger.info(f"   ✓ Loaded {len(regression_analysis.full_results_df):,} simulation results")
    logger.info(f"   ✓ Merged with model properties from {models_data_table_path}")

    # =========================================================================
    # STEP 3: Define Predictors and Target Variable
    # =========================================================================
    logger.info(f"\n🎯 Regression specification:")
    
    # Predictor variables loaded from YAML configuration (cfg.predictors)
    # Can be overridden via command line:
    #   python Table_S3_regression.py predictors='["n_params", "embed_dim"]'
    
    target_variable = "accuracy_difference"  
    
    logger.info(f"   • Target variable: {target_variable}")
    logger.info(f"   • Predictor variables ({len(predictors)}):")
    for predictor in predictors:
        logger.info(f"      - {predictor}")

    # =========================================================================
    # STEP 4: Run OLS Regression with Multicollinearity Diagnostics
    # =========================================================================
    logger.info(f"\n🔬 Running OLS regression...")
    
    res_accuracy_difference_ols, vif_df = regression_analysis.ols_paired(
        predictors_list=predictors,
        target_variable=target_variable,
        return_vif=True
    )
    
    logger.info(f"\n📊 Variance Inflation Factors (VIF):")
    logger.info(f"   Note: VIF > 10 indicates problematic multicollinearity")
    logger.info("\n" + vif_df.to_string(index=False))

    # =========================================================================
    # STEP 5: Bootstrap Analysis for Robust Confidence Intervals
    # =========================================================================
    logger.info(f"\n🔄 Running bootstrap analysis...")
    logger.info(f"   • Bootstrap iterations: {n_boot:,}")
    logger.info(f"   • Confidence interval: {bootstrap_ci}%")
    logger.info(f"   • Resampling strategy: Model-level (with replacement)")
    logger.info(f"   • Reference model filtering: {filter_reference_models}")
    logger.info(f"   • Self-pair exclusion: {exclude_self_pairs}")
    
    res_accuracy_bootstrap = bootstrap_ols(
        models_names=models_list,
        predictors_list=predictors,
        target_variable=target_variable,
        n_boot=n_boot,
        ci=bootstrap_ci,
        output_dir=output_base_dir,
        filter_reference_models=filter_reference_models,
        exclude_self_pairs=exclude_self_pairs,
        save_bootstrap_files=save_bootstrap_files,
        regression_analysis_args=reg_args
    )
    
    logger.info(f"   ✓ Bootstrap analysis completed")

    # =========================================================================
    # STEP 6: Create and Save Final Results Table (Table S3)
    # =========================================================================
    logger.info(f"\n💾 Generating final results table...")
    
    final_res_df = final_results_table(res_accuracy_bootstrap, res_accuracy_difference_ols)
    
    # Save results using configured output settings
    os.makedirs(output_base_dir, exist_ok=True)
    output_path = os.path.join(output_base_dir, results_filename)
    final_res_df.to_csv(output_path, index=False)
    
    logger.info(f"   ✓ Saved final results to: {output_path}")
    
    if not save_bootstrap_files:
        logger.info(f"   ℹ  Bootstrap intermediate files not saved (save_bootstrap_files=False)")
    
    logger.info(f"\n📈 Final Results Summary (Table S3):")
    logger.info(f"   • Total predictors: {len(final_res_df)}")
    logger.info(f"   • Significant after Bonferroni correction: {final_res_df['reject'].sum()}")
    logger.info(f"\n   Significant predictors (corrected p < 0.05):")
    
    significant = final_res_df[final_res_df['reject'] == True]
    if len(significant) > 0:
        for _, row in significant.iterrows():
            direction = "+" if row['coefficient'] > 0 else "-"
            logger.info(f"      {direction} {row['predictor']}: coef={row['coefficient']:.4f}, "
                       f"p={row['corrected_p_value']:.4e}, "
                       f"CI=[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
    else:
        logger.info(f"      None (all p-values > 0.05 after correction)")
    
    logger.info(f"\n" + "=" * 80)
    logger.info(f"ANALYSIS COMPLETE - Results saved to {output_base_dir}/")
    logger.info(f"Main results file: {results_filename}")
    logger.info(f"=" * 80)





if __name__ == "__main__":
    main()