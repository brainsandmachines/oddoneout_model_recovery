"""
Database Analysis and Visualization Module

This module provides comprehensive analysis and visualization tools for model recovery experiments.
It queries SQLite databases containing simulation results and generates publication-ready figures.

Key classes:
- DBResultsAnalysis: Base class for querying and analyzing simulation results
- ModelRecoveryVisualizer: Extended class with publication-quality visualization methods

Main use case:
- Used by all figure generation scripts (create_Figures_1_and_3.py, create_Figure_5_tradeoff.py, etc.)
- Queries: Results/Simulations_experiments/<experiment>/<experiment>.db
- Outputs: Publication figures to Plots/ directory

Analysis capabilities:
- Confusion matrices (data-gen model × reference model accuracy)
- Recovery curves (accuracy vs. training set size)
- Precision/Recall/F1/MRR metrics
- Ranking analysis (when correct vs. incorrect)
"""

import os
import sys
# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Compute the parent directory
parent_dir = os.path.dirname(current_dir)

# Add parent directory to sys.path if not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require GUI
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union,Dict,Optional,Tuple
import seaborn as sns
import warnings
import matplotlib.colors as mcolors
from scipy.stats import binomtest
#add the parent folder to path


import hydra
from omegaconf import DictConfig
import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from pdf2image import convert_from_path
import scipy

def calculate_binomial_ci(k, n, conf=0.95):
    """
    Calculate the confidence interval for a binomial proportion.
    
    This function computes the confidence interval for a binomial proportion using the
    binomial test from scipy.stats. It's useful for estimating the uncertainty in
    accuracy measurements or other proportion-based metrics.
    
    Parameters:
        k (int): Number of successes observed
        n (int): Total number of trials
        conf (float): Confidence level (default: 0.95 for 95% confidence interval)
        
    Returns:
        Tuple[float, float]: Lower and upper bounds of the confidence interval
        
    Notes:
        - The calculation uses the exact binomial method, which is appropriate for
          small and large sample sizes
        - The returned interval is a two-sided interval with the specified confidence level
    """
    res = binomtest(int(k), int(n))
    ci = res.proportion_ci(confidence_level=conf)
    return ci.low, ci.high

def replace_vit_large(obj):
    """
    Replace 'Google_ViT_Large_224' with 'Google_ViT_Large' in DataFrame or Series.
    
    This utility function standardizes model names by replacing 'Google_ViT_Large_224' with
    'Google_ViT_Large' in various data structures. It handles both pandas DataFrame and Series
    objects, applying the replacement to string values only.
    
    Parameters:
        obj: The object to process. Can be a pandas DataFrame, Series, or any other object.
            If it's not a DataFrame or Series, the function returns the original object.
            
    Returns:
        The processed object with standardized model names. If the input was not a DataFrame
        or Series, the original object is returned unchanged.
        
    Notes:
        - For DataFrames, all string values in any column are processed
        - For Series, only string values are processed
        - This function is important for ensuring consistent model naming in analyses
        - The function is specifically used to handle Google ViT model name variations
    """
    if isinstance(obj, pd.DataFrame):
        return obj.applymap(lambda x: x.replace("Google_ViT_Large_224", "Google_ViT_Large") if isinstance(x, str) else x)
    elif isinstance(obj, pd.Series):
        return obj.map(lambda x: x.replace("Google_ViT_Large_224", "Google_ViT_Large") if isinstance(x, str) else x)
    return obj

# Add this near the top of your script
logging.getLogger('fontTools').setLevel(logging.WARNING)

# Set Arial font for all plots
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42  # Ensures text remains editable in PDF
plt.rcParams['ps.fonttype'] = 42   # Ensures text remains editable in PostScript

warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

##############################
# Optimized Database Analysis
##############################
def corrected_and_normalized_ratios_auto(fig_width, fig_height, row_ratios, col_ratios):
    """
    Calculate normalized row and column ratios for GridSpec to ensure square cells when appropriate.
    
    This function takes the dimensions of a figure and initial row and column ratios, and produces
    corrected ratios that maintain proper proportions while forcing cells to be square when their
    row and column ratios are equal. This is particularly useful for creating aesthetically pleasing
    grid-based visualizations such as confusion matrices.
    
    Parameters:
        fig_width (float): Total width of the figure in display units
        fig_height (float): Total height of the figure in display units
        row_ratios (list of float): Initial relative sizes for each row
        col_ratios (list of float): Initial relative sizes for each column
    
    Returns:
        tuple: A tuple (normalized_row_ratios, normalized_col_ratios) where:
            - normalized_row_ratios is a list of values that sum to fig_height
            - normalized_col_ratios is a list of values that sum to fig_width
            
    Methodology:
        1. Compute "ideal" row and column sizes based on the given ratios
        2. For every cell (i,j), if row_ratios[i] equals col_ratios[j], adjust both dimensions 
           to the smaller of the two sizes to force square cells
        3. Normalize the adjusted sizes to maintain the overall figure dimensions
        
    Note:
        This function is particularly useful for creating GridSpec layouts where certain cells
        should be square (e.g., confusion matrices, correlation plots) while maintaining the
        overall figure dimensions and relative proportions of other rows and columns.
    """
    import math
    
    nrows = len(row_ratios)
    ncols = len(col_ratios)
    
    # Compute ideal sizes based on the given ratios.
    total_row = sum(row_ratios)
    total_col = sum(col_ratios)
    ideal_rows = [fig_height * r / total_row for r in row_ratios]
    ideal_cols = [fig_width * c / total_col for c in col_ratios]
    
    # Start with the ideal sizes.
    corrected_rows = ideal_rows.copy()
    corrected_cols = ideal_cols.copy()
    
    # For every cell (i,j), if the row and column ratios are equal, force that cell to be square.
    tolerance = 1e-9
    for i in range(nrows):
        for j in range(ncols):
            if math.isclose(row_ratios[i], col_ratios[j], rel_tol=tolerance):
                new_dim = min(ideal_rows[i], ideal_cols[j])
                corrected_rows[i] = min(corrected_rows[i], new_dim)
                corrected_cols[j] = min(corrected_cols[j], new_dim)
    
    # Normalize the corrected sizes so that the sums equal fig_height and fig_width.
    sum_rows = sum(corrected_rows)
    sum_cols = sum(corrected_cols)
    normalized_rows = [h * fig_height / sum_rows for h in corrected_rows]
    normalized_cols = [w * fig_width / sum_cols for w in corrected_cols]
    
    return (normalized_rows, normalized_cols)







class DBResultsAnalysis:
    """
    Analyzer for model recovery experiment results stored in a SQLite database.
    
    This class provides core functionality for analyzing model recovery experiment results,
    including loading database data, calculating confusion matrices, computing metrics like
    precision, recall, F1, accuracy, and Mean Reciprocal Rank (MRR), and performing 
    statistical analyses. It forms the foundation for more specialized visualization and
    analysis classes.
    
    The class is designed to work with a specific database schema where:
    - 'jobs' table contains information about simulation runs, including data_generating_model
    - 'reference_model_results' table contains test accuracies for each job and reference model
    
    This design allows for efficient analysis of model recovery performance across different
    simulation conditions, triplet counts, and model pairings.
    """
    def __init__(self, db_path: str,
                data_generating_models_list: list,
                simulations_list: Optional[list] = None,
                reference_models_list: list = None,
               ):
        """
        Initialize the DBResultsAnalysis with database and analysis parameters.
        
        Parameters:
            db_path (str): Path to the SQLite database file containing experiment results.
            data_generating_models_list (list): List of model names that were used to generate data.
                                                Analysis will be restricted to these models.
            simulations_list (Optional[list]): List of simulation indices to analyze. If None,
                                              all complete simulations from the database are used.
            reference_models_list (list): List of reference model names to analyze. If None,
                                         uses data_generating_models_list.
                                         
        Notes:
            - Upon initialization, the class loads data from the database, filters for models
              of interest, and performs basic data validation.
            - Any incomplete simulations are excluded from analysis.
            - Google ViT model names are standardized for consistency.
        """
        self.db_path = db_path
        self.data_generating_models_list = data_generating_models_list
        self.reference_models_list = (reference_models_list 
                                    if reference_models_list is not None 
                                    else data_generating_models_list)
        # Load and cache the database tables once]
        self.jobs_df, self.ref_results_df = self._load_data()
        
        # Get simulation span as a dictionary: {n_triplets: [sim_indices]}
        self.simulations_dict = self.get_full_simulation_span() if simulations_list is None else {}
        self.full_n_triplets_span = self.get_full_n_triplets_span()
        self.simulations_list = simulations_list if simulations_list is not None else []
        self.add_ranks_to_results()
   
    def get_full_n_triplets_span(self):
        """
        Get the complete list of training triplet counts used in the experiments.
        
        This method extracts the unique values of n_train_triplets from the jobs_df
        to identify all triplet counts that were used across the experiments. This
        is useful for analyses that need to examine performance as a function of
        training set size.
        
        Returns:
            numpy.ndarray: Array of unique training triplet counts from the database.
        """
        return self.jobs_df['n_train_triplets'].unique()
    
    def get_full_simulation_span(self):
        """
        Get the indices of simulations that are complete for each n_triplets value.
        
        This method identifies simulation indices that have complete coverage for each
        n_train_triplets value separately (all data-generating models have completed jobs
        WITH results in the reference_model_results table). This allows different
        n_triplets values to have different sets of complete simulations.
        
        Returns:
            dict: Dictionary mapping n_triplets values to sorted lists of simulation indices
                  that are complete for that specific n_triplets value.
                  Format: {n_triplets: [sim_idx1, sim_idx2, ...]}
            
        Side effects:
            Sets self.undone_sim_per_n_triplets with a dictionary of incomplete simulations
            per n_triplets value.
        """
        # Get all unique n_train_triplets values
        all_n_triplets = sorted(self.jobs_df['n_train_triplets'].unique())
        
        # Get all unique simulation indices
        all_sim_indices = self.jobs_df['simulation_idx'].unique()
        
        # OPTIMIZATION: Pre-compute which (sim_idx, model, n_triplets) combinations have BOTH jobs AND results
        # Merge jobs with ref_results to see which combinations have results
        jobs_with_results = pd.merge(
            self.jobs_df[self.jobs_df['status'] == 'done'],
            self.ref_results_df[['job_id']].drop_duplicates(),
            left_on='id',
            right_on='job_id',
            how='inner'  # Only keep jobs that have results
        )
        
        # Create a set of valid (sim_idx, model, n_triplets) tuples for fast lookup
        valid_combinations = set(
            zip(jobs_with_results['simulation_idx'],
                jobs_with_results['data_generating_model'],
                jobs_with_results['n_train_triplets'])
        )
        
        # For EACH n_triplets value, find which simulations are complete
        simulations_per_n_triplets = {}
        undone_sim_per_n_triplets = {}
        
        for n_triplets in all_n_triplets:
            eligible_simulations = []
            undone_sim = []
            
            for sim_idx in all_sim_indices:
                # Check if ALL models are complete for this (sim_idx, n_triplets) combination
                models_complete = sum(
                    1 for model in self.data_generating_models_list
                    if (sim_idx, model, n_triplets) in valid_combinations
                )
                
                if models_complete == len(self.data_generating_models_list):
                    eligible_simulations.append(sim_idx)
                else:
                    undone_sim.append(sim_idx)
            
            eligible_simulations.sort()
            simulations_per_n_triplets[n_triplets] = eligible_simulations
            undone_sim_per_n_triplets[n_triplets] = undone_sim
            
            print(f"n_triplets={n_triplets}: {len(eligible_simulations)} complete simulation(s) "
                  f"across all {len(self.data_generating_models_list)} models")

        # Check if any n_triplets has zero complete simulations
        empty_n_triplets = [n for n, sims in simulations_per_n_triplets.items() if len(sims) == 0]
        if empty_n_triplets:
            print(f"\nWARNING: The following n_triplets values have NO complete simulations: {empty_n_triplets}")
            print(f"Models expected: {len(self.data_generating_models_list)} models")
            
            # For the first empty n_triplets, show which simulations are most complete
            from collections import defaultdict
            n_triplets = empty_n_triplets[0]
            sim_completion_count = defaultdict(int)
            for combo in valid_combinations:
                if combo[2] == n_triplets:  # Filter by n_triplets
                    sim_completion_count[combo[0]] += 1
            
            if sim_completion_count:
                sorted_sims = sorted(sim_completion_count.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"\nFor n_triplets={n_triplets}, top 5 most complete simulations:")
                for sim_idx, count in sorted_sims:
                    pct = (count / len(self.data_generating_models_list)) * 100
                    print(f"  Simulation {sim_idx}: {count}/{len(self.data_generating_models_list)} models ({pct:.1f}%)")
        
        self.undone_sim_per_n_triplets = undone_sim_per_n_triplets
        
        return simulations_per_n_triplets
    
    def _load_data(self,print_info: bool = False):
        """
        Load experiment data from the SQLite database.
        
        This internal method connects to the SQLite database and loads job information 
        and reference model results. It performs data validation, handles model name 
        standardization (especially for Google ViT models), and filters data to include
        only the specified models and completed jobs.
        
        Parameters:
            print_info (bool): Whether to print detailed information about the loaded data,
                              including row counts and null value checks. Defaults to False.
                              
        Returns:
            tuple: A tuple containing:
                - jobs_df (pd.DataFrame): DataFrame with job information (filtered to include
                  only data-generating models in data_generating_models_list).
                - ref_results_df (pd.DataFrame): DataFrame with reference model results (filtered
                  to include only reference models in reference_models_list).
                  
        Raises:
            Exception: If there's an error connecting to or reading from the database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First check the total counts
                jobs_count = pd.read_sql_query("SELECT COUNT(*) as count FROM jobs", conn).iloc[0]['count']
                ref_results_count = pd.read_sql_query("SELECT COUNT(*) as count FROM reference_model_results", conn).iloc[0]['count']
                
                if print_info:
                    print(f"Total rows in jobs table: {jobs_count}")
                    print(f"Total rows in reference_model_results table: {ref_results_count}")
                
                # Load the actual data
                jobs_df = pd.read_sql_query(
                    "SELECT id, simulation_idx, data_generating_model, n_train_triplets, status FROM jobs", conn)
                ref_results_df = pd.read_sql_query(
                    "SELECT job_id, reference_model, test_accuracy FROM reference_model_results", conn)
                
                if print_info:
                    print(f"Loaded rows in jobs_df: {len(jobs_df)}")
                    print(f"Loaded rows in ref_results_df: {len(ref_results_df)}")
                
                # Check for any null values
                if jobs_df.isnull().values.any():
                    print("Warning: Null values found in jobs_df")
                    print(jobs_df[jobs_df.isnull().any(axis=1)])
                
                if ref_results_df.isnull().values.any():
                    print("Warning: Null values found in ref_results_df")
                    print(ref_results_df[ref_results_df.isnull().any(axis=1)])
                # Replace "Google_ViT_Large_224" with "Google_ViT_Large" in the loaded DataFrames
                jobs_df = replace_vit_large(jobs_df)
                ref_results_df = replace_vit_large(ref_results_df)
                # Keep only rows whose data_generating_model appears in the requested list
                jobs_df = jobs_df[jobs_df['data_generating_model'].isin(self.data_generating_models_list)]
                # Keep only rows whose reference_model appears in the requested list
                ref_results_df = ref_results_df[ref_results_df['reference_model'].isin(self.reference_models_list)]
                return jobs_df, ref_results_df
        except Exception as e:
            print(f"Error loading data from database: {str(e)}")
            raise

    def get_model_simulations_results(self, data_generating_model_name: str, n_triplets: int,return_df: bool = False) -> dict:
        """
        Retrieve simulation results for a specific data-generating model and triplet count.
        
        This method filters the database results to extract test accuracies for all reference
        models across all simulation runs that used the specified data-generating model and
        number of training triplets. It's a core method used by many analysis functions.
        
        Parameters:
            data_generating_model_name (str): Name of the data-generating model to analyze.
            n_triplets (int): Number of training triplets used in the simulations.
            return_df (bool): Whether to return a merged dataframe instead of a dictionary.
                             Defaults to False.
                             
        Returns:
            dict or pd.DataFrame: 
                If return_df is False (default):
                    A nested dictionary of the form {simulation_idx: {reference_model: test_accuracy}}
                    mapping each simulation index to another dictionary containing reference model
                    accuracies for that simulation.
                If return_df is True:
                    The merged DataFrame containing all data for the specified model and triplet count.
                    
        Notes:
            - The method checks for and removes any duplicate entries.
            - Asserts that the correct number of simulations are found.
            - Only reference models in self.reference_models_list are included.
        """
        # Get the n_triplets-specific simulation list
        simulations_to_use = self.simulations_dict.get(n_triplets, self.simulations_list)
        
        # Filter jobs for this model, simulation indices, and triplet count
        filtered_jobs = self.jobs_df[
            (self.jobs_df['data_generating_model'] == data_generating_model_name)
            & (self.jobs_df['simulation_idx'].isin(simulations_to_use))
            & (self.jobs_df['n_train_triplets'] == n_triplets)
        ]
        
        if filtered_jobs.empty:
            return {}
        
        # Merge with reference results
        merged = pd.merge(filtered_jobs, self.ref_results_df, left_on='id', right_on='job_id')
        #Check for duplicates
        duplicates = merged[merged.duplicated(subset=['simulation_idx', 'reference_model'])]
        if not duplicates.empty:
            #print which simulations are duplicated
            # print(f"There are {len(duplicates)} duplicates for {data_generating_model_name} and {n_triplets} triplets")
            # print(duplicates['simulation_idx'].unique())
            merged = merged.drop_duplicates(subset=['simulation_idx', 'reference_model'])
        
        # Verify we got the expected simulations for this specific n_triplets
        # NOTE: We filter by simulations_to_use in filtered_jobs, so merged should only contain those simulations
        found_simulations = len(merged) / len(self.ref_results_df.reference_model.unique())
        if found_simulations != len(simulations_to_use):
            # This is expected if not all simulations were run for this specific n_triplets
            # The simulations_to_use contains only simulations that are complete for THIS n_triplets
            # So for this n_triplets, we should have exactly those simulations
            raise ValueError(
                f"Expected {len(simulations_to_use)} simulations for {data_generating_model_name} "
                f"and {n_triplets} triplets (from simulations_dict: {simulations_to_use}), "
                f"but found {found_simulations}. This suggests the database has incomplete data "
                f"for some simulations that were marked as complete in get_full_simulation_span()."
            )
        
        if merged.empty:
            return {}
        
        # 1) Only keep rows where 'reference_model' is in the current reference_models_list
        valid_refs = set(self.reference_models_list)
        merged = merged[merged['reference_model'].isin(valid_refs)]
        if return_df:
            return merged
        # 2) Group results by simulation index
        results_dict = {}
        for sim_idx, group in merged.groupby('simulation_idx'):
            # Build a {reference_model: test_accuracy} dictionary
            ref_acc_dict = dict(zip(group['reference_model'], group['test_accuracy']))
            results_dict[sim_idx] = ref_acc_dict
        else:
            return results_dict

    def confusion_matrix_row(self, data_generating_model_name: str, n_triplets: int) -> pd.DataFrame:
        """
        Create a single row for a model recovery confusion matrix.
        
        This method analyzes the given data-generating model with the specified number of
        training triplets and determines which reference model performed best in each
        simulation. These counts form a single row of the confusion matrix, where the
        row represents the data-generating model and each column shows the count of times
        each reference model was selected as best.
        
        Parameters:
            data_generating_model_name (str): Name of the data-generating model to analyze.
            n_triplets (int): Number of training triplets used in the simulations.
            
        Returns:
            pd.DataFrame: A single-row DataFrame where:
                - The index is the data_generating_model_name
                - The columns are reference model names
                - Each value is the count of simulations where that reference model
                  had the highest accuracy for the given data-generating model
                  
        Notes:
            - This is a fundamental component for building the full confusion matrix.
            - The ideal case for perfect model recovery would be all counts in the diagonal.
        """
        results_dict = self.get_model_simulations_results(data_generating_model_name, n_triplets)
        # Get the n_triplets-specific simulation list
        simulations_to_use = self.simulations_dict.get(n_triplets, self.simulations_list)
        # Initialize counts for each reference model
        model_counts = {model: 0 for model in self.reference_models_list}
        # For each simulation, determine the best performing reference model
        for sim_idx in simulations_to_use:
            if sim_idx in results_dict:
                sim_results = results_dict[sim_idx]
                best_model = max(sim_results, key=sim_results.get, default=None)
                model_counts[best_model] += 1
        # Return as a one-row DataFrame
        return pd.DataFrame([model_counts], index=[data_generating_model_name],
                            columns=self.reference_models_list)
        
    def add_ranks_to_results(self) -> pd.DataFrame:
        """
        Add ranking information to the reference model results.
        
        This method adds a 'rank' column to self.ref_results_df by ranking reference models 
        within each job based on their test accuracy. The ranking uses the 'average' method
        to handle ties appropriately. Ranks start at 1 (highest accuracy) and increase.
        
        For each job (a single data-generating model, simulation, and triplet count), all
        reference models are ranked by their test accuracy. This ranking is essential for
        metrics like Mean Reciprocal Rank (MRR) and analyzing how highly each model ranks
        itself when it is the true data-generating model.
        
        Returns:
            pd.DataFrame: Updates self.ref_results_df in-place by adding a 'rank' column,
                        but also returns None to maintain interface consistency.
                        
        Notes:
            - The method uses descending order for ranks (highest accuracy = rank 1).
            - The 'average' method assigns the average rank to tied values.
        """
        self.ref_results_df['rank'] = self.ref_results_df.groupby('job_id')['test_accuracy'].rank(method='average', ascending=False)


    def calculate_model_MRR(self, n_triplets: int) -> dict:
        """
        Calculate Mean Reciprocal Rank (MRR) for each model when it's the data generating model.
        
        Mean Reciprocal Rank measures how well a model recognizes itself. For each model
        as the data-generating model, MRR is calculated as the mean of (1/rank) across all
        simulations, where 'rank' is the position of the model in the sorted accuracy list
        when comparing to itself as a reference model.
        
        Parameters:
            n_triplets (int): Number of training triplets to analyze.
            
        Returns:
            dict: Dictionary mapping model names to their MRR scores. Higher MRR scores
                 (closer to 1.0) indicate better self-identification. An MRR of 1.0 means
                 the model always ranked itself first (complete self-recognition).
                 
        Notes:
            - MRR calculations are restricted to simulations in self.simulations_list.
            - If a model has no relevant data, its MRR will be 0.0.
            - The method automatically calls add_ranks_to_results to ensure ranks are available.
            - MRR is a key metric for evaluating how well models can identify their own generated data.
            - This method provides a focused implementation for a specific triplet count, while
              get_mrr_for_model and get_all_models_mean_reciprocal_ranks offer more flexible interfaces.
        """
        # Add ranks to results if not already done
        self.add_ranks_to_results()
        
        mrr_dict = {}
        
        # For each model as a data generating model
        for model_name in self.data_generating_models_list:
            # Get the n_triplets-specific simulation list
            simulations_to_use = self.simulations_dict.get(n_triplets, self.simulations_list)
            # Get all jobs where this model was the data generating model
            relevant_jobs = self.jobs_df[
                (self.jobs_df['data_generating_model'] == model_name) &
                (self.jobs_df['n_train_triplets'] == n_triplets) &
                (self.jobs_df['simulation_idx'].isin(simulations_to_use))
            ]
            
            if relevant_jobs.empty:
                mrr_dict[model_name] = 0.0
                continue
            
            # Get all reference model results for these jobs
            job_ranks = self.ref_results_df[
                (self.ref_results_df['job_id'].isin(relevant_jobs['id'])) &
                (self.ref_results_df['reference_model'] == model_name)
            ]
            
            if job_ranks.empty:
                mrr_dict[model_name] = 0.0
                continue
            
            # Calculate MRR: mean of 1/rank for each instance
            reciprocal_ranks = 1.0 / job_ranks['rank']
            mrr = reciprocal_ranks.mean()
            mrr_dict[model_name] = mrr
        
        return mrr_dict

    def calculate_models_mean_rank_correct_and_incorrect(self, n_triplets: int, return_df: bool = True) -> Tuple[pd.DataFrame, dict, dict]:
        """
        Calculate mean ranks for models in both correct and incorrect data-generating conditions.
        
        This method calculates two key metrics for each model:
        1. "Correct Rank": The average rank of a model when it is the data-generating model
                          (lower = better; 1 = always ranked first)
        2. "False Rank": The average rank of a model when it is NOT the data-generating model
                        (higher = better; avoiding false recognition)
                        
        These metrics provide insight into both how well models recognize themselves and how
        strongly they incorrectly identify with other models' data.
        
        Parameters:
            n_triplets (int): Number of training triplets to analyze.
            return_df (bool): Whether to return a DataFrame with the results in addition to
                             the dictionaries. Defaults to True.
                             
        Returns:
            Tuple containing various combinations depending on return_df:
            
            If return_df is True:
                (pd.DataFrame, dict, dict, dict, dict): A tuple containing:
                - DataFrame with model names and their correct/false ranks and standard errors
                - Dictionary mapping models to their correct ranks
                - Dictionary mapping models to their false ranks
                - Dictionary mapping models to their correct rank standard errors
                - Dictionary mapping models to their false rank standard errors
                
            If return_df is False:
                (dict, dict, dict, dict): A tuple containing:
                - Dictionary mapping models to their correct ranks
                - Dictionary mapping models to their false ranks
                - Dictionary mapping models to their correct rank standard errors
                - Dictionary mapping models to their false rank standard errors
                
        Notes:
            - The method calculates standard errors (SEM) for all metrics.
            - Ranks are computed in descending order of accuracy (rank 1 = highest accuracy).
            - For "Correct Rank", lower values are better (model recognizes its own data).
            - For "False Rank", higher values are better (model doesn't claim other models' data).
            - This analysis is particularly useful for understanding both the sensitivity
              (self-recognition) and specificity (avoiding false positives) of each model.
            - Models with low correct ranks and high false ranks have the best overall
              recovery characteristics.
        """
        # Add ranks to results if not already done
        self.add_ranks_to_results()
        # Get the n_triplets-specific simulation list
        simulations_to_use = self.simulations_dict.get(n_triplets, self.simulations_list)
        false_ranks_dict_std = {}
        correct_ranks_dict_std = {}
        correct_ranks_dict = {}
        false_ranks_dict = {}
        # For each model as a data generating model
        for ref_model in self.reference_models_list:
            false_ranks_list  = []
            for data_generating_model in self.data_generating_models_list:
                # Get all jobs where this model was the data generating model
                relevant_jobs = self.jobs_df[
                    (self.jobs_df['data_generating_model'] == data_generating_model) &
                    (self.jobs_df['n_train_triplets'] == n_triplets) &
                    (self.jobs_df['simulation_idx'].isin(simulations_to_use))
                ]
                assert len(relevant_jobs) == len(simulations_to_use), f"There are {len(relevant_jobs)} simulations for {data_generating_model} and {n_triplets} triplets, but {len(simulations_to_use)} simulations in total need to be analyzed"
                # Get all reference model results for these jobs
                job_ranks = self.ref_results_df[
                    (self.ref_results_df['job_id'].isin(relevant_jobs['id'])) &
                    (self.ref_results_df['reference_model'] == ref_model)
                ]

                if ref_model == data_generating_model:
                    # Calculate MRR: mean of 1/rank for each instance
                    correct_ranks_dict[data_generating_model] = np.mean(job_ranks['rank'])
                    #correct_ranks_dict_std[data_generating_model] = (np.std(job_ranks['rank'],ddof=1)/np.sqrt(len(self.simulations_list))).item()
                    ci  = scipy.stats.bootstrap((job_ranks['rank'],), np.mean, confidence_level=0.99, n_resamples=10000, method='BCa').confidence_interval               
                    if np.isnan(ci.low) or np.isnan(ci.high):
                        correct_ranks_dict_std[data_generating_model] = 0.0001
                    else:
                        correct_ranks_dict_std[data_generating_model] = ((ci.high - ci.low)/2).item()
                else: 
                    false_ranks_list.append(job_ranks['rank'].mean().item())
            false_ranks_dict[ref_model] = np.mean(false_ranks_list).item()
            false_ranks_dict_std[ref_model] = (np.std(false_ranks_list,ddof=1)/np.sqrt(len(simulations_to_use))).item()
            
        
        # Create DataFrame with both correct and false ranks
        df = pd.DataFrame({
            'Model': self.reference_models_list,
            'Correct Rank': [correct_ranks_dict[model] for model in self.reference_models_list],
            "Correct Rank Std": [correct_ranks_dict_std[model] for model in self.reference_models_list],
            'False Rank': [false_ranks_dict[model] for model in self.reference_models_list],
            "False Rank Std": [false_ranks_dict_std[model] for model in self.reference_models_list]
        })
        if return_df:   
            return df, correct_ranks_dict, false_ranks_dict, correct_ranks_dict_std, false_ranks_dict_std
        else:
            return correct_ranks_dict, false_ranks_dict, correct_ranks_dict_std, false_ranks_dict_std
        
        
    def compute_precision_recall_at_k(self, k: int, n_triplets: int) -> dict:
        """
        Compute binary precision@k and recall@k for each reference model.
        
        This method calculates standard information retrieval metrics (precision and recall)
        in the context of model recovery at a specific 'k' value:
        
        For each reference model and simulation:
        - Predicted Positive: The reference model appears in the top-k ranked models.
        - Actual Positive: The simulation's data-generating model is this reference model.
        
        Parameters:
            k (int): Number of top-ranked models to consider (e.g., k=1 means only the best model,
                    k=3 means the top three ranked models, etc.).
            n_triplets (int): Number of training triplets to analyze.
            
        Returns:
            dict: Dictionary mapping each reference model name to a nested dictionary with keys:
                - 'precision': Fraction of simulations where the model appeared in the top-k 
                              that were actually generated by that model.
                - 'recall': Fraction of simulations generated by the model where it appeared
                           in the top-k results.
                           
        Notes:
            - Precision@k measures the model's specificity: among all cases where the model
              was in the top-k predictions, how many were actually generated by that model.
            - Recall@k measures the model's sensitivity: among all cases where data was actually
              generated by the model, how many times did it appear in the top-k predictions.
            - Ties in ranking are broken deterministically by alphabetical ordering of model names.
            - Higher k values generally increase recall at the expense of precision.
            - This metric is particularly useful for evaluating model recovery in scenarios
              where selecting multiple candidate models is acceptable.
            - The calculation uses a vectorized approach for efficiency.
            - For k=1, this metric corresponds to traditional confusion matrix analysis.
        """
        import numpy as np
        import pandas as pd
        
        # Get the n_triplets-specific simulation list
        simulations_to_use = self.simulations_dict.get(n_triplets, self.simulations_list)
        
        # --- Step 1: Filter and Merge ---
        # Apply the same filters as in get_model_simulations_results:
        filtered_jobs = self.jobs_df[
            (self.jobs_df['n_train_triplets'] == n_triplets) &
            (self.jobs_df['simulation_idx'].isin(simulations_to_use)) &
            (self.jobs_df['data_generating_model'].isin(self.data_generating_models_list))
        ]
        
        # Merge with reference results.
        merged = pd.merge(filtered_jobs, self.ref_results_df, left_on='id', right_on='job_id')
        
        # Only keep rows where 'reference_model' is in self.reference_models_list.
        valid_refs = set(self.reference_models_list)
        merged = merged[merged['reference_model'].isin(valid_refs)]
        
        # --- Step 2: Pivot ---
        # Use pivot_table with 'first' as aggregation, matching get_model_simulations_results.
        pivot = merged.pivot_table(
            index=['simulation_idx', 'data_generating_model'],
            columns='reference_model',
            values='test_accuracy',
            aggfunc='first'
        )  
        
        # Ensure the columns (reference models) are sorted alphabetically for tie-breaking.
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        
        # --- Step 3: Determine Top-K for Each Simulation ---
        top_k_dict = {}  # Key: (simulation_idx, true_model), Value: list of top-k reference models.
        for idx, row in pivot.iterrows():
            # Use mergesort (a stable sort) to break ties according to the column order.
            sorted_row = row.sort_values(ascending=False, kind='mergesort')
            top_k = list(sorted_row.index[:k])
            top_k_dict[idx] = top_k
        
        # --- Step 4: Aggregate TP, FP, FN Counts ---
        counts = {ref: {"TP": 0, "FP": 0, "FN": 0} for ref in self.reference_models_list}
        
        # Iterate over each simulation (indexed by (sim_idx, true_model)).
        for (sim_idx, true_model), top_k in top_k_dict.items():
            for ref in self.reference_models_list:
                predicted_positive = (ref in top_k)
                actual_positive = (ref == true_model)
                if predicted_positive and actual_positive:
                    counts[ref]["TP"] += 1
                elif predicted_positive and not actual_positive:
                    counts[ref]["FP"] += 1
                elif not predicted_positive and actual_positive:
                    counts[ref]["FN"] += 1
        
        # --- Step 5: Compute Metrics ---
        metrics = {}
        for ref in self.reference_models_list:
            TP = counts[ref]["TP"]
            FP = counts[ref]["FP"]
            FN = counts[ref]["FN"]
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            metrics[ref] = {"precision": precision, "recall": recall}
        
        return metrics

    def get_k_mean_precision_recall(self,n_triplets: int,save_path: str = None,return_full: bool = False) -> Union[list,Tuple[list,dict]]:
        """
        Compute mean precision and recall across all k values for each reference model.
        
        This method calculates precision and recall at each possible k value (from 1 to the
        number of reference models) for each model, then averages these metrics across
        all k values. This provides a comprehensive measure of model recovery performance 
        that accounts for different ranking thresholds.
        
        Parameters:
            n_triplets (int): Number of training triplets to analyze.
            save_path (str, optional): Path to save the results as a CSV file. If None,
                                      results are not saved to disk. Defaults to None.
            return_full (bool): Whether to return the full results for each k value or just
                               the averaged results. Defaults to False.
                               
        Returns:
            If return_full is False (default):
                dict: Dictionary mapping each model name to a nested dictionary with keys:
                    - 'precision': Mean precision across all k values
                    - 'recall': Mean recall across all k values
                    
            If return_full is True:
                tuple(dict, dict): A tuple containing:
                    - Dictionary of averaged metrics as described above
                    - Dictionary mapping each k value to the results from compute_precision_recall_at_k
                    
        Notes:
            - This approach evaluates model recovery performance across the entire spectrum
              of possible rank thresholds, from strict (k=1) to lenient (k=all models).
            - Results can optionally be saved to a CSV file for further analysis.
        """
        k_list = range(1, len(self.reference_models_list) + 1) # All possible values for k 
        full_results = {}
        for k in k_list:
            precision_recall = self.compute_precision_recall_at_k(k, n_triplets)
            full_results[k] = precision_recall
        df_results = []
        final_results = {}
        # Average over k's and get the mean for each model
        for model in self.reference_models_list:
            mean_precision = np.mean([full_results[k][model]["precision"] for k in k_list])
            mean_recall = np.mean([full_results[k][model]["recall"] for k in k_list])
            df_results.append({
                        "model": model,
                        "mean_precision": mean_precision,
                        "mean_recall": mean_recall
                        })
            final_results[model] = {
                                    "precision": mean_precision,
                                    "recall": mean_recall}
        if save_path is not None:
            # Prepare results for saving as a CSV
            results_df = pd.DataFrame(df_results)
            results_df.to_csv(save_path)
        
        return final_results if not return_full else (final_results,full_results)
            

    def confusion_matrix(self, n_triplets: int) -> pd.DataFrame:
        """
        Build the full confusion matrix for all data-generating models.
        
        This method constructs a complete confusion matrix showing model recovery performance
        for all data-generating models at the specified number of training triplets. Each cell
        in the matrix contains the count of simulations where a particular reference model
        (column) was identified as the best match for a data-generating model (row).
        
        Parameters:
            n_triplets (int): Number of training triplets to analyze.
            
        Returns:
            pd.DataFrame: A DataFrame where:
                - Rows represent data-generating models
                - Columns represent reference models
                - Each value is the count of simulations where that reference model was
                  selected as the best match for the data-generating model
                  
        Notes:
            - The diagonal elements represent correct model recovery (self-identification).
            - In perfect recovery, all counts would be along the diagonal.
            - Each row sums to the number of simulations.
        """      
        
        cm_df = pd.DataFrame(columns=self.reference_models_list)
        for model in self.data_generating_models_list:
            row = self.confusion_matrix_row(model, n_triplets)
            cm_df = pd.concat([cm_df, row], axis=0)
        return cm_df

    def get_model_TP_FP_TN_FN(self, data_generating_model_name: str, n_triplets: int) -> dict:
        """
        Calculate binary classification metrics for a specified model.
        
        This method computes true positives (TP), false positives (FP), true negatives (TN),
        and false negatives (FN) for a specific model in the model recovery context:
        
        - TP: Number of simulations where the specified model correctly recovered itself
        - FP: Number of simulations where other models were misidentified as this model
        - FN: Number of simulations where this model was misidentified as other models
        - TN: Number of simulations where other models correctly identified other models
        
        Parameters:
            data_generating_model_name (str): Name of the model to analyze.
            n_triplets (int): Number of training triplets to analyze.
            
        Returns:
            dict: Dictionary with keys 'TP', 'FP', 'TN', 'FN' containing the respective counts.
            
        Notes:
            - These metrics form the basis for precision, recall, F1, and accuracy calculations.
            - Calculations are based on the confusion matrix from the model recovery results.
        """
        cm = self.confusion_matrix(n_triplets)
        TP = cm.loc[data_generating_model_name, data_generating_model_name]
        FN = cm.loc[data_generating_model_name].sum() - TP
        FP = cm[data_generating_model_name].sum() - TP
        TN = cm.values.sum() - (TP + FP + FN)
        return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

    def get_model_precision_recall_f1_accuracy(self, data_generating_model_name: str, n_triplets: int) -> dict:
        """
        Compute standard classification metrics for a specific model.
        
        This method calculates precision, recall, F1 score, and accuracy for a given model
        in the context of model recovery experiments. These metrics provide different 
        perspectives on how well the model can be identified:
        
        - Precision: Proportion of times the model was correctly identified out of all
                    times it was selected as the best match (TP/(TP+FP))
        - Recall: Proportion of times the model was correctly identified out of all
                 times it was the true data-generating model (TP/(TP+FN))
        - F1: Harmonic mean of precision and recall (2*precision*recall/(precision+recall))
        - Accuracy: Proportion of all correct identifications (TP+TN)/(TP+TN+FP+FN)
        
        Parameters:
            data_generating_model_name (str): Name of the model to analyze.
            n_triplets (int): Number of training triplets to analyze.
            
        Returns:
            dict: Dictionary with keys 'precision', 'recall', 'accuracy', 'f1' containing
                 the respective metric values.
                 
        Notes:
            - All metrics range from 0 to 1, with higher values indicating better performance.
            - Calculations are based on the confusion matrix from the model recovery results.
            - Edge cases (e.g., division by zero) are handled appropriately.
        """
        stats = self.get_model_TP_FP_TN_FN(data_generating_model_name, n_triplets)
        total = self.confusion_matrix(n_triplets).values.sum()
        precision = stats["TP"] / (stats["TP"] + stats["FP"]) if (stats["TP"] + stats["FP"]) > 0 else 0
        recall = stats["TP"] / (stats["TP"] + stats["FN"]) 
        accuracy = (stats["TP"] + stats["TN"]) / total if total > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        return {"precision": precision, "recall": recall, "accuracy": accuracy, "f1": f1}

    def get_total_model_recovery_accuracy(self, n_triplets: int) -> float:
        """
        Compute the overall model recovery accuracy across all models.
        
        This method calculates the average accuracy of model recovery by summing all
        diagonal elements of the confusion matrix (correct identifications) and dividing
        by the total number of simulations across all models. It provides a single metric
        that summarizes model recovery performance at a given triplet count.
        
        Parameters:
            n_triplets (int): Number of training triplets to analyze.
            
        Returns:
            float: The overall model recovery accuracy, ranging from 0.0 to 1.0.
                  1.0 indicates perfect recovery for all models, while lower values
                  indicate worse recovery performance.
                  
        Notes:
            - This metric weights all models equally in the final accuracy calculation.
            - The value is equal to the sum of the confusion matrix diagonal divided
              by the sum of all confusion matrix elements.
        """
        cm = self.confusion_matrix(n_triplets)
        total_TP = np.diag(cm).sum()
        return total_TP / cm.values.sum() if cm.values.sum() > 0 else 0

    def get_all_models_precision_recall_f1_accuracy(self, n_triplets: int) -> dict:
        """
        Compute performance metrics for all data-generating models.
        
        This method calculates precision, recall, F1 score, and accuracy for each model
        in data_generating_models_list at the specified triplet count. It provides a
        comprehensive view of model recovery performance across the entire model set.
        
        Parameters:
            n_triplets (int): Number of training triplets to analyze.
            
        Returns:
            dict: A nested dictionary where:
                - The outer keys are model names
                - The inner keys are 'precision', 'recall', 'accuracy', 'f1'
                - The values are the corresponding metric values
                
        Notes:
            - Progress is displayed with a tqdm progress bar showing model-by-model completion.
            - This method calls get_model_precision_recall_f1_accuracy for each model.
            - The computation may be time-consuming with many models since it requires
              calculating the confusion matrix for each model.
        """
        results = {}
        for model in tqdm(self.data_generating_models_list,
                          desc=f"Computing metrics for {n_triplets} triplets",
                          unit="model"):
            results[model] = self.get_model_precision_recall_f1_accuracy(model, n_triplets)
        return results
        
    def spearman_correlation_wrong_rank(self, n_triplets: int, ranked_compare_df_data_path: str) -> float:
        """
        Calculate Spearman correlation between model rankings in different contexts.
        
        This method computes the correlation between:
        1. The ranking of models when they are NOT the data-generating model ("False Rank")
        2. The ranking of models from an external source (typically dataset performance)
           
        A strong correlation would suggest that models that perform poorly in model
        recovery (higher false ranks) also perform poorly on the external measure.
        
        Parameters:
            n_triplets (int): Number of training triplets to analyze.
            ranked_compare_df_data_path (str): Path to a CSV file containing model rankings
                                              from an external source. The file should have
                                              columns 'model_name' and 'rank'.
                                              
        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: The Spearman correlation matrix between false ranks and external ranks
                - pd.DataFrame: The merged DataFrame containing both rank measures
                
        Notes:
            - The method handles the special case of Google ViT models by standardizing their names.
            - The correlation indicates whether model performance generalizes across tasks.
            - A positive correlation suggests consistent model behavior across contexts.
        """
        df_compare = pd.read_csv(ranked_compare_df_data_path,index_col=0)
        # Create the DataFrame containing correct and incorrect ranks
        df_wrong_correct,_,_ = self.calculate_models_mean_rank_correct_and_incorrect(n_triplets,return_df=True)
        # Align model names with those used in df_compare
        df_wrong_correct["Model"][df_wrong_correct["Model"]=='Google_ViT_Large_224'] = 'Google_ViT_Large'
        # Rename the column in df_compare_rank to match the join key
        df_compare.rename(columns={'model_name':'Model'},inplace=True)
        
        df_wrong_correct_rank = df_wrong_correct[["Model","False Rank"]]
        df_compare_rank = df_compare[["Model","rank"]]
        # Merge the two DataFrames on the model name
        df_merged = pd.merge(df_wrong_correct_rank, df_compare_rank, on='Model', how='inner')
        # Calculate the Spearman correlation
        spearman_corr = df_merged[["False Rank","rank"]].corr(method='spearman')
        return spearman_corr,df_merged
    
    def get_model_recovery_binomial_error(self, n_triplets: int, CI: bool = False) -> float:
        """
        Compute the binomial error or confidence interval for model recovery accuracy.
        
        This method calculates either:
        1. The standard error (if CI=False) of the model recovery accuracy using
           the binomial proportion formula: SE = sqrt(p * (1-p) / n)
        2. The confidence interval (if CI=True) for the model recovery accuracy
        
        The calculation treats model recovery as a binomial process (success/failure),
        which is appropriate for analyzing classification accuracy.
        
        Parameters:
            n_triplets (int): Number of training triplets used in the model.
            CI (bool): Whether to return confidence interval bounds (True) or standard
                      error (False). Defaults to False.
            
        Returns:
            If CI=False:
                float: The standard error of the model recovery accuracy.
            If CI=True:
                tuple: The lower and upper bounds of the 95% confidence interval.
                
        Raises:
            ValueError: If the number of trials or accuracy value is invalid.
            
        Notes:
            - The number of trials (n) is calculated as the number of simulations times
              the number of data-generating models.
            - The confidence interval is calculated using the binomial test from scipy.stats.
            - This provides a measure of statistical uncertainty in the recovery accuracy.
        """
        # Get the model recovery accuracy for this number of triplets
        p = self.get_total_model_recovery_accuracy(n_triplets)
    
        # Get the n_triplets-specific simulation list
        simulations_to_use = self.simulations_dict.get(n_triplets, self.simulations_list)
        # Get the total number of trials (number of simulations * number of models)
        n = len(simulations_to_use) * len(self.data_generating_models_list)
        k = p * n
        # Calculate binomial confidence interval
        ci = calculate_binomial_ci(k, n)
        # Calculate standard error using binomial proportion formula
        # SE = sqrt(p * (1-p) / n)
        if n > 0 and 0 <= p <= 1:
            standard_error = np.sqrt((p * (1 - p)) / n)
            return ci if CI else standard_error
        else:
            raise ValueError(f"Invalid number of trials or accuracy: n={n}, p={p}")

    def get_auc_of_model_recovery(self, model_name, n_triplets=None):
        """
        Calculates the Area Under the Curve (AUC) for a specific model's recovery performance.
        
        This method computes the AUC score for binary classification of a particular model
        versus all other models, providing a single metric of model discriminability that
        is robust to class imbalance.
        
        Parameters:
            model_name (str): Name of the model to calculate AUC for.
            n_triplets (int, optional): Number of triplets used for analysis. If None,
                                      uses all available triplet counts and returns a dict
                                      mapping each count to its corresponding AUC.
        
        Returns:
            Union[float, Dict[int, float]]: If n_triplets is specified, returns AUC score (float).
                                          If n_triplets is None, returns a dict mapping each
                                          triplet count to its AUC.
        
        Notes:
            - AUC is calculated by treating the model recovery as a binary classification:
              "this model" vs "not this model".
            - The method uses scikit-learn's roc_auc_score with the 'weighted' averaging strategy
              to account for potential class imbalance.
            - Returns 0.5 for chance-level performance or when calculation fails.
        """
        from sklearn.metrics import roc_auc_score
        
        if n_triplets is not None:
            triplet_counts = [n_triplets]
        else:
            triplet_counts = self.full_n_triplets_span
        
        results = {}
        
        for n_trip in triplet_counts:
            # Create binary true labels (1 for target model, 0 for others)
            y_true = []
            # Create predicted probabilities
            y_score = []
            
            # For each data generating model
            for dgm in self.data_generating_models_list:
                sim_results = self.get_model_simulations_results(dgm, n_trip)
                
                # Across all simulations for this data generating model
                for sim_idx, sim_data in sim_results.items():
                    # Get accuracy scores for all models in this simulation
                    if model_name not in sim_data:
                        continue
                    
                    for ref_model, acc in sim_data.items():
                        # Binary label: 1 if this is the target model, 0 otherwise
                        y_true.append(1 if ref_model == model_name else 0)
                        # Use accuracy as the predicted probability
                        y_score.append(acc)
            
            # Calculate AUC if we have enough data
            try:
                auc = roc_auc_score(y_true, y_score, average='weighted')
                results[n_trip] = auc
            except Exception:
                # Return 0.5 (chance level) if calculation fails
                results[n_trip] = 0.5
        
        # Return single value if specific n_triplets was requested
        if n_triplets is not None:
            return results[n_triplets]
        
        return results

    def get_auc_of_all_model_recovery(self, n_triplets=None):
        """
        Calculates the Area Under the Curve (AUC) for all models' recovery performance.
        
        This method computes AUC scores for all models and returns them in a dictionary,
        providing a comprehensive view of model discriminability across the entire set.
        
        Parameters:
            n_triplets (int, optional): Number of triplets used for analysis. If None,
                                      calculates AUC for all available triplet counts.
        
        Returns:
            Dict[str, Union[float, Dict[int, float]]]: A dictionary mapping model names to:
                                                     - AUC score (float) if n_triplets specified
                                                     - Dictionary of triplet counts to AUC scores
                                                       if n_triplets is None
        
        Notes:
            - For each model, the method calls get_auc_of_model_recovery to compute the AUC.
            - This provides a comprehensive view of how distinguishable each model is from
              the others based on its recovery performance.
            - Results are useful for comparing models' discriminability across different
              training data sizes.
        """
        results = {}
        
        for model in self.reference_models_list:
            results[model] = self.get_auc_of_model_recovery(model, n_triplets)
        
        return results

    def get_mrr_for_model(self, data_generating_model, n_triplets=None):
        """
        Calculates the Mean Reciprocal Rank (MRR) for a specific data-generating model.
        
        This method computes how well the correct data-generating model is ranked among
        all reference models based on accuracy, providing a measure of model recovery
        that considers the entire ranking rather than just correct/incorrect classification.
        
        Parameters:
            data_generating_model (str): Name of the data-generating model to calculate MRR for.
            n_triplets (int, optional): Number of triplets used for analysis. If None,
                                      calculates MRR for all available triplet counts.
        
        Returns:
            Union[float, Dict[int, float]]: If n_triplets is specified, returns MRR value (float).
                                          If n_triplets is None, returns a dict mapping each
                                          triplet count to its MRR value.
        
        Notes:
            - MRR is computed as the mean of reciprocal ranks across all simulations.
            - The rank is determined by ordering reference models by their test accuracies.
            - Higher MRR values (closer to 1.0) indicate better model recovery performance.
            - MRR = 1.0 means the correct model was always ranked first.
        """
        if n_triplets is not None:
            triplet_counts = [n_triplets]
        else:
            triplet_counts = self.full_n_triplets_span
        
        results = {}
        
        for n_trip in triplet_counts:
            # Get simulation results for this data generating model
            sim_results = self.get_model_simulations_results(data_generating_model, n_trip)
            
            reciprocal_ranks = []
            
            # For each simulation
            for sim_idx, sim_data in sim_results.items():
                # Skip if no data
                if not sim_data:
                    continue
                
                # Sort models by accuracy in descending order
                sorted_models = sorted(sim_data.items(), key=lambda x: x[1], reverse=True)
                
                # Find the rank of the true data generating model
                for rank, (model, _) in enumerate(sorted_models, 1):
                    if model == data_generating_model:
                        reciprocal_ranks.append(1.0 / rank)
                        break
                else:
                    # If model not found (shouldn't happen), use lowest possible rank
                    reciprocal_ranks.append(1.0 / len(sorted_models))
            
            # Calculate mean reciprocal rank
            if reciprocal_ranks:
                mrr = np.mean(reciprocal_ranks)
                results[n_trip] = mrr
            else:
                results[n_trip] = 0.0
        
        # Return single value if specific n_triplets was requested
        if n_triplets is not None:
            return results[n_triplets]
        
        return results

    def get_all_models_mean_reciprocal_ranks(self, n_triplets=None):
        """
        Calculates Mean Reciprocal Rank (MRR) for all data-generating models.
        
        This method computes MRR values for all data-generating models and returns them
        in a dictionary, providing a comprehensive view of model recovery performance
        across the entire model set based on ranking.
        
        Parameters:
            n_triplets (int, optional): Number of triplets used for analysis. If None,
                                      calculates MRR for all available triplet counts.
        
        Returns:
            Dict[str, Union[float, Dict[int, float]]]: A dictionary mapping model names to:
                                                     - MRR value (float) if n_triplets specified
                                                     - Dictionary of triplet counts to MRR values
                                                       if n_triplets is None
        
        Notes:
            - For each data-generating model, the method calls get_mrr_for_model to compute MRR.
            - This provides a comprehensive view of how well each model is recovered based
              on its ranking among all reference models.
            - Results are useful for comparing models' recoverability across different
              training data sizes.
            - The method also calculates and includes the mean MRR across all models under
              the key 'mean'.
        """
        results = {}
        
        for model in self.data_generating_models_list:
            results[model] = self.get_mrr_for_model(model, n_triplets)
        
        # Add mean across all models
        if n_triplets is not None:
            # Simple mean for single triplet count
            results['mean'] = np.mean([v for v in results.values()])
        else:
            # Mean per triplet count
            mean_results = {}
            for n_trip in self.full_n_triplets_span:
                mean_values = [results[model][n_trip] for model in self.data_generating_models_list 
                              if n_trip in results[model]]
                if mean_values:
                    mean_results[n_trip] = np.mean(mean_values)
                else:
                    mean_results[n_trip] = 0.0
            results['mean'] = mean_results
        
        return results
