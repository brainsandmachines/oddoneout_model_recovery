"""
Team-Based Model Recovery Analysis - Figures S4 and S5
Figures Generated:
------------------
1. Figure S4: Model recovery accuracy vs training set size
   - Line plot showing how model recovery accuracy across growing number of training triplets
   - Three confusion matrices at key training sizes (small, medium, large)

2. Figure S5: Team-based recovery analysis
   - Line plot comparing architecture-based vs objective-based team recovery
   - Six confusion matrices showing recovery patterns for both groupings

3. Supplementary ranking analysis figure
   - Bar plots showing mean model rankings when correctly vs incorrectly recovered (correspond to figure 3 with 30 models)

Key Concepts:
------------
- Model Recovery: Identifying which model generated the data by comparing test accuracies
- Team Recovery: Success when the best model comes from the same group (architecture or objective)

Usage:
------
    python create_figure_S5_S4.py

    Configuration is loaded from: scripts_configurations/figure_S5.yaml

Output Structure:
----------------
    Plots/grouped_analysis/
        ├── figure_S4_model_recovery_accuracy.pdf
        ├── figure_S5_team_recovery.pdf
        ├── supplementary_ranking_analysis.pdf
        └── analysis_data/
            ├── figure_S4_accuracy_by_training_size.csv
            ├── overall_model_recovery.csv
            └── team_recovery_by_group.csv

Dependencies:
------------
- Hydra: Configuration management
- Pandas/NumPy: Data manipulation
- Matplotlib/Seaborn: Visualization
- SciPy: Statistical calculations (binomial confidence intervals)
- SQLite3: Database access for simulation results
"""

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
import matplotlib
import matplotlib.gridspec as gridspec

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Union, Dict, Optional, List, Tuple,cast
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import sqlite3
import hydra
from omegaconf import DictConfig

from tools.db_analysis import DBResultsAnalysis
from figures_scripts.create_Figures_1_and_3 import create_colors_list

# ============================================================================
# GLOBAL FONT SIZE CONSTANTS FOR STANDARDIZED FIGURE FORMATTING
# ============================================================================
# These constants ensure consistent typography across all generated figures,
# following publication standards for scientific journals.

# Panel letters (A, B, C, D, etc.) - Bold uppercase letters marking figure panels
PANEL_LETTER_FONTSIZE = 10
PANEL_LETTER_FONTWEIGHT = "bold"

# Line plot annotation letters - Smaller letters pointing to specific data points
LINE_ANNOTATION_FONTSIZE = 8

# Axis labels (e.g., "number of training triplets", "recovery accuracy")
AXIS_LABEL_FONTSIZE = 8
AXIS_LABEL_FONTWEIGHT = "bold"

# Axis tick labels (numerical values on axes)
TICK_LABEL_FONTSIZE = 5

# Legend text
LEGEND_FONTSIZE = 5

# Colorbar labels and ticks (for confusion matrix heatmaps)
COLORBAR_LABEL_FONTSIZE = 6
COLORBAR_TICK_FONTSIZE = 5

# Figure titles (rarely used in publication figures)
TITLE_FONTSIZE = 12
TITLE_FONTWEIGHT = "bold"

# Confusion matrix cell annotations (numbers inside heatmap cells)
HEATMAP_ANNOT_FONTSIZE = 4

# ============================================================================


class TeamRecoveryVisualizer(DBResultsAnalysis):
    """
    Visualizer for team-based model recovery analysis.

    This class extends DBResultsAnalysis to create visualizations showing how well
    models can be recovered when grouped by architecture (CNN vs ViT) or training
    objective (Supervised, Self-supervised, Image & Text Alignment).

    Team Recovery Concept:
    ---------------------
    Instead of requiring exact model identification, team recovery considers it a
    success when the best-performing model belongs to the same group as the
    data-generating model. For example, if a CNN generated the data and any CNN
    achieves the highest test accuracy, that counts as successful team recovery.

    Figures Created:
    ---------------
    - Figure S4: Model recovery accuracy vs number of training triplets
                 (line plot with confusion matrices at three key training sizes)

    - Figure S5: Combined team recovery analysis with line plots and confusion
                 matrices for both architecture-based and objective-based groupings

    - Supplementary ranking figure: Mean ranking analysis showing which models
                 are consistently ranked highly even when incorrect

    Key Attributes:
    --------------
    models_architecture : dict
        Maps each model name to its architecture type ('CNN' or 'ViT')

    models_objective : dict
        Maps each model name to its training objective
        ('Supervised', 'Self-supervised', or 'Image & Text Alignment')

    architecture_groups : dict
        Maps architecture types to lists of models with that architecture

    objective_groups : dict
        Maps training objectives to lists of models with that objective

    Parameters:
    ----------
    db_path : str
        Path to SQLite database containing simulation results

    data_generating_models_list : list
        List of model names used as data generators in simulations

    simulations_list : list, optional
        List of simulation indices to analyze (default: all available)

    formal_names_dict : dict, optional
        Mapping from internal model names to publication-ready display names

    models_color_dict : dict, optional
        Color assignments for each model (for consistent visualization)

    models_architecture : dict, optional
        Architecture type for each model (required for architecture-based analysis)

    models_objective : dict, optional
        Training objective for each model (required for objective-based analysis)

    models_order_as_colors_dict : list, optional
        Preferred ordering of models for bar plots

    save_plot_folder : str, optional
        Directory where figures and data files will be saved
        (default: "Plots/grouped_analysis/")

    reference_models_list : list, optional
        List of models to use as reference/candidate models
        (default: same as data_generating_models_list)

    n_triplets_list : list, optional
        Training set sizes to analyze (default: standard experimental sizes)

    Example:
    -------
    >>> visualizer = TeamRecoveryVisualizer(
    ...     db_path="path/to/results.db",
    ...     data_generating_models_list=model_names,
    ...     models_architecture=arch_dict,
    ...     models_objective=obj_dict,
    ...     save_plot_folder="Plots/output/"
    ... )
    >>> visualizer.create_figure_S4(n_triplets_list=[100, 400, 1600])
    """

    def __init__(
        self,
        db_path: str,
        data_generating_models_list: list,
        simulations_list: Optional[list] = None,
        formal_names_dict: Optional[dict] = None,
        models_color_dict: Optional[dict] = None,
        models_architecture: Optional[dict] = None,
        models_objective: Optional[dict] = None,
        models_order_as_colors_list: Optional[list] = None,
        save_plot_folder: str = "Plots/grouped_analysis/",
        reference_models_list: Optional[list] = None,
        n_triplets_list: Optional[list] = None,
    ):

        if reference_models_list is None:
            reference_models_list = data_generating_models_list

        super().__init__(
            db_path,
            data_generating_models_list,
            simulations_list,
            reference_models_list,
        )

        # Store additional attributes
        self.formal_names_dict = formal_names_dict or {}
        self.models_color_dict = models_color_dict or {}
        self.models_architecture = models_architecture or {}
        self.models_objective = models_objective or {}
        self.models_order_as_colors_dict = models_order_as_colors_list or []
        self.save_plot_folder = save_plot_folder

        # Store n_triplets_list for validation
        if n_triplets_list is None:
            # Default values from the experiments
            self.n_triplets_list = [
                100,
                400,
                1600,
                6400,
                25600,
                51200,
                204800,
                819200,
                1638400,
                4200000,
            ]
        else:
            self.n_triplets_list = n_triplets_list

        # Create save folder if it doesn't exist
        if not os.path.exists(self.save_plot_folder):
            os.makedirs(self.save_plot_folder)

        # Get available regularization combinations first
        self.regularization_combinations = self._get_regularization_combinations()

        # Filter simulations to only include those with complete data
        self.filtered_simulations_per_combo = self.get_complete_simulations_per_combo()

        # Create group mappings
        self._create_group_mappings()

        # Print architecture and objective group statistics are printed in main()
        # self.print_model_group_stats()  # This is already printed in main()

    def _lowercase_strings(self, string_list: List[str]) -> List[str]:
        """
        Convert a list of strings to lowercase.

        Args:
            string_list: List of strings to convert

        Returns:
            List of lowercase strings
        """
        return [s.lower() for s in string_list]

    def _create_group_mappings(self):
        """
        Create mappings for architecture and objective groups.

        This method organizes models into groups based on their architecture type
        (CNN vs ViT) and training objective (Supervised, Self-supervised, or
        Image & Text Alignment). These groupings are used for team recovery analysis.

        Creates:
        -------
        self.architecture_groups : dict
            Maps architecture names to lists of model names
            Example: {'CNN': ['ResNet50', 'VGG16'], 'ViT': ['ViT_Base']}

        self.models_by_architecture : dict
            Same as architecture_groups (maintained for compatibility)

        self.objective_groups : dict
            Maps objective names to lists of model names
            Example: {'Supervised': ['ResNet50'], 'Self-supervised': ['SimCLR']}

        self.models_by_objective : dict
            Same as objective_groups (maintained for compatibility)

        Also prints a summary of group statistics to console.
        """
        # Architecture groups
        self.architecture_groups = {}
        self.models_by_architecture = {}
        for model, arch in self.models_architecture.items():
            if model in self.data_generating_models_list:
                if arch not in self.architecture_groups:
                    self.architecture_groups[arch] = []
                    self.models_by_architecture[arch] = []
                self.architecture_groups[arch].append(model)
                self.models_by_architecture[arch].append(model)

        # Objective groups
        self.objective_groups = {}
        self.models_by_objective = {}
        for model, obj in self.models_objective.items():
            if model in self.data_generating_models_list:
                if obj not in self.objective_groups:
                    self.objective_groups[obj] = []
                    self.models_by_objective[obj] = []
                self.objective_groups[obj].append(model)
                self.models_by_objective[obj].append(model)

        print(f"Architecture groups: {list(self.architecture_groups.keys())}")
        print(f"Objective groups: {list(self.objective_groups.keys())}")
        for arch, models in self.models_by_architecture.items():
            print(f"  {arch}: {len(models)} models")
        for obj, models in self.models_by_objective.items():
            print(f"  {obj}: {len(models)} models")

    def get_complete_simulations_per_combo(self) -> dict:
        """
        Identify simulations with complete data for each regularization combination.

        A simulation is considered "complete" only if it has results for ALL models
        at ALL training triplet sizes for a given regularization combination. This
        ensures fair comparisons across different experimental conditions.

        Returns:
        -------
        dict
            Maps (dg_regularization, candidate_regularization) tuples to lists of
            valid simulation indices. Example:
            {('eye_distance', 'eye_distance'): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        Notes:
        -----
        - Incomplete simulations are automatically excluded from analysis
        - Prints summary statistics showing X/Y complete simulations for each combo
        - This filtering prevents biased results from missing data
        """
        complete_simulations = {}

        print("Filtering simulations for complete data...")

        for dg_reg, candidate_reg in self.regularization_combinations:
            valid_simulations = []

            # Check each simulation for completeness across all models and n_triplets
            for sim_idx in self.simulations_list:
                is_complete = True

                # Check if this simulation has data for all models at all n_triplets values
                for model in self.data_generating_models_list:
                    for n_triplets in self.n_triplets_list:
                        # Check if this simulation has results for this model/n_triplets/combo
                        filtered_jobs = self.jobs_df[
                            (self.jobs_df["data_generating_model"] == model)
                            & (self.jobs_df["simulation_idx"] == sim_idx)
                            & (self.jobs_df["n_train_triplets"] == n_triplets)
                            & (
                                self.jobs_df["data_generating_model_regularization"]
                                == dg_reg
                            )
                            & (
                                self.jobs_df["candidate_model_regularization"]
                                == candidate_reg
                            )
                        ]

                        if len(filtered_jobs) == 0:
                            is_complete = False
                            break

                    if not is_complete:
                        break

                if is_complete:
                    valid_simulations.append(sim_idx)

            complete_simulations[(dg_reg, candidate_reg)] = valid_simulations
            print(
                f"  {dg_reg}->{candidate_reg}: {len(valid_simulations)}/{len(self.simulations_list)} complete simulations"
            )

        return complete_simulations

    def _load_data(self, print_info: bool = False):
        """Load experiment data from the SQLite database including regularization info."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load jobs data with regularization information
                jobs_df = pd.read_sql_query(
                    """
                    SELECT id, simulation_idx, data_generating_model, n_train_triplets, status,
                           data_generating_model_regularization, candidate_model_regularization
                    FROM jobs
                """,
                    conn,
                )

                # Load reference model results
                ref_results_df = pd.read_sql_query(
                    """
                    SELECT job_id, reference_model, test_accuracy,
                           data_generating_model_regularization, candidate_model_regularization
                    FROM reference_model_results
                """,
                    conn,
                )

                if print_info:
                    print(
                        f"Loaded {len(jobs_df)} jobs and {len(ref_results_df)} reference results"
                    )

                # Filter for models of interest
                jobs_df = jobs_df[
                    jobs_df["data_generating_model"].isin(
                        self.data_generating_models_list
                    )
                ]
                ref_results_df = ref_results_df[
                    ref_results_df["reference_model"].isin(self.reference_models_list)
                ]

                return jobs_df, ref_results_df

        except Exception as e:
            print(f"Error loading data from database: {str(e)}")
            raise

    def _get_regularization_combinations(self):
        """Get all unique regularization combinations from the database."""
        if hasattr(self, "jobs_df") and self.jobs_df is not None:
            try:
                combinations = self.jobs_df[
                    [
                        "data_generating_model_regularization",
                        "candidate_model_regularization",
                    ]
                ].drop_duplicates()
                return [
                    (
                        row["data_generating_model_regularization"],
                        row["candidate_model_regularization"],
                    )
                    for _, row in combinations.iterrows()
                ]
            except:
                return []
        return []

    def go_formal(self, model_name):
        """Convert internal model names to formal display names."""
        if self.formal_names_dict is None:
            return model_name

        if isinstance(model_name, list):
            return [self.formal_names_dict.get(name, name) for name in model_name]
        elif isinstance(model_name, pd.DataFrame):
            df = model_name.copy()
            index_mapping = {
                model: self.formal_names_dict.get(model, model) for model in df.index
            }
            column_mapping = {
                model: self.formal_names_dict.get(model, model) for model in df.columns
            }
            df.rename(index=index_mapping, columns=column_mapping, inplace=True)
            return df
        else:
            return self.formal_names_dict.get(model_name, model_name)

    def get_model_simulations_results_for_combo(
        self,
        data_generating_model_name: str,
        n_triplets: int,
        dg_regularization: str,
        candidate_regularization: str,
    ):
        """Get simulation results for a specific regularization combination."""
        # Filter jobs for this specific combination
        filtered_jobs = self.jobs_df[
            (self.jobs_df["data_generating_model"] == data_generating_model_name)
            & (self.jobs_df["simulation_idx"].isin(self.simulations_list))
            & (self.jobs_df["n_train_triplets"] == n_triplets)
            & (
                self.jobs_df["data_generating_model_regularization"]
                == dg_regularization
            )
            & (
                self.jobs_df["candidate_model_regularization"]
                == candidate_regularization
            )
        ]

        if len(filtered_jobs) == 0:
            return {}

        # Merge with reference results
        merged = pd.merge(
            filtered_jobs, self.ref_results_df, left_on="id", right_on="job_id"
        )
        merged = merged.drop_duplicates(subset=["simulation_idx", "reference_model"])
        valid_refs = set(self.reference_models_list)
        merged = merged[merged["reference_model"].isin(valid_refs)]

        # Group results by simulation index
        results_dict = {}
        if len(merged) > 0:
            for sim_idx, group in merged.groupby("simulation_idx"):
                ref_acc_dict = dict(
                    zip(group["reference_model"], group["test_accuracy"])
                )
                results_dict[sim_idx] = ref_acc_dict

        return results_dict

    def get_total_model_recovery_accuracy_for_combo(
        self, n_triplets: int, dg_regularization: str, candidate_regularization: str
    ) -> float:
        """
        Calculate overall model recovery accuracy for a specific experimental condition.

        Model recovery is considered successful when the model with the highest test
        accuracy on a triplet embedding task matches the model that generated the data.
        This is the strictest criterion - requiring exact model identification.

        Parameters:
        ----------
        n_triplets : int
            Number of training triplets used (e.g., 100, 400, 25600)

        dg_regularization : str
            Regularization type for data-generating models (e.g., 'eye_distance')

        candidate_regularization : str
            Regularization type for candidate models (e.g., 'eye_distance')

        Returns:
        -------
        float
            Recovery accuracy as a proportion from 0.0 to 1.0
            Example: 0.75 means 75% of simulations correctly identified the model

        Notes:
        -----
        - Only uses simulations with complete data (see get_complete_simulations_per_combo)
        - Each simulation tests one data-generating model against all candidate models
        - The candidate with highest test accuracy is considered the "recovered" model
        """
        # Use only the complete simulations for this combination
        combo_key = (dg_regularization, candidate_regularization)
        if combo_key in self.filtered_simulations_per_combo:
            valid_simulations = self.filtered_simulations_per_combo[combo_key]
        else:
            valid_simulations = self.simulations_list

        total_correct = 0
        total_simulations = 0

        for data_generating_model in self.data_generating_models_list:
            results_dict = self.get_model_simulations_results_for_combo(
                data_generating_model,
                n_triplets,
                dg_regularization,
                candidate_regularization,
            )

            for sim_idx in valid_simulations:
                if sim_idx in results_dict:
                    sim_results = results_dict[sim_idx]
                    if sim_results:  # Check if not empty
                        best_model = max(sim_results, key=sim_results.get)
                        if best_model == data_generating_model:
                            total_correct += 1
                        total_simulations += 1

        return total_correct / total_simulations if total_simulations > 0 else 0.0

    def get_combo_recovery_binomial_error(
        self,
        n_triplets: int,
        dg_regularization: str,
        candidate_regularization: str,
        CI: bool = False,
    ) -> Union[float, Tuple[float, float]]:
        """
        Calculate error bounds for model recovery accuracy.

        Provides either binomial confidence intervals (95% CI) or standard error
        for model recovery accuracy, enabling statistical comparison of results.

        Parameters:
        ----------
        n_triplets : int
            Number of training triplets

        dg_regularization : str
            Data-generating model regularization type

        candidate_regularization : str
            Candidate model regularization type

        CI : bool, optional
            If True, returns 95% binomial confidence interval (lower, upper)
            If False, returns standard error (default: False)

        Returns:
        -------
        float or tuple
            If CI=False: Standard error as a single float
            If CI=True: Tuple of (lower_bound, upper_bound) for 95% CI

        Example:
        -------
        >>> # Get standard error
        >>> error = visualizer.get_combo_recovery_binomial_error(
        ...     n_triplets=25600,
        ...     dg_regularization='eye_distance',
        ...     candidate_regularization='eye_distance',
        ...     CI=False
        ... )
        >>> print(f"Recovery accuracy: {accuracy:.2f} ± {error:.2f}")

        >>> # Get 95% confidence interval
        >>> lower, upper = visualizer.get_combo_recovery_binomial_error(
        ...     n_triplets=25600,
        ...     dg_regularization='eye_distance',
        ...     candidate_regularization='eye_distance',
        ...     CI=True
        ... )
        >>> print(f"95% CI: [{lower:.2f}, {upper:.2f}]")
        """
        # Use only the complete simulations for this combination
        combo_key = (dg_regularization, candidate_regularization)
        if combo_key in self.filtered_simulations_per_combo:
            valid_simulations = self.filtered_simulations_per_combo[combo_key]
        else:
            valid_simulations = self.simulations_list

        total_correct = 0
        total_simulations = 0

        for data_generating_model in self.data_generating_models_list:
            results_dict = self.get_model_simulations_results_for_combo(
                data_generating_model,
                n_triplets,
                dg_regularization,
                candidate_regularization,
            )

            for sim_idx in valid_simulations:
                if sim_idx in results_dict:
                    sim_results = results_dict[sim_idx]
                    if sim_results:  # Check if not empty
                        best_model = max(sim_results, key=sim_results.get)
                        if best_model == data_generating_model:
                            total_correct += 1
                        total_simulations += 1

        if total_simulations == 0:
            return (0.0, 0.0) if CI else 0.0

        # Calculate binomial confidence interval
        from scipy import stats

        if CI:
            # Use binomial confidence interval for CI=True
            ci_lower, ci_upper = stats.binom.interval(
                0.95, total_simulations, total_correct / total_simulations
            )
            return (ci_lower / total_simulations, ci_upper / total_simulations)
        else:
            # Use standard error for CI=False
            p = total_correct / total_simulations
            se = (p * (1 - p) / total_simulations) ** 0.5
            return se

    def confusion_matrix_for_combo(
        self, n_triplets: int, dg_regularization: str, candidate_regularization: str
    ) -> pd.DataFrame:
        """
        Generate confusion matrix showing model recovery patterns.

        Creates a matrix where rows are data-generating models and columns are
        recovered (best-performing) models. Cell values show how many times each
        model was recovered when a particular model generated the data.

        Parameters:
        ----------
        n_triplets : int
            Number of training triplets

        dg_regularization : str
            Data-generating model regularization

        candidate_regularization : str
            Candidate model regularization

        Returns:
        -------
        pd.DataFrame
            Confusion matrix with data-generating models as rows,
            recovered models as columns, and counts as values

        Example:
        -------
        The matrix might show:
                        ResNet50  ViT_Base  CLIP_Large
        ResNet50            25         3           2
        ViT_Base             2        26           2
        CLIP_Large           1         1          28

        This means when ResNet50 generated data, it was correctly recovered 25 times,
        while ViT_Base was incorrectly selected 3 times, etc.
        """
        confusion_matrix = pd.DataFrame(
            index=self.data_generating_models_list,
            columns=self.reference_models_list,
            dtype=int,
        ).fillna(0)

        for data_generating_model in self.data_generating_models_list:
            results_dict = self.get_model_simulations_results_for_combo(
                data_generating_model,
                n_triplets,
                dg_regularization,
                candidate_regularization,
            )

            # Initialize counts for each reference model
            model_counts = {model: 0 for model in self.reference_models_list}

            # For each simulation, determine the best performing reference model
            for sim_idx in self.simulations_list:
                if sim_idx in results_dict and len(results_dict[sim_idx]) > 0:
                    sim_results = results_dict[sim_idx]
                    best_model = max(sim_results, key=sim_results.get)
                    model_counts[best_model] += 1

            for model, count in model_counts.items():
                confusion_matrix.loc[data_generating_model, model] = count

        return confusion_matrix

    def create_figure_layout(
        self,
        width_cm: float,
        height_cm: float,
        gridspec_dict: dict,
        subplot_specifics_dict: dict,
    ):
        """Create a figure with precise layout control using GridSpec."""
        width_inches = width_cm / 2.54
        height_inches = height_cm / 2.54
        fig = plt.figure(
            figsize=(width_inches, height_inches), constrained_layout=False
        )

        gs = gridspec.GridSpec(
            **gridspec_dict, left=0, right=1, bottom=0, top=1, wspace=0, hspace=0
        )

        subplots_dict = {}
        for name, args in subplot_specifics_dict.items():
            h_index = args["height_index"]
            w_index = args["width_index"]
            if not isinstance(w_index, int):
                w_index = slice(min(w_index), max(w_index) + 1)
            if not isinstance(h_index, int):
                h_index = slice(min(h_index), max(h_index) + 1)
            ax = fig.add_subplot(gs[h_index, w_index])
            # Don't turn off frame here - let individual plot functions control spines
            # ax.set_frame_on(False)
            ax.grid(False)
            ax.margins(0, 0)
            subplots_dict[name] = ax

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        return fig, subplots_dict

    def draw_confusion_matrix_for_combo(
        self,
        ax,
        n_triplets: int,
        dg_regularization: str,
        candidate_regularization: str,
        show_x_label: bool = True,
        show_y_label: bool = True,
        show_x_ticks: bool = True,
        show_y_ticks: bool = True,
        show_colorbar: bool = True,
        colorbar_ax=None,
        panel_letter: Optional[str] = None,
    ):
        """Create a confusion matrix heatmap for a specific regularization combination."""
        confusion_matrix = self.confusion_matrix_for_combo(
            n_triplets, dg_regularization, candidate_regularization
        )

        # Verify data integrity - like in the original working version
        for model in self.data_generating_models_list:
            row_sum = confusion_matrix.loc[model].sum()
            assert row_sum == len(
                self.simulations_list
            ), f"{model} has data problem for {dg_regularization}->{candidate_regularization}"

        confusion_matrix = confusion_matrix.astype(int)

        # Create custom colormap
        n_colors = 256
        colors_base = plt.cm.get_cmap("YlOrRd")(np.linspace(0, 1, n_colors))
        zero_color = [0.95, 0.95, 1, 1]
        colors_base[0] = zero_color
        colors_base[1:] = plt.cm.get_cmap("YlOrRd")(np.linspace(0.1, 1, n_colors - 1))
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_YlOrRd", colors_base, N=n_colors
        )

        vmax = float(confusion_matrix.to_numpy().max())

        # Create the heatmap
        sns.heatmap(
            self.go_formal(confusion_matrix),
            annot=False,
            cmap=custom_cmap,
            vmin=0,
            vmax=vmax,
            linewidths=0.1,
            linecolor="gray",
            cbar=show_colorbar,
            cbar_kws={
                "shrink": 0.6,
                "aspect": 50,
                "ticks": np.linspace(0, vmax, 10),
                "format": "%d",
            },
            cbar_ax=colorbar_ax,
            square=True,
            ax=ax,
        )

        if show_colorbar and colorbar_ax is not None:
            colorbar_ax.set_ylabel(
                "number of simulations", fontsize=COLORBAR_LABEL_FONTSIZE, labelpad=1
            )
            colorbar_ax.yaxis.set_label_position("left")
            colorbar_ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

        # Set labels
        if show_x_label:
            ax.set_xlabel(
                "recovered model",
                fontsize=AXIS_LABEL_FONTSIZE,
                fontweight=AXIS_LABEL_FONTWEIGHT,
            )
        else:
            ax.set_xlabel("")

        if show_y_label:
            ax.set_ylabel(
                "data-generating model",
                fontsize=AXIS_LABEL_FONTSIZE,
                fontweight=AXIS_LABEL_FONTWEIGHT,
            )
        else:
            ax.set_ylabel("")

        # Handle tick labels
        ax.set_xticks(np.arange(len(confusion_matrix.columns)) + 0.5)
        ax.set_yticks(np.arange(len(confusion_matrix.index)) + 0.5)

        if show_x_ticks:
            formal_matrix = cast(pd.DataFrame, self.go_formal(confusion_matrix))
            ax.set_xticklabels(
                formal_matrix.columns,
                rotation=90,
                ha="right",
                va="top",
                fontsize=HEATMAP_ANNOT_FONTSIZE,
            )
        else:
            ax.set_xticklabels([])

        if show_y_ticks:
            ax.set_yticklabels(
                self.go_formal(confusion_matrix).index, fontsize=HEATMAP_ANNOT_FONTSIZE
            )
        else:
            ax.set_yticklabels([])

        # Add panel letter if provided
        if panel_letter is not None:
            ax.text(
                -0.1,
                1.1,
                panel_letter,
                transform=ax.transAxes,
                fontsize=PANEL_LETTER_FONTSIZE,
                fontweight=PANEL_LETTER_FONTWEIGHT,
            )

        return ax

    def draw_line_plot_for_combo(
        self,
        ax,
        n_triplets_list: list,
        dg_regularization: str,
        candidate_regularization: str,
        x_label: str,
        y_label: str,
        panel_letter: Optional[str] = None,
        show_error_bars: bool = True,
        CI: bool = False,
    ):
        """Create a line plot for a specific regularization combination."""
        # Calculate accuracy for each n_triplets
        accuracy_values = []
        error_values = []

        for n_triplets in n_triplets_list:
            accuracy = self.get_total_model_recovery_accuracy_for_combo(
                n_triplets, dg_regularization, candidate_regularization
            )
            accuracy_values.append(accuracy)

            if show_error_bars:
                error = self.get_combo_recovery_binomial_error(
                    n_triplets, dg_regularization, candidate_regularization, CI=CI
                )
                error_values.append(error)

        # Format x-axis labels
        x_labels = []
        for n in n_triplets_list:
            if n >= 1_000_000:
                x_labels.append(f"{n/1_000_000:.1f}M")
            elif n >= 1000:
                x_labels.append(f"{n/1000:.1f}K")
            else:
                x_labels.append(str(n))

        (line,) = ax.plot(x_labels, accuracy_values, "o-", markersize=2, linewidth=1)

        # Add error patches if requested
        if show_error_bars and error_values:
            error_color = line.get_color()
            if CI:
                lower_bound = [err[0] for err in error_values]
                upper_bound = [err[1] for err in error_values]
            else:
                lower_bound = np.array(accuracy_values) - np.array(error_values)
                upper_bound = np.array(accuracy_values) + np.array(error_values)

            lower_bound = np.maximum(lower_bound, 0)
            ax.fill_between(
                range(len(x_labels)),
                lower_bound,
                upper_bound,
                alpha=0.2,
                color=error_color,
                linewidth=0,
            )

        # Formatting
        ax.set_xlabel(
            x_label, fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT
        )
        ax.set_ylabel(
            y_label, fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT
        )
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, len(x_labels) - 0.5)
        y_tick_labels = list(np.arange(0.0, 1.1, 0.1))
        ax.set_yticks(y_tick_labels)
        ax.set_yticklabels(
            [f"{x:.1f}" for x in y_tick_labels], fontsize=TICK_LABEL_FONTSIZE
        )
        ax.set_xticklabels(x_labels, fontsize=TICK_LABEL_FONTSIZE)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.3)
        ax.tick_params(
            axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE, width=0.3
        )

        if panel_letter is not None:
            ax.text(
                -0.05,
                0.98,
                panel_letter,
                transform=ax.transAxes,
                fontsize=PANEL_LETTER_FONTSIZE,
                fontweight=PANEL_LETTER_FONTWEIGHT,
            )

        return ax

    def create_figure_S4(
        self,
        n_triplets_list: list,
        dg_regularization: str,
        candidate_regularization: str,
        width_cm: float = 20,
        height_cm: float = 13,
        conf_size_dict: Optional[dict] = None,
        show_error_bars: bool = True,
        CI: bool = False,
    ):
        """
        Create Figure S4: Model recovery accuracy vs number of training triplets.

        This figure shows how model recovery accuracy improves with more training data,
        including confusion matrices at three key training set sizes (small, medium, large).

        Args:
            n_triplets_list: List of training triplet counts to plot
            dg_regularization: Data-generating model regularization type
            candidate_regularization: Candidate model regularization type
            width_cm: Figure width in centimeters (default: 20)
            height_cm: Figure height in centimeters (default: 13)
            conf_size_dict: Optional dictionary specifying confusion matrix panel sizes
            show_error_bars: Whether to display error bars/confidence intervals
            CI: If True, use 95% binomial CI; if False, use standard error
        """
        # Define gridspec layout (same as original Figure 1)
        gridspec_dict = {
            "nrows": 7,
            "ncols": 10,
            "height_ratios": [0.3, 2.9266, 0.7, 0.5, 2.9266666, 0.1, 1.4],
            "width_ratios": [
                1.2,
                1,
                2.92666666,
                0.7,
                2.92666666,
                0.7,
                2.9266666,
                0.5,
                0.5,
                0.5,
            ],
        }

        subplots_dict = {
            "model_recovery_n_triplets": {
                "height_index": 1,
                "width_index": [1, 2, 3, 4, 5, 6, 7, 8],
            },
            "conf_matrix_small": {"height_index": 4, "width_index": 2},
            "conf_matrix_medium": {"height_index": 4, "width_index": 4},
            "conf_matrix_large": {"height_index": 4, "width_index": 6},
            "colorbar": {"height_index": 4, "width_index": 8},
        }

        if conf_size_dict is None:
            conf_size_dict = {"small": 400, "medium": 25600, "large": 1638400}

        # Save figure directly in the main plot folder
        os.makedirs(self.save_plot_folder, exist_ok=True)
        save_path = f"{self.save_plot_folder}/figure_S4_model_recovery_accuracy.pdf"

        # Calculate model recovery accuracies and errors for CSV export
        accuracy_values = []
        error_values = []
        for n_triplets in n_triplets_list:
            accuracy = self.get_total_model_recovery_accuracy_for_combo(
                n_triplets, dg_regularization, candidate_regularization
            )
            accuracy_values.append(accuracy)
            if show_error_bars:
                error = self.get_combo_recovery_binomial_error(
                    n_triplets, dg_regularization, candidate_regularization, CI=CI
                )
                error_values.append(error)

        # Save CSV with training set size vs model recovery data
        analysis_save_path = os.path.join(
            self.save_plot_folder,
            "grouped_analysis",
            "figure_S4_accuracy_by_training_size.csv",
        )
        os.makedirs(os.path.dirname(analysis_save_path), exist_ok=True)

        # Create DataFrame with analysis data
        data_dict = {
            "n_triplets": n_triplets_list,
            "model_recovery_accuracy": accuracy_values,
        }

        if show_error_bars and error_values:
            # Unpack error values if they are tuples (low_CI, high_CI)
            if error_values and isinstance(error_values[0], (tuple, list)):
                low_errors, high_errors = zip(*error_values)
                data_dict["low_CI"] = list(low_errors)
                data_dict["high_CI"] = list(high_errors)
            else:
                data_dict["error"] = error_values

        analysis_df = pd.DataFrame(data_dict)
        analysis_df.to_csv(analysis_save_path, index=False)
        print(f"💾 Figure S4 analysis data saved: {analysis_save_path}")

        # Create the proper figure layout
        fig, ax = self.create_figure_layout(
            width_cm, height_cm, gridspec_dict, subplots_dict
        )

        # Draw line plot
        self.draw_line_plot_for_combo(
            ax["model_recovery_n_triplets"],
            n_triplets_list,
            dg_regularization,
            candidate_regularization,
            "number of training triplets",
            "model recovery accuracy",
            panel_letter="A",
            show_error_bars=show_error_bars,
            CI=CI,
        )

        # Draw confusion matrices
        for size, n_triplets in conf_size_dict.items():
            ax_name = f"conf_matrix_{size}"
            show_x_label = size == "small"
            show_y_label = size == "small"
            show_x_ticks = True
            show_y_ticks = size == "small"
            show_colorbar = size == "large"
            colorbar_ax = ax["colorbar"] if size == "large" else None
            panel_letter = {"small": "B", "medium": "C", "large": "D"}[size]

            self.draw_confusion_matrix_for_combo(
                ax[ax_name],
                n_triplets,
                dg_regularization,
                candidate_regularization,
                show_x_label=show_x_label,
                show_y_label=show_y_label,
                show_x_ticks=show_x_ticks,
                show_y_ticks=show_y_ticks,
                show_colorbar=show_colorbar,
                colorbar_ax=colorbar_ax,
                panel_letter=panel_letter,
            )

        plt.savefig(save_path, dpi=600, format="pdf")
        plt.close()
        print(f"✅ Figure S4 saved to: {save_path}")

    def create_combined_line_plot(
        self, n_triplets_list: list, width_cm: float = 25, height_cm: float = 15
    ):
        """Create combined line plot showing all regularization combinations."""
        # Create colors for each combination
        colors = create_colors_list(
            len(self.regularization_combinations), start=0, rot=0.8
        )

        # Format x-axis labels
        x_labels = []
        for n in n_triplets_list:
            if n >= 1_000_000:
                x_labels.append(f"{n/1_000_000:.1f}M")
            elif n >= 1000:
                x_labels.append(f"{n/1000:.1f}K")
            else:
                x_labels.append(str(n))

        # Create figure
        width_inches = width_cm / 2.54
        height_inches = height_cm / 2.54
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))

        # For each regularization combination
        for i, (dg_reg, candidate_reg) in enumerate(self.regularization_combinations):
            # combo_label = f"DG: {dg_reg}, Candidate: {candidate_reg}"

            # Calculate accuracy for each n_triplets
            accuracy_values = []
            error_values = []

            for n_triplets in n_triplets_list:
                accuracy = self.get_total_model_recovery_accuracy_for_combo(
                    n_triplets, dg_reg, candidate_reg
                )
                accuracy_values.append(accuracy)

                error = self.get_combo_recovery_binomial_error(
                    n_triplets, dg_reg, candidate_reg, CI=True
                )
                error_values.append(error)

            # Plot line
            (line,) = ax.plot(
                range(len(n_triplets_list)),
                accuracy_values,
                "o-",
                markersize=4,
                linewidth=2,
                color=colors[i],
                label=[],
            )

            # Add confidence interval bands
            lower_bounds = [err[0] for err in error_values]
            upper_bounds = [err[1] for err in error_values]
            ax.fill_between(
                range(len(n_triplets_list)),
                lower_bounds,
                upper_bounds,
                alpha=0.2,
                color=colors[i],
            )

        # Formatting
        ax.set_xlabel(
            "number of training triplets",
            fontsize=AXIS_LABEL_FONTSIZE,
            fontweight=AXIS_LABEL_FONTWEIGHT,
        )
        ax.set_ylabel(
            "model recovery accuracy",
            fontsize=AXIS_LABEL_FONTSIZE,
            fontweight=AXIS_LABEL_FONTWEIGHT,
        )
        ax.set_title(
            "model recovery accuracy vs training triplets\nfor different regularization combinations",
            fontsize=TITLE_FONTSIZE,
            fontweight=TITLE_FONTWEIGHT,
        )

        # Set axis properties
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, len(n_triplets_list) - 0.5)
        ax.set_xticks(range(len(n_triplets_list)))
        ax.set_xticklabels(x_labels, fontsize=TICK_LABEL_FONTSIZE)

        # Set y-axis ticks
        y_ticks = np.linspace(0.0, 1.0, 11).tolist()
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(
            [f"{x:.2f}" if x in [0.65, 0.75] else f"{x:.1f}" for x in y_ticks],
            fontsize=TICK_LABEL_FONTSIZE,
        )

        # Grid and spines
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)

        # Legend below the plot
        ax.legend(
            bbox_to_anchor=(0.5, -0.15),
            loc="upper center",
            fontsize=LEGEND_FONTSIZE,
            ncol=2,
        )

        # Save figure
        save_path = f"{self.save_plot_folder}/combined_regularization_line_plot.pdf"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.close()

        print(f"Combined line plot saved to: {save_path}")

    def calculate_models_mean_rank_for_combo(
        self, n_triplets: int, dg_regularization: str, candidate_regularization: str
    ):
        """Calculate mean ranks for models when correct and incorrect for a specific combo."""
        correct_ranks_dict = {}
        false_ranks_dict = {}
        correct_ranks_dict_std = {}
        false_ranks_dict_std = {}

        # For each reference model
        for ref_model in self.reference_models_list:
            false_ranks_list = []

            for data_generating_model in self.data_generating_models_list:
                results_dict = self.get_model_simulations_results_for_combo(
                    data_generating_model,
                    n_triplets,
                    dg_regularization,
                    candidate_regularization,
                )

                ranks_for_ref_model = []

                for sim_idx in self.simulations_list:
                    if sim_idx in results_dict and len(results_dict[sim_idx]) > 0:
                        sim_results = results_dict[sim_idx]
                        # Sort models by accuracy (highest first)
                        sorted_models = sorted(
                            sim_results.items(), key=lambda x: x[1], reverse=True
                        )

                        # Find rank of this reference model (1-indexed)
                        for rank, (model, _) in enumerate(sorted_models, 1):
                            if model == ref_model:
                                ranks_for_ref_model.append(rank)
                                break

                if ref_model == data_generating_model:
                    # This is the correct case
                    if ranks_for_ref_model:
                        correct_ranks_dict[data_generating_model] = np.mean(
                            ranks_for_ref_model
                        )
                        correct_ranks_dict_std[data_generating_model] = (
                            np.std(ranks_for_ref_model, ddof=1)
                            / np.sqrt(len(ranks_for_ref_model))
                            if len(ranks_for_ref_model) > 1
                            else 0
                        )
                else:
                    # This is the false case
                    if ranks_for_ref_model:
                        false_ranks_list.extend(ranks_for_ref_model)

            # Calculate false ranks for this reference model
            if false_ranks_list:
                false_ranks_dict[ref_model] = np.mean(false_ranks_list)
                false_ranks_dict_std[ref_model] = (
                    np.std(false_ranks_list, ddof=1) / np.sqrt(len(false_ranks_list))
                    if len(false_ranks_list) > 1
                    else 0
                )

        return (
            correct_ranks_dict,
            false_ranks_dict,
            correct_ranks_dict_std,
            false_ranks_dict_std,
        )

    def create_ranking_analysis_figure(
        self,
        n_triplets: int,
        dg_regularization: str,
        candidate_regularization: str,
        width_cm: float = 13.97,
        height_cm: float = 5,
    ):
        """
        Create supplementary ranking analysis figure.

        This figure shows mean model rankings when correctly vs incorrectly recovered,
        helping to understand which models are consistently ranked highly even when
        they are not the correct data-generating model.

        Args:
            n_triplets: Number of training triplets to analyze
            dg_regularization: Data-generating model regularization type
            candidate_regularization: Candidate model regularization type
            width_cm: Figure width in centimeters
            height_cm: Figure height in centimeters
        """
        # Define gridspec layout (same as original Figure 3)
        gridspec_dict = {
            "nrows": 3,
            "ncols": 5,
            "height_ratios": [0.5, 4, 0.5],
            "width_ratios": [2, 4, 3, 4, 1],
        }

        subplots_dict = {
            "mean_rank_correct": {"height_index": 1, "width_index": 1},
            "mean_rank_opposite": {"height_index": 1, "width_index": 3},
        }

        # Save figure directly in the main plot folder
        os.makedirs(self.save_plot_folder, exist_ok=True)
        save_path = f"{self.save_plot_folder}/supplementary_ranking_analysis.pdf"

        fig, ax = self.create_figure_layout(
            width_cm, height_cm, gridspec_dict, subplots_dict
        )

        # Calculate mean ranks
        (
            correct_ranks_dict,
            false_ranks_dict,
            correct_ranks_dict_std,
            false_ranks_dict_std,
        ) = self.calculate_models_mean_rank_for_combo(
            n_triplets, dg_regularization, candidate_regularization
        )

        # Invert ranks so higher values are better
        correct_ranks_dict = {
            k: len(self.reference_models_list) - v + 1
            for k, v in correct_ranks_dict.items()
        }
        false_ranks_dict = {
            k: len(self.reference_models_list) - v + 1
            for k, v in false_ranks_dict.items()
        }

        # Ensure both subplots start at the same position
        for ax_name in ax.keys():
            ax[ax_name].set_anchor("W")  # Anchor to the west (left) side

        # Create model ordering based on color dictionary (same as original Figures_1_and_3.py)
        if self.models_order_as_colors_dict is not None:
            models_order_as_colors_dict = list(self.models_order_as_colors_dict)
        elif self.models_color_dict is not None:
            models_order_as_colors_dict = list(self.models_color_dict.keys())
        else:
            models_order_as_colors_dict = list(self.data_generating_models_list)

        # Draw bar plots
        for ax_name in ax.keys():
            if "correct" in ax_name:
                self.draw_bar_plot(
                    ax[ax_name],
                    y_data=correct_ranks_dict,
                    x_tick_labels_ordered_no_formal=models_order_as_colors_dict,
                    error_data=correct_ranks_dict_std,
                    show_error_bars=True,
                    error_capsize=3,
                    error_lw=0.8,
                    error_color="black",
                )
                ## Add panel letter
                ax[ax_name].text(x=-9.5, y=20.5, s="A", fontsize=8, fontweight="bold")

            elif "opposite" in ax_name:
                self.draw_bar_plot(
                    ax[ax_name],
                    y_data=false_ranks_dict,
                    x_tick_labels_ordered_no_formal=models_order_as_colors_dict,
                    error_data=false_ranks_dict_std,
                    show_error_bars=True,
                    error_capsize=3,
                    error_lw=0.8,
                    error_color="black",
                )
                ## Add panel letter
                ax[ax_name].text(x=-10, y=20.5, s="B", fontsize=8, fontweight="bold")

            # Set up ticks and labels (same as original)
            ticks = np.linspace(
                1, len(self.reference_models_list), len(self.reference_models_list)
            )
            ax[ax_name].set_xticks(ticks)
            tick_labels = [""] * len(ticks)
            tick_labels[0] = "ranked last"
            tick_labels[int((len(tick_labels) - 1) / 2)] = "chance\nlevel"
            tick_labels[-1] = "ranked first"
            ax[ax_name].set_xticklabels(tick_labels, fontsize=TICK_LABEL_FONTSIZE)
            ax[ax_name].tick_params(
                axis="both", length=1, width=0.5, labelsize=TICK_LABEL_FONTSIZE
            )
            ax[ax_name].axvline(
                x=len(self.reference_models_list) / 2,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
            )
            ax[ax_name].set_xlim(0, len(self.reference_models_list))

            # Show frame and spines
            ax[ax_name].set_frame_on(True)
            ax[ax_name].spines["top"].set_visible(False)
            ax[ax_name].spines["right"].set_visible(False)
            ax[ax_name].spines["bottom"].set_visible(True)
            ax[ax_name].spines["left"].set_visible(True)
            ax[ax_name].spines["bottom"].set_linewidth(0.5)
            ax[ax_name].spines["left"].set_linewidth(0.5)

        plt.savefig(save_path, dpi=600, format="pdf")
        plt.close()
        print(f"✅ Supplementary ranking analysis figure saved to: {save_path}")

    # ========== ARCHITECTURE-BASED ANALYSIS ==========

    def get_architecture_group_recovery_accuracy(
        self,
        n_triplets: int,
        dg_regularization: str,
        candidate_regularization: str,
        architecture: str,
    ) -> float:
        """Calculate model recovery accuracy for a specific architecture group."""
        if architecture not in self.models_by_architecture:
            return 0.0

        architecture_models = self.models_by_architecture[architecture]
        total_correct = 0
        total_simulations = 0

        for data_generating_model in architecture_models:
            results_dict = self.get_model_simulations_results_for_combo(
                data_generating_model,
                n_triplets,
                dg_regularization,
                candidate_regularization,
            )

            for sim_idx in self.simulations_list:
                if sim_idx in results_dict and len(results_dict[sim_idx]) > 0:
                    sim_results = results_dict[sim_idx]
                    best_model = max(sim_results, key=sim_results.get)
                    if best_model == data_generating_model:
                        total_correct += 1
                    total_simulations += 1

        return total_correct / total_simulations if total_simulations > 0 else 0.0

    def get_architecture_recovery_accuracy(
        self,
        n_triplets: int,
        dg_regularization: str,
        candidate_regularization: str,
        architecture: str,
    ) -> float:
        """Wrapper for get_architecture_group_recovery_accuracy for compatibility."""
        return self.get_architecture_group_recovery_accuracy(
            n_triplets, dg_regularization, candidate_regularization, architecture
        )

    def get_objective_group_recovery_accuracy(
        self,
        n_triplets: int,
        dg_regularization: str,
        candidate_regularization: str,
        objective: str,
    ) -> float:
        """Calculate model recovery accuracy for a specific objective group."""
        if objective not in self.models_by_objective:
            return 0.0

        objective_models = self.models_by_objective[objective]
        total_correct = 0
        total_simulations = 0

        for data_generating_model in objective_models:
            results_dict = self.get_model_simulations_results_for_combo(
                data_generating_model,
                n_triplets,
                dg_regularization,
                candidate_regularization,
            )

            for sim_idx in self.simulations_list:
                if sim_idx in results_dict and len(results_dict[sim_idx]) > 0:
                    sim_results = results_dict[sim_idx]
                    best_model = max(sim_results, key=sim_results.get)
                    if best_model == data_generating_model:
                        total_correct += 1
                    total_simulations += 1

        return total_correct / total_simulations if total_simulations > 0 else 0.0

    def get_objective_recovery_accuracy(
        self,
        n_triplets: int,
        dg_regularization: str,
        candidate_regularization: str,
        objective: str,
    ) -> float:
        """Wrapper for get_objective_group_recovery_accuracy for compatibility."""
        return self.get_objective_group_recovery_accuracy(
            n_triplets, dg_regularization, candidate_regularization, objective
        )

    def get_team_recovery_accuracy_based_on_objectives(
        self, n_triplets: int, dg_regularization: str, candidate_regularization: str
    ) -> float:
        """
        Calculate team recovery accuracy using objective-based grouping.

        Team recovery succeeds when the best-performing model has the same training
        objective as the data-generating model, even if it's not the exact same model.

        For example, if a Supervised model generated the data, recovery is successful
        if ANY Supervised model achieves the highest test accuracy (not just the exact
        model that generated the data).

        Parameters:
        ----------
        n_triplets : int
            Number of training triplets

        dg_regularization : str
            Data-generating model regularization

        candidate_regularization : str
            Candidate model regularization

        Returns:
        -------
        float
            Team recovery accuracy (proportion of successful team recoveries from 0.0 to 1.0)

        Notes:
        -----
        Training objectives are:
        - 'Supervised': Models trained with labeled supervision
        - 'Self-supervised': Models trained without labels (e.g., SimCLR, MoCo)
        - 'Image & Text Alignment': Multimodal models (e.g., CLIP, ALIGN)
        """
        total_correct = 0
        total_simulations = 0

        for data_generating_model in self.data_generating_models_list:
            dg_objective = self.models_objective.get(data_generating_model)
            if dg_objective is None:
                continue

            results_dict = self.get_model_simulations_results_for_combo(
                data_generating_model,
                n_triplets,
                dg_regularization,
                candidate_regularization,
            )

            for sim_idx in self.simulations_list:
                if sim_idx in results_dict and len(results_dict[sim_idx]) > 0:
                    sim_results = results_dict[sim_idx]
                    best_model = max(sim_results, key=sim_results.get)
                    best_objective = self.models_objective.get(best_model)

                    if best_objective == dg_objective:
                        total_correct += 1
                    total_simulations += 1

        return total_correct / total_simulations if total_simulations > 0 else 0.0

    def get_team_recovery_accuracy_based_on_architectures(
        self, n_triplets: int, dg_regularization: str, candidate_regularization: str
    ) -> float:
        """
        Calculate team recovery accuracy using architecture-based grouping.

        Team recovery succeeds when the best-performing model has the same architecture
        type as the data-generating model, even if it's not the exact same model.

        For example, if a CNN generated the data, recovery is successful if ANY CNN
        achieves the highest test accuracy (not just the exact CNN that generated the data).

        Parameters:
        ----------
        n_triplets : int
            Number of training triplets

        dg_regularization : str
            Data-generating model regularization

        candidate_regularization : str
            Candidate model regularization

        Returns:
        -------
        float
            Team recovery accuracy (proportion of successful team recoveries from 0.0 to 1.0)

        Notes:
        -----
        Architecture types are:
        - 'CNN': Convolutional Neural Networks (e.g., ResNet, VGG, EfficientNet)
        - 'ViT': Vision Transformers (e.g., ViT-Base, ViT-Large, DeiT)
        """
        total_correct = 0
        total_simulations = 0

        for data_generating_model in self.data_generating_models_list:
            dg_architecture = self.models_architecture.get(data_generating_model)
            if dg_architecture is None:
                continue

            results_dict = self.get_model_simulations_results_for_combo(
                data_generating_model,
                n_triplets,
                dg_regularization,
                candidate_regularization,
            )

            for sim_idx in self.simulations_list:
                if sim_idx in results_dict and len(results_dict[sim_idx]) > 0:
                    sim_results = results_dict[sim_idx]
                    best_model = max(sim_results, key=sim_results.get)
                    best_architecture = self.models_architecture.get(best_model)

                    if best_architecture == dg_architecture:
                        total_correct += 1
                    total_simulations += 1

        return total_correct / total_simulations if total_simulations > 0 else 0.0

    def _get_team_objective_binomial_error(
        self,
        n_triplets: int,
        dg_regularization: str,
        candidate_regularization: str,
        CI: bool = False,
    ) -> Union[float, Tuple[float, float]]:
        """Calculate binomial confidence interval or standard error for objective-based team recovery."""
        from scipy import stats

        total_correct = 0
        total_simulations = 0

        for data_generating_model in self.data_generating_models_list:
            dg_objective = self.models_objective.get(data_generating_model)
            if dg_objective is None:
                continue

            results_dict = self.get_model_simulations_results_for_combo(
                data_generating_model,
                n_triplets,
                dg_regularization,
                candidate_regularization,
            )

            for sim_idx in self.simulations_list:
                if sim_idx in results_dict and len(results_dict[sim_idx]) > 0:
                    sim_results = results_dict[sim_idx]
                    best_model = max(sim_results, key=sim_results.get)
                    best_objective = self.models_objective.get(best_model)

                    if best_objective == dg_objective:
                        total_correct += 1
                    total_simulations += 1

        if total_simulations == 0:
            return (0.0, 0.0) if CI else 0.0

        if CI:
            # Use binomial confidence interval for CI=True
            ci_lower, ci_upper = stats.binom.interval(
                0.95, total_simulations, total_correct / total_simulations
            )
            return (ci_lower / total_simulations, ci_upper / total_simulations)
        else:
            # Use standard error for CI=False
            p = total_correct / total_simulations
            se = (p * (1 - p) / total_simulations) ** 0.5
            return se

    def _get_team_architecture_binomial_error(
        self,
        n_triplets: int,
        dg_regularization: str,
        candidate_regularization: str,
        CI: bool = False,
    ) -> Union[float, Tuple[float, float]]:
        """Calculate binomial confidence interval or standard error for architecture-based team recovery."""
        from scipy import stats

        total_correct = 0
        total_simulations = 0

        for data_generating_model in self.data_generating_models_list:
            dg_architecture = self.models_architecture.get(data_generating_model)
            if dg_architecture is None:
                continue

            results_dict = self.get_model_simulations_results_for_combo(
                data_generating_model,
                n_triplets,
                dg_regularization,
                candidate_regularization,
            )

            for sim_idx in self.simulations_list:
                if sim_idx in results_dict and len(results_dict[sim_idx]) > 0:
                    sim_results = results_dict[sim_idx]
                    best_model = max(sim_results, key=sim_results.get)
                    best_architecture = self.models_architecture.get(best_model)

                    if best_architecture == dg_architecture:
                        total_correct += 1
                    total_simulations += 1

        if total_simulations == 0:
            return (0.0, 0.0) if CI else 0.0

        if CI:
            # Use binomial confidence interval for CI=True
            ci_lower, ci_upper = stats.binom.interval(
                0.95, total_simulations, total_correct / total_simulations
            )
            return (ci_lower / total_simulations, ci_upper / total_simulations)
        else:
            # Use standard error for CI=False
            p = total_correct / total_simulations
            se = (p * (1 - p) / total_simulations) ** 0.5
            return se

    def create_team_recovery_line_plots(
        self,
        ax,
        n_triplets_list: list,
        dg_regularization: str,
        candidate_regularization: str,
        show_error_bars: bool = True,
        show_x_label: bool = True,
        show_y_label: bool = True,
        panel_letter: Optional[str] = None,
        CI: bool = True,
        show_exact_recovery: bool = True,
        confusion_matrix_n_triplets: Optional[Dict[str, int]] = None,
    ):
        """
        Create line plots showing team recovery for both architecture and objective groupings with confidence intervals.

        Args:
            ax: Matplotlib axes object
            n_triplets_list: List of n_triplets values to plot
            dg_regularization: Data generating model regularization
            candidate_regularization: Candidate model regularization
            show_error_bars: Whether to show confidence intervals
            show_x_label: Whether to show x-axis label
            show_y_label: Whether to show y-label
            panel_letter: Panel letter (e.g., 'A') to display
            CI: Whether to use confidence intervals (True) or standard error (False)
            show_exact_recovery: Whether to show the exact model recovery line
            confusion_matrix_n_triplets: Dict mapping size ('small', 'medium', 'large') to n_triplets values
                                        for annotating confusion matrix points with letters
        """
        # Calculate team recovery for objectives and architectures
        obj_accuracies = []
        arch_accuracies = []
        overall_accuracies = []
        obj_errors = []
        arch_errors = []
        overall_errors = []

        for n_triplets in n_triplets_list:
            obj_acc = self.get_team_recovery_accuracy_based_on_objectives(
                n_triplets, dg_regularization, candidate_regularization
            )
            arch_acc = self.get_team_recovery_accuracy_based_on_architectures(
                n_triplets, dg_regularization, candidate_regularization
            )

            obj_accuracies.append(obj_acc)
            arch_accuracies.append(arch_acc)

            if show_exact_recovery:
                overall_acc = self.get_total_model_recovery_accuracy_for_combo(
                    n_triplets, dg_regularization, candidate_regularization
                )
                overall_accuracies.append(overall_acc)

            if show_error_bars:
                # Get errors for each metric
                if show_exact_recovery:
                    overall_err = self.get_combo_recovery_binomial_error(
                        n_triplets, dg_regularization, candidate_regularization, CI=CI
                    )
                    overall_errors.append(overall_err)

                # For team metrics, calculate binomial CI similarly
                obj_err = self._get_team_objective_binomial_error(
                    n_triplets, dg_regularization, candidate_regularization, CI=CI
                )
                arch_err = self._get_team_architecture_binomial_error(
                    n_triplets, dg_regularization, candidate_regularization, CI=CI
                )

                obj_errors.append(obj_err)
                arch_errors.append(arch_err)

        # Format x-axis labels
        x_labels = []
        for n in n_triplets_list:
            if n >= 1_000_000:
                x_labels.append(f"{n/1_000_000:.1f}M")
            elif n >= 1000:
                x_labels.append(f"{n/1000:.1f}K")
            else:
                x_labels.append(str(n))

        # Plot lines
        if show_exact_recovery:
            (line1,) = ax.plot(
                range(len(n_triplets_list)),
                overall_accuracies,
                "o-",
                label="Exact Model Recovery",
                color="#1f77b4",
                markersize=2,
                linewidth=1.5,
            )

        (line2,) = ax.plot(
            range(len(n_triplets_list)),
            obj_accuracies,
            "o-",
            label="objective family recovery",
            color="#2ca02c",
            markersize=2,
            linewidth=1.5,
        )
        (line3,) = ax.plot(
            range(len(n_triplets_list)),
            arch_accuracies,
            "o-",
            label="architecture family recovery",
            color="#1f77b4",
            markersize=2,
            linewidth=1.5,
        )  # Changed to blue

        # Add confidence interval bands if requested
        if show_error_bars:
            if show_exact_recovery and overall_errors:
                if CI:
                    # Extract lower and upper bounds from tuples
                    overall_lower = [err[0] for err in overall_errors]
                    overall_upper = [err[1] for err in overall_errors]
                else:
                    # Use standard error
                    overall_lower = np.array(overall_accuracies) - np.array(
                        overall_errors
                    )
                    overall_upper = np.array(overall_accuracies) + np.array(
                        overall_errors
                    )

                # Ensure bounds are within [0, 1]
                overall_lower = np.maximum(overall_lower, 0)
                overall_upper = np.minimum(overall_upper, 1)

                # Add confidence band
                ax.fill_between(
                    range(len(x_labels)),
                    overall_lower,
                    overall_upper,
                    alpha=0.2,
                    color=line1.get_color(),
                    linewidth=0,
                )

            if obj_errors:
                if CI:
                    obj_lower = [err[0] for err in obj_errors]
                    obj_upper = [err[1] for err in obj_errors]
                else:
                    obj_lower = np.array(obj_accuracies) - np.array(obj_errors)
                    obj_upper = np.array(obj_accuracies) + np.array(obj_errors)

                obj_lower = np.maximum(obj_lower, 0)
                obj_upper = np.minimum(obj_upper, 1)
                ax.fill_between(
                    range(len(x_labels)),
                    obj_lower,
                    obj_upper,
                    alpha=0.2,
                    color=line2.get_color(),
                    linewidth=0,
                )

            if arch_errors:
                if CI:
                    arch_lower = [err[0] for err in arch_errors]
                    arch_upper = [err[1] for err in arch_errors]
                else:
                    arch_lower = np.array(arch_accuracies) - np.array(arch_errors)
                    arch_upper = np.array(arch_accuracies) + np.array(arch_errors)

                arch_lower = np.maximum(arch_lower, 0)
                arch_upper = np.minimum(arch_upper, 1)
                ax.fill_between(
                    range(len(x_labels)),
                    arch_lower,
                    arch_upper,
                    alpha=0.2,
                    color=line3.get_color(),
                    linewidth=0,
                )

        # Add confusion matrix point annotations if provided
        if confusion_matrix_n_triplets is not None:
            # Map size to letter for objective confusion matrices
            obj_letters = {"small": "B", "medium": "C", "large": "D"}
            # Map size to letter for architecture confusion matrices
            arch_letters = {"small": "E", "medium": "F", "large": "G"}

            for size, n_triplets in confusion_matrix_n_triplets.items():
                if n_triplets in n_triplets_list:
                    idx = n_triplets_list.index(n_triplets)

                    # Position B,C below, D above
                    if size in ["small", "medium"]:  # B, C
                        obj_va = "top"
                        obj_offset = -0.03
                    else:  # D (large)
                        obj_va = "bottom"
                        obj_offset = 0.03

                    # Position E,F above, G below
                    if size in ["small", "medium"]:  # E, F
                        arch_va = "bottom"
                        arch_offset = 0.03
                    else:  # G (large)
                        arch_va = "top"
                        arch_offset = -0.03

                    # Add dotted line from objective point to letter
                    ax.plot(
                        [idx, idx],
                        [obj_accuracies[idx], obj_accuracies[idx] + obj_offset],
                        "k:",
                        linewidth=0.5,
                        alpha=0.5,
                    )

                    # Add letter for objective point (black color, smaller font)
                    ax.text(
                        idx,
                        obj_accuracies[idx] + obj_offset,
                        obj_letters[size],
                        ha="center",
                        va=obj_va,
                        fontsize=LINE_ANNOTATION_FONTSIZE,
                        fontweight=PANEL_LETTER_FONTWEIGHT,
                        color="black",
                    )

                    # Add dotted line from architecture point to letter
                    ax.plot(
                        [idx, idx],
                        [arch_accuracies[idx], arch_accuracies[idx] + arch_offset],
                        "k:",
                        linewidth=0.5,
                        alpha=0.5,
                    )

                    # Add letter for architecture point (black color, smaller font)
                    ax.text(
                        idx,
                        arch_accuracies[idx] + arch_offset,
                        arch_letters[size],
                        ha="center",
                        va=arch_va,
                        fontsize=LINE_ANNOTATION_FONTSIZE,
                        fontweight=PANEL_LETTER_FONTWEIGHT,
                        color="black",
                    )

        # Formatting
        if show_x_label:
            ax.set_xlabel(
                "number of training triplets",
                fontsize=AXIS_LABEL_FONTSIZE,
                fontweight=AXIS_LABEL_FONTWEIGHT,
            )
        if show_y_label:
            ax.set_ylabel(
                "recovery accuracy",
                fontsize=AXIS_LABEL_FONTSIZE,
                fontweight=AXIS_LABEL_FONTWEIGHT,
            )

        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, len(n_triplets_list) - 0.5)
        ax.set_xticks(range(len(n_triplets_list)))
        ax.set_xticklabels(x_labels, fontsize=TICK_LABEL_FONTSIZE)
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.set_yticklabels(
            [f"{x:.1f}" for x in np.linspace(0, 1, 11)], fontsize=TICK_LABEL_FONTSIZE
        )
        ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=TICK_LABEL_FONTSIZE,
            width=0.5,
            length=3,
        )

        if panel_letter is not None:
            ax.text(
                -0.07,
                1.03,
                panel_letter,
                transform=ax.transAxes,
                fontsize=PANEL_LETTER_FONTSIZE,
                fontweight=PANEL_LETTER_FONTWEIGHT,
            )

    def create_team_objective_confusion_matrix(
        self,
        ax,
        n_triplets: int,
        dg_regularization: str,
        candidate_regularization: str,
        show_x_label: bool = True,
        show_y_label: bool = True,
        show_y_ticks: bool = True,
        show_colorbar: bool = True,
        colorbar_ax=None,
        panel_letter: Optional[str] = None,
    ):
        """Create confusion matrix for objective-based team recovery."""
        objectives = list(self.objective_groups.keys())
        confusion_matrix = pd.DataFrame(
            index=objectives, columns=objectives, dtype=int
        ).fillna(0)

        for dg_obj in objectives:
            for sim_idx in self.simulations_list:
                for dg_model in self.models_by_objective[dg_obj]:
                    results_dict = self.get_model_simulations_results_for_combo(
                        dg_model,
                        n_triplets,
                        dg_regularization,
                        candidate_regularization,
                    )

                    if sim_idx in results_dict and len(results_dict[sim_idx]) > 0:
                        sim_results = results_dict[sim_idx]
                        best_model = max(sim_results, key=sim_results.get)
                        best_obj = self.models_objective.get(best_model)

                        if best_obj in objectives:
                            confusion_matrix.loc[dg_obj, best_obj] += 1

        # Ensure all values are integers
        confusion_matrix = confusion_matrix.astype(int)

        # Create custom colormap
        n_colors = 256
        colors_base = plt.cm.get_cmap("Greens")(np.linspace(0, 1, n_colors))
        zero_color = [0.95, 0.95, 1, 1]
        colors_base[0] = zero_color
        colors_base[1:] = plt.cm.get_cmap("Greens")(np.linspace(0.1, 1, n_colors - 1))
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_Greens", colors_base, N=n_colors
        )

        vmax = int(confusion_matrix.to_numpy().max())

        # Create heatmap
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap=custom_cmap,
            vmin=0,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            linecolor="gray",
            cbar=show_colorbar,
            cbar_ax=colorbar_ax,
            cbar_kws=(
                {"shrink": 0.6, "aspect": 50, "format": "%d"} if show_colorbar else None
            ),
            annot_kws={"size": AXIS_LABEL_FONTSIZE},
            ax=ax,
        )

        if show_x_label:
            ax.set_xlabel(
                "recovered objective",
                fontsize=AXIS_LABEL_FONTSIZE,
                fontweight=AXIS_LABEL_FONTWEIGHT,
            )
        else:
            ax.set_xlabel("")

        if show_y_label:
            ax.set_ylabel(
                "data-generating objective",
                fontsize=AXIS_LABEL_FONTSIZE,
                fontweight=AXIS_LABEL_FONTWEIGHT,
            )
        else:
            ax.set_ylabel("")

        # Lowercase objectives and format tick labels: split "image & text alignment" into two lines
        objectives_lower = self._lowercase_strings(objectives)
        x_tick_labels = [
            obj.replace("image & text alignment", "image & text\nalignment")
            for obj in objectives_lower
        ]
        y_tick_labels = [
            obj.replace("image & text alignment", "image & text\nalignment  ")
            for obj in objectives_lower
        ]
        x_tick_labels = [
            obj.replace("self-supervised", "self\nsupervised") for obj in x_tick_labels
        ]
        y_tick_labels = [
            obj.replace("self-supervised", "self      \nsupervised")
            for obj in y_tick_labels
        ]

        ax.set_xticklabels(
            x_tick_labels, rotation=0, ha="center", fontsize=TICK_LABEL_FONTSIZE
        )

        if show_y_ticks:
            ax.set_yticklabels(y_tick_labels, rotation=0, fontsize=TICK_LABEL_FONTSIZE)
        else:
            ax.set_yticklabels([])

        if panel_letter is not None:
            ax.text(
                -0.1,
                1.1,
                panel_letter,
                transform=ax.transAxes,
                fontsize=PANEL_LETTER_FONTSIZE,
                fontweight=PANEL_LETTER_FONTWEIGHT,
            )

        if show_colorbar and colorbar_ax is not None:
            colorbar_ax.set_ylabel(
                "number of simulations", fontsize=COLORBAR_LABEL_FONTSIZE, labelpad=1
            )
            colorbar_ax.yaxis.set_label_position("left")
            colorbar_ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    def create_team_architecture_confusion_matrix(
        self,
        ax,
        n_triplets: int,
        dg_regularization: str,
        candidate_regularization: str,
        show_x_label: bool = True,
        show_y_label: bool = True,
        show_y_ticks: bool = True,
        show_colorbar: bool = True,
        colorbar_ax=None,
        panel_letter: Optional[str] = None,
    ):
        """Create confusion matrix for architecture-based team recovery."""
        architectures = list(self.architecture_groups.keys())
        confusion_matrix = pd.DataFrame(
            index=architectures, columns=architectures, dtype=int
        ).fillna(0)

        for dg_arch in architectures:
            for sim_idx in self.simulations_list:
                for dg_model in self.models_by_architecture[dg_arch]:
                    results_dict = self.get_model_simulations_results_for_combo(
                        dg_model,
                        n_triplets,
                        dg_regularization,
                        candidate_regularization,
                    )

                    if sim_idx in results_dict and len(results_dict[sim_idx]) > 0:
                        sim_results = results_dict[sim_idx]
                        best_model = max(sim_results, key=sim_results.get)
                        best_arch = self.models_architecture.get(best_model)

                        if best_arch in architectures:
                            confusion_matrix.loc[dg_arch, best_arch] += 1

        # Ensure all values are integers
        confusion_matrix = confusion_matrix.astype(int)

        # Create custom colormap
        n_colors = 256
        colors_base = plt.cm.get_cmap("Blues")(np.linspace(0, 1, n_colors))
        zero_color = [0.95, 0.95, 1, 1]
        colors_base[0] = zero_color
        colors_base[1:] = plt.cm.get_cmap("Blues")(np.linspace(0.1, 1, n_colors - 1))
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_Blues", colors_base, N=n_colors
        )

        vmax = int(confusion_matrix.to_numpy().max())

        # Create heatmap
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap=custom_cmap,
            vmin=0,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            linecolor="gray",
            cbar=show_colorbar,
            cbar_ax=colorbar_ax,
            cbar_kws=(
                {"shrink": 0.6, "aspect": 50, "format": "%d"} if show_colorbar else None
            ),
            annot_kws={"size": AXIS_LABEL_FONTSIZE},
            ax=ax,
        )

        if show_x_label:
            ax.set_xlabel(
                "recovered architecture",
                fontsize=AXIS_LABEL_FONTSIZE,
                fontweight=AXIS_LABEL_FONTWEIGHT,
            )
        else:
            ax.set_xlabel("")

        if show_y_label:
            ax.set_ylabel(
                "data-generating architecture",
                fontsize=AXIS_LABEL_FONTSIZE,
                fontweight=AXIS_LABEL_FONTWEIGHT,
            )
        else:
            ax.set_ylabel("")

        ax.set_xticklabels(
            architectures, rotation=0, ha="center", fontsize=TICK_LABEL_FONTSIZE
        )

        if show_y_ticks:
            ax.set_yticklabels(architectures, rotation=0, fontsize=TICK_LABEL_FONTSIZE)
        else:
            ax.set_yticklabels([])

        if panel_letter is not None:
            ax.text(
                -0.1,
                1.1,
                panel_letter,
                transform=ax.transAxes,
                fontsize=PANEL_LETTER_FONTSIZE,
                fontweight=PANEL_LETTER_FONTWEIGHT,
            )

        if show_colorbar and colorbar_ax is not None:
            colorbar_ax.set_ylabel(
                "number of simulations", fontsize=COLORBAR_LABEL_FONTSIZE, labelpad=1
            )
            colorbar_ax.yaxis.set_label_position("left")
            colorbar_ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    def draw_bar_plot(
        self,
        ax,
        y_data: Union[list, pd.DataFrame, Dict],
        x_tick_labels_ordered_no_formal: Optional[list] = None,
        horizontal: bool = True,
        show_ticks_labels: bool = True,
        show_ticks_y: bool = False,
        zero_pad: bool = True,
        tick_label_size: int = 5,
        error_data: Optional[Dict] = None,
        show_error_bars: bool = True,
        error_capsize: int = 3,
        error_color: Optional[str] = None,
        error_lw: float = 1.0,
    ):
        """
        Creates a customized bar plot with extensive formatting options.

        This method creates publication-quality bar plots with support for horizontal or
        vertical orientation, error bars, custom colors per bar, and numerous visualization
        settings. The method intelligently handles different input data types (lists,
        DataFrames, dictionaries) and applies model-specific colors from the color dictionary.
        """
        if isinstance(y_data, pd.DataFrame):
            if x_tick_labels_ordered_no_formal is None:
                x_tick_labels_ordered_no_formal = y_data.index.tolist()[::-1]
                x_formal = self.go_formal(x_tick_labels_ordered_no_formal)
                y = y_data.values.tolist()
            else:
                y = []
                for model in x_tick_labels_ordered_no_formal:
                    y.append(y_data.loc[model])
                x_formal = self.go_formal(list(x_tick_labels_ordered_no_formal))
        elif isinstance(y_data, Dict):
            if x_tick_labels_ordered_no_formal is None:
                x_tick_labels_ordered_no_formal = list(y_data.keys())
                x_tick_labels_ordered_no_formal = x_tick_labels_ordered_no_formal[::-1]
                x_formal = self.go_formal(x_tick_labels_ordered_no_formal)
                y = list(y_data.values())[::-1]
            else:
                y = []
                for model in x_tick_labels_ordered_no_formal:
                    y.append(y_data[model])
                x_formal = self.go_formal(list(x_tick_labels_ordered_no_formal))
        elif isinstance(y_data, list):
            assert (
                x_tick_labels_ordered_no_formal is not None
            ), "x_tick_labels_ordered_no_formal is required when y_data is a list"
            x_formal = self.go_formal(x_tick_labels_ordered_no_formal)
            y = y_data[::-1]

        # Create error bars if provided
        yerr = None
        if show_error_bars and error_data is not None:
            yerr = [
                0 if model not in error_data else error_data[model]
                for model in x_tick_labels_ordered_no_formal
            ]
            yerr = (
                yerr[::-1]
                if isinstance(y_data, (pd.DataFrame, Dict))
                and x_tick_labels_ordered_no_formal is None
                else yerr
            )

        if horizontal:
            bars = ax.barh(
                x_formal,
                y,
                left=0,
                xerr=yerr,
                capsize=error_capsize,
                error_kw={
                    "linewidth": error_lw,
                    "capthick": error_lw,
                    "color": error_color,
                },
            )
            tick_labels = ax.get_yticklabels()
        else:
            bars = ax.bar(
                x_formal,
                y,
                yerr=yerr,
                capsize=error_capsize,
                error_kw={
                    "linewidth": error_lw,
                    "capthick": error_lw,
                    "color": error_color,
                },
            )
            tick_labels = ax.get_xticklabels()

        if self.models_color_dict is not None:
            for bar, model_name, tick_label, value in zip(
                bars, x_tick_labels_ordered_no_formal, tick_labels, y
            ):
                if model_name == "Google_ViT_Large_224":
                    color = self.models_color_dict.get("Google_ViT_Large", "#000000")
                else:
                    color = self.models_color_dict.get(model_name, "#000000")

                if value == 0:
                    bar.set_alpha(0)  # Make zero-value bars completely transparent
                else:
                    bar.set_color(color)
                tick_label.set_color(color)
                tick_label.set_fontsize(
                    tick_label_size
                    if tick_label_size is not None
                    else TICK_LABEL_FONTSIZE
                )

        if not show_ticks_labels:
            if horizontal:
                ax.set_yticks([])
            else:
                ax.set_xticks([])

        if show_ticks_y:
            if horizontal:
                ax.set_xticks(np.linspace(0, len(self.reference_models_list) - 1, 5))
                ax.tick_params(axis="x", labelsize=tick_label_size)
                ax.set_xticklabels(
                    [
                        f"{x:.1f}"
                        for x in np.linspace(0, len(self.reference_models_list) - 1, 5)
                    ]
                )
            else:
                ax.set_yticks(np.linspace(0, len(self.reference_models_list) - 1, 5))
                ax.tick_params(axis="y", labelsize=tick_label_size)
                ax.set_yticklabels(
                    [
                        f"{y:.1f}"
                        for y in np.linspace(0, len(self.reference_models_list) - 1, 5)
                    ]
                )
        else:
            if horizontal:
                ax.set_xticks([])
            else:
                ax.set_yticks([])

        if zero_pad:
            if horizontal:
                ax.set_xlim(0, ax.get_xlim()[1])
            else:
                ax.set_ylim(0, ax.get_ylim()[1])

        return bars


def create_figure_S4_and_S5(
    db_path: str,
    data_generating_models_list: list,
    n_triplets_list: list,
    formal_names_dict: dict,
    models_color_dict: dict,
    models_architecture: dict,
    models_objective: dict,
    simulations_list: Optional[list] = None,
    save_plot_folder: str = "Plots/new_objective_analysis_beta",
):
    """
    Create Figures S4 and S5 for team-based model recovery analysis.

    This function generates publication-quality figures showing how well models can be
    recovered when grouped by architecture or training objective.

    Figures Created:
    ---------------
    - Figure S4: Model recovery accuracy vs number of training triplets
                 Line plot with confusion matrices at small, medium, and large training sizes

    - Figure S5: Combined team recovery analysis
                 Line plots and confusion matrices for architecture-based and objective-based groupings

    - Supplementary ranking figure: Mean model rankings when correctly vs incorrectly recovered

    Parameters:
    ----------
    db_path : str
        Path to SQLite database containing simulation results
        Example: "Results/simulations.db"

    data_generating_models_list : list
        List of model names used as data generators in simulations
        Example: ['ResNet50', 'ViT_Base', 'CLIP_Large', ...]

    n_triplets_list : list
        List of training triplet counts to analyze
        Example: [100, 400, 1600, 6400, 25600, ...]

    formal_names_dict : dict
        Mapping from internal model names to publication-ready display names
        Example: {'Google_ViT_Large_224': 'ViT-L/16'}

    models_color_dict : dict
        Color assignments for each model (ensures consistent visualization)
        Example: {'ResNet50': '#1f77b4', 'ViT_Base': '#ff7f0e'}

    models_architecture : dict
        Mapping from model name to architecture type
        Example: {'ResNet50': 'CNN', 'ViT_Base': 'ViT'}

    models_objective : dict
        Mapping from model name to training objective
        Example: {'ResNet50': 'Supervised', 'SimCLR': 'Self-supervised'}

    simulations_list : list, optional
        List of simulation indices to analyze (default: None, uses all available)

    save_plot_folder : str, optional
        Directory where figures and CSV files will be saved
        (default: "Plots/new_objective_analysis_beta")

    Output Files:
    ------------
    PDF Figures:
        - {save_plot_folder}/figure_S4_model_recovery_accuracy.pdf
        - {save_plot_folder}/figure_S5_team_recovery.pdf
        - {save_plot_folder}/supplementary_ranking_analysis.pdf

    CSV Data Files:
        - {save_plot_folder}/analysis_data/figure_S4_accuracy_by_training_size.csv
        - {save_plot_folder}/analysis_data/overall_model_recovery.csv
        - {save_plot_folder}/analysis_data/team_recovery_by_group.csv

    Example:
    -------
    >>> create_figure_S4_and_S5(
    ...     db_path="Results/simulations.db",
    ...     data_generating_models_list=model_names,
    ...     n_triplets_list=[100, 400, 1600, 6400, 25600],
    ...     formal_names_dict=name_mapping,
    ...     models_color_dict=colors,
    ...     models_architecture=arch_dict,
    ...     models_objective=obj_dict,
    ...     save_plot_folder="Plots/output"
    ... )
    """

    # Initialize visualizer with architecture and objective dictionaries
    visualizer = TeamRecoveryVisualizer(
        db_path=db_path,
        data_generating_models_list=data_generating_models_list,
        simulations_list=simulations_list,
        formal_names_dict=formal_names_dict,
        models_color_dict=models_color_dict,
        models_architecture=models_architecture,
        models_objective=models_objective,
        models_order_as_colors_list=list(models_color_dict.keys()),
        save_plot_folder=save_plot_folder,
        n_triplets_list=n_triplets_list,
    )

    # Get the single regularization combination (should only be one now)
    if len(visualizer.regularization_combinations) == 0:
        print("ERROR: No regularization combinations found in database!")
        return
    elif len(visualizer.regularization_combinations) > 1:
        print(
            f"WARNING: Found {len(visualizer.regularization_combinations)} regularization combinations. Using the first one."
        )
        for combo in visualizer.regularization_combinations:
            print(f"  DG: {combo[0]}, Candidate: {combo[1]}")

    dg_reg, candidate_reg = visualizer.regularization_combinations[0]
    print(f"📌 Regularization: Data-Generating={dg_reg}, Candidate={candidate_reg}")

    print(f"\n{'='*80}")
    print(f"📊 CREATING FIGURES S4, S5, AND SUPPLEMENTARY RANKING FIGURE")
    print(f"{'='*80}")

    # Collect data for CSV files
    consolidated_data = {"overall_recovery": [], "team_recovery": []}

    for n_triplets in n_triplets_list:
        # Overall recovery
        overall_acc = visualizer.get_total_model_recovery_accuracy_for_combo(
            n_triplets, dg_reg, candidate_reg
        )
        consolidated_data["overall_recovery"].append(
            {"n_triplets": n_triplets, "model_recovery_accuracy": overall_acc}
        )

        # Team recovery
        team_obj_acc = visualizer.get_team_recovery_accuracy_based_on_objectives(
            n_triplets, dg_reg, candidate_reg
        )
        team_arch_acc = visualizer.get_team_recovery_accuracy_based_on_architectures(
            n_triplets, dg_reg, candidate_reg
        )
        consolidated_data["team_recovery"].extend(
            [
                {
                    "n_triplets": n_triplets,
                    "team_type": "by_objective",
                    "model_recovery_accuracy": team_obj_acc,
                },
                {
                    "n_triplets": n_triplets,
                    "team_type": "by_architecture",
                    "model_recovery_accuracy": team_arch_acc,
                },
            ]
        )

    # Create grouped axes and gridspec for combined figure
    fig_height_cm = 16.8
    fig_width_cm = 13.97
    n_rows = 7
    n_cols = 10
    height_ratios = [0.5, 5.3, 2, 3, 2, 3, 1]
    width_ratios = [1, 0.96, 3, 0.5, 3, 0.5, 3, 0.7, 0.6, 0.7]

    gridspec_dict = {
        "nrows": n_rows,
        "ncols": n_cols,
        "height_ratios": height_ratios,
        "width_ratios": width_ratios,
    }

    subplots_dict = {
        "recovery_n_triplets": {
            "height_index": 1,
            "width_index": [1, 2, 3, 4, 5, 6, 7, 8],
        },
        "conf_obj_small": {
            "height_index": 3,
            "width_index": 2,
        },
        "conf_obj_medium": {
            "height_index": 3,
            "width_index": 4,
        },
        "conf_obj_large": {
            "height_index": 3,
            "width_index": 6,
        },
        "colorbar_obj": {
            "height_index": 3,
            "width_index": 8,
        },
        "conf_arch_small": {
            "height_index": 5,
            "width_index": 2,
        },
        "conf_arch_medium": {
            "height_index": 5,
            "width_index": 4,
        },
        "conf_arch_large": {
            "height_index": 5,
            "width_index": 6,
        },
        "colorbar_arch": {
            "height_index": 5,
            "width_index": 8,
        },
    }

    # Create the figure with all axes
    print("\n📈 Creating Figure S5 (combined team recovery with confusion matrices)...")
    fig, ax = visualizer.create_figure_layout(
        fig_width_cm, fig_height_cm, gridspec_dict, subplots_dict
    )

    # Create team confusion matrices for key n_triplets values
    key_n_triplets = {
        "small": min(n_triplets_list),
        "medium": n_triplets_list[len(n_triplets_list) // 2],
        "large": max(n_triplets_list),
    }

    # Draw team recovery line plot
    visualizer.create_team_recovery_line_plots(
        ax["recovery_n_triplets"],
        n_triplets_list=n_triplets_list,
        dg_regularization=dg_reg,
        candidate_regularization=candidate_reg,
        show_error_bars=True,
        show_x_label=True,
        show_y_label=True,
        panel_letter="A",
        CI=True,
        show_exact_recovery=False,  # Can be set to False to hide exact model recovery
        confusion_matrix_n_triplets=key_n_triplets,  # Pass the n_triplets for annotation
    )

    # Draw objective confusion matrices
    for size, n_triplets in key_n_triplets.items():
        ax_name = f"conf_obj_{size}"
        show_x_label = size == "small"
        show_y_label = size == "small"
        show_y_ticks = size == "small"
        show_colorbar = size == "large"
        colorbar_ax = ax["colorbar_obj"] if size == "large" else None
        panel_letter = {"small": "B", "medium": "C", "large": "D"}[size]

        visualizer.create_team_objective_confusion_matrix(
            ax=ax[ax_name],
            n_triplets=n_triplets,
            dg_regularization=dg_reg,
            candidate_regularization=candidate_reg,
            show_x_label=show_x_label,
            show_y_label=show_y_label,
            show_y_ticks=show_y_ticks,
            show_colorbar=show_colorbar,
            colorbar_ax=colorbar_ax,
            panel_letter=panel_letter,
        )

    # Draw architecture confusion matrices
    for size, n_triplets in key_n_triplets.items():
        ax_name = f"conf_arch_{size}"
        show_x_label = size == "small"
        show_y_label = size == "small"
        show_y_ticks = size == "small"
        show_colorbar = size == "large"
        colorbar_ax = ax["colorbar_arch"] if size == "large" else None
        panel_letter = {"small": "E", "medium": "F", "large": "G"}[size]

        visualizer.create_team_architecture_confusion_matrix(
            ax=ax[ax_name],
            n_triplets=n_triplets,
            dg_regularization=dg_reg,
            candidate_regularization=candidate_reg,
            show_x_label=show_x_label,
            show_y_label=show_y_label,
            show_y_ticks=show_y_ticks,
            show_colorbar=show_colorbar,
            colorbar_ax=colorbar_ax,
            panel_letter=panel_letter,
        )

    # Save Figure S5
    os.makedirs(visualizer.save_plot_folder, exist_ok=True)
    save_path = f"{visualizer.save_plot_folder}/figure_S5.pdf"
    plt.savefig(save_path, dpi=600, format="pdf")
    plt.close()
    print(f"✅ Figure S5 saved to: {save_path}")

    # Create Figure S4 (model recovery accuracy line plot)
    print("\n📈 Creating Figure S4 (model recovery accuracy vs training triplets)...")
    visualizer.create_figure_S4(
        n_triplets_list=n_triplets_list,
        dg_regularization=dg_reg,
        candidate_regularization=candidate_reg,
        show_error_bars=True,
        CI=True,
    )

    # Create supplementary ranking analysis figure
    print("\n📊 Creating supplementary ranking analysis figure...")
    visualizer.create_ranking_analysis_figure(
        n_triplets=max(n_triplets_list),
        dg_regularization=dg_reg,
        candidate_regularization=candidate_reg,
    )

    # Save CSV files
    print(f"\n{'='*80}")
    print("💾 SAVING ANALYSIS DATA (CSV FILES)")
    print(f"{'='*80}")

    save_folder = f"{visualizer.save_plot_folder}/grouped_analysis"
    os.makedirs(save_folder, exist_ok=True)

    # Save overall recovery data
    overall_df = pd.DataFrame(consolidated_data["overall_recovery"])
    overall_path = f"{save_folder}/overall_model_recovery.csv"
    overall_df.to_csv(overall_path, index=False)
    print(f"✅ Overall model recovery data: {overall_path}")

    # Save team recovery data
    team_df = pd.DataFrame(consolidated_data["team_recovery"])
    team_path = f"{save_folder}/team_recovery_by_group.csv"
    team_df.to_csv(team_path, index=False)
    print(f"✅ Team-based recovery by group: {team_path}")

    print(f"\n{'='*80}")
    print("✨ FIGURES S4 AND S5 GENERATION COMPLETE!")
    print(f"📁 Analysis data saved in: {save_folder}")
    print(f"{'='*80}")


@hydra.main(
    version_base=None,
    config_path="../scripts_configurations",
    config_name="figure_S5_S4",
)
def main(cfg: DictConfig):
    """
    Main function to generate Figures S4 and S5.

    This function orchestrates the entire figure generation pipeline, loading data
    from the configuration file, setting up model groups and colors, and creating
    all publication figures and analysis data files.

    Configuration:
    -------------
    Loaded from: scripts_configurations/figure_S5_S4.yaml

    Figures Created:
    ---------------
    - Figure S4: Model recovery accuracy vs training set size
                 Line plot with confusion matrices at three key sizes

    - Figure S5: Team-based recovery analysis
                 Combined line plots and confusion matrices for architecture
                 and objective groupings

    - Supplementary ranking figure: Mean model rankings analysis

    Output Files:
    ------------
    PDFs saved to: Plots/new_objective_analysis_beta/
    - figure_S4_model_recovery_accuracy.pdf
    - figure_S5_team_recovery.pdf
    - supplementary_ranking_analysis.pdf

    CSVs saved to: Plots/new_objective_analysis_beta/analysis_data/
    - figure_S4_accuracy_by_training_size.csv
    - overall_model_recovery.csv
    - team_recovery_by_group.csv
    """
    # Configuration
    db_path = cfg.db_path

    data_generating_models_list = cfg.models + cfg.new_models

    n_triplets_list = cfg.n_triplets_list

    # Create colors dictionary - this time make them match objectives for consistency
    models_objective = cfg.models_objective
    models_architecture = cfg.models_architecture
    formal_names_dict = cfg.formal_names_dict
    # Create colors dictionary with individual colors for each model within objective groups
    # (similar to Figures_1_and_3.py approach)

    # Group models by objective
    supervised_models = [
        m
        for m in data_generating_models_list
        if models_objective.get(m) == "Supervised"
    ]
    self_supervised_models = [
        m
        for m in data_generating_models_list
        if models_objective.get(m) == "Self-supervised"
    ]
    multimodal_models = [
        m
        for m in data_generating_models_list
        if models_objective.get(m) == "Image & Text Alignment"
    ]

    # Create color lists for each objective type (matching Figures_1_and_3.py approach)
    N_models = len(data_generating_models_list)
    supervised_colors_list = create_colors_list(N_models, start=0, rot=0.5)
    self_supervised_colors_list = create_colors_list(N_models, start=1, rot=0.5)
    multimodal_colors_list = create_colors_list(N_models, start=2, rot=0.5)

    # Assign colors to each model
    models_color_dict = {}

    # Assign colors to supervised models
    for i, model in enumerate(supervised_models):
        models_color_dict[model] = supervised_colors_list[
            i % len(supervised_colors_list)
        ]

    # Assign colors to self-supervised models
    for i, model in enumerate(self_supervised_models):
        models_color_dict[model] = self_supervised_colors_list[
            i % len(self_supervised_colors_list)
        ]

    # Assign colors to multimodal models
    for i, model in enumerate(multimodal_models):
        models_color_dict[model] = multimodal_colors_list[
            i % len(multimodal_colors_list)
        ]

    save_plot_folder = cfg.save_folder
    print("🚀 Starting Figure S4 and S5 Generation")
    print("=" * 80)
    print(f"📊 Analyzing {len(data_generating_models_list)} models")
    print(f"📈 {len(n_triplets_list)} training triplet configurations")
    print(f"🎯 Architecture groups: {len(set(models_architecture.values()))}")
    print(f"🎯 Objective groups: {len(set(models_objective.values()))}")
    print("=" * 80)

    # Create Figures S4 and S5
    create_figure_S4_and_S5(
        db_path=db_path,
        data_generating_models_list=data_generating_models_list,
        n_triplets_list=n_triplets_list,
        formal_names_dict=formal_names_dict,
        models_color_dict=models_color_dict,
        models_architecture=models_architecture,
        models_objective=models_objective,
        simulations_list=None,  # Will use all available simulations
        save_plot_folder=save_plot_folder,
    )


if __name__ == "__main__":
    main()
