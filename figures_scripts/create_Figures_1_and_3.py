"""
================================================================================
Figures 1 and 3: Model Recovery Analysis
================================================================================

This script generates Figure 1 (model recovery accuracy curves and confusion
matrices) and Figure 3 (ranking analysis) from the paper.


Purpose:
--------
Demonstrates that even with 4.2M behavioral judgments, model recovery accuracy
plateaus below 80%, revealing fundamental identifiability issues in flexible
alignment metrics. Shows systematic patterns in when models fail to be correctly
identified.

Figure 1 Components:
--------------------
- Panel A: Recovery accuracy vs. number of training triplets (line plot)
- Panels B-D: Confusion matrices at different training set sizes (400, 25.6K, 1.6M triplets)
- Color coding: Models grouped by training objective
  * Supervised (ImageNet classification): Orange/yellow gradient
  * Self-supervised (contrastive/predictive): Purple/blue gradient  
  * Image-Text Alignment (multimodal): Green/teal gradient

Figure 3 Components:
--------------------
- Panel A: Mean rank when correct model is recovered
- Panel B: Mean rank when wrong model is selected  
- Statistical analysis: Ranking patterns reveal systematic confusions

Model Selection and Color Assignment:
--------------------------------------
This script follows the SAME pattern as create_Figure_2.py for consistency:

1. **Model Selection**: Controlled by `use_additional_models` flag PER EXPERIMENT
   - False: Uses 20 base models from cfg.models
   - True: Uses all 30 models (cfg.models + cfg.new_models)
   - **Important**: Flag is experiment-specific (set in each experiment's config)
   - Must match what models were included when the simulation database was created

2. **Model Ordering**: Derived automatically from cfg.models_objective dictionary
   - Order determined by keys() of models_objective dict
   - Groups by training objective: Supervised → Self-supervised → Multimodal
   - NO hardcoded model lists in configuration

3. **Color Assignment**: Generated using cubehelix palettes (same as Figure 2)
   - Supervised models: Start hue 0, rotation 0.5
   - Self-supervised models: Start hue 1, rotation 0.5
   - Multimodal models: Start hue 2, rotation 0.5
   - Colors assigned based on position in models_ordered list

4. **Counts**: Computed dynamically from models_objective
   - No hardcoded color counts in configuration
   - Adapts automatically based on experiment's use_additional_models setting
   - Example: {"supervised": 8, "self_supervised": 9, "multimodal": 3} for 20 models
   - Example: {"supervised": 12, "self_supervised": 15, "multimodal": 3} for 30 models

Experiment Database Files:
--------------------------
Each experiment_name in the configuration points to a DIFFERENT SQLite database
containing model recovery simulation results:

**Currently Available Experiments:**
- "full_W": Full unconstrained W matrix transformation (no dimensionality reduction)
- "full_W_PCA_500": Full W matrix on PCA-reduced features (500 dimensions)

Both use the same constraint type (full_W = unconstrained transformation matrix)
but differ in whether features undergo PCA dimensionality reduction before
transformation. This allows direct comparison of model recovery performance
with and without dimensionality reduction.

**You can add new experiments** by following the workflow:
1. Create database: `python experiment_scripts/create_simulations_db.py experiment_type=my_experiment`
2. Run simulations: `python run_simulations.py experiment_type=my_experiment`
3. Add configuration in figures_1_and_3.yaml with db_path, N_simulations, n_triplets
4. Generate figures: `python create_Figures_1_and_3.py experiment_name="my_experiment"`

Examples of additional experiment types you could configure:
- Different constraints: diagonal, rectangular_50, zero_shot, orthogonal
- Different regularizations: L1, L2 (eye_distance is the default)
- Regularization mismatch studies: DG_eye_distance_CG_L2
- Different datasets: CC0, CC0_resized

**Database Structure:**
- Location: Results/Simulations_experiments/{experiment_type}/*.db
- Created by: experiment_scripts/create_simulations_db.py
- Populated by: run_simulations.py

**Database Tables:**
```sql
-- jobs table: Simulation parameters
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    n_train_triplets INTEGER,          -- Training set size
    simulation_idx INTEGER,              -- Simulation repetition number
    data_generating_model TEXT,          -- Model that generated synthetic data
    constraint_type TEXT,                -- W matrix constraint (full_W, diagonal, etc.)
    regularization_type TEXT,            -- L1, L2, or eye_distance
    status TEXT,                         -- pending/running/done
    start_time TEXT,
    finish_time TEXT
)

-- reference_model_results table: Recovery accuracy metrics
CREATE TABLE reference_model_results (
    job_id INTEGER,
    reference_model TEXT,                -- Candidate model being tested
    test_accuracy REAL,                  -- % of triplets correctly recovered
    train_accuracy REAL,
    val_accuracy REAL,
    test_NLL REAL,                       -- Negative log-likelihood
    chosen_regularization_constant REAL, -- Best reg. constant from CV
    FOREIGN KEY (job_id) REFERENCES jobs(id)
)
```

**Example Experiments:**

1. "full_W" (DEFAULT) (Figure 1 and 3 in paper)
   - DB: Results/Simulations_experiments/full_W/full_W.db
   - Constraint: Full W matrix (no dimensionality reduction)
   - Purpose: Establish baseline recovery performance

2. "full_W_PCA_500" (Figure S3 in supplemental figures)
   - DB: Results/Simulations_experiments/full_W_PCA_500/full_W_PCA_500.db
   - Constraint: Full W on PCA-reduced features (500 dimensions)
   - Purpose: Test if dimensionality reduction improves recovery
  
**Experiment Naming Convention:**
- "full_W": Full unconstrained transformation matrix
- "PCA_500": Features reduced to 500D via PCA
- "L1"/"L2"/"eye_distance": Regularization type

**How Model Recovery Works:**
1. Fit neural network features to human odd-one-out judgments using learned W matrix
2. Generate synthetic behavioral data from fitted models
3. Try to recover which model generated the data by fitting all candidate models following common used methodology for model-behavioral model recovery alignment
4. Success = correctly identifying the data-generating model

Configuration:
--------------
All parameters specified in: scripts_configurations/figures_1_and_3.yaml

Key configuration sections:
- experiment_name: Selects which experiment DB to visualize
- Per-experiment settings:
  - use_additional_models: Boolean flag for 20 vs 30 models (experiment-specific!)
  - db_path: Path to SQLite database with results
  - N_simulations: Number of simulation runs we want to analyze
  - n_triplets: Training set sizes to analyze
- create_figure_1: Boolean flag to generate Figure 1 (default: True)
- create_figure_3: Boolean flag to generate Figure 3 (default: True)
- Figures_settings_dict: Panel layouts, sizes, confusion matrix triplet counts

Note: use_additional_models is set per-experiment because it must match what
      models were included when the simulation database was created.

Usage:
------
    # Default: Experiment "full_W" (20 models as configured in that experiment)
    python create_Figures_1_and_3.py
    
    # Use different experiment (PCA-reduced features, also 20 models)
    python create_Figures_1_and_3.py experiment_name="full_W_PCA_500"
    
    # Override use_additional_models (only if your database actually has 30 models!)
    # WARNING: This will fail if the database only contains 20 models
    python create_Figures_1_and_3.py experiment_name="full_W" \
        full_W.use_additional_models=True
    
    # Generate only Figure 1 (skip Figure 3)
    python create_Figures_1_and_3.py create_figure_3=False
    
    # Generate only Figure 3 (skip Figure 1)
    python create_Figures_1_and_3.py create_figure_1=False
    
    # Override font sizes for presentation
    python create_Figures_1_and_3.py axis_labels_fontsize=12 tick_labels_fontsize=10
    
    # Example: New experiment with all 30 models (if configured)
    # python create_Figures_1_and_3.py experiment_name="full_W_all_models"
    # (where full_W_all_models.use_additional_models=True in the YAML)

Outputs:
--------
Figures saved to: Plots/{experiment_name}/
- Figure_1.pdf: Main recovery accuracy figure
- Figure_1.png: Raster version
- Figure_3.pdf: Ranking analysis figure  
- Figure_3.png: Raster version

Dependencies:
-------------
- tools/db_analysis.py: DBResultsAnalysis and ModelRecoveryVisualizer classes
- scripts_configurations/: Hydra configuration files
  * figure_base.yaml: Font sizes and formal names
  * shared_args.yaml: Model lists, objectives, architecture
  * figures_1_and_3.yaml: Experiment-specific configuration
  
================================================================================
"""
import os
import sys
import warnings
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
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require GUI
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union,Dict,Optional

# Suppress matplotlib and scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')
warnings.filterwarnings('ignore', message='Using categorical units')
warnings.filterwarnings('ignore', message='The BCa confidence interval')
warnings.filterwarnings('ignore', message='invalid value encountered')

# Set matplotlib logging to ERROR level only (suppress INFO messages)
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)
logging.getLogger('scipy').setLevel(logging.ERROR)

# Suppress figure warnings
plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sns
import matplotlib.colors as mcolors

import hydra
from omegaconf import DictConfig

from tools.db_analysis import DBResultsAnalysis

##############################
# Visualizer using the Analyzer
##############################
class ModelRecoveryVisualizer(DBResultsAnalysis):
    """
    A specialized visualization class for model recovery analysis that extends DBResultsAnalysis.
    
    This class provides comprehensive functionality for creating a variety of visualizations
    related to model recovery experiments, including confusion matrices, rank plots, accuracy
    plots, and precision/recall metrics. It handles plotting with consistent formatting,
    supports panel lettering for publication-ready figures, and manages saving results.
    
    The class is designed to work with simulation data stored in a database containing
    information about models, their performance on various triplet sets, and rankings.
    """
    def __init__(self,
                db_path: str,
                data_generating_models_list: list,
                simulations_list: list,
                n_triplets_list: list,
                reference_models_list: list = None,
                models_color_dict: dict = None,
                save_plot_folder: str = "Plots/",
                formal_names_dict: dict = None,
                models_order_as_colors_dict:list = None
                ):
        """
        Initialize the ModelRecoveryVisualizer with database and visualization parameters.
        
        Parameters:
            db_path (str): Path to the SQLite database file containing experiment results.
            data_generating_models_list (list): List of model names that were used to generate data.
            simulations_list (list): List of simulation indices to analyze.
            n_triplets_list (list): List of training triplet counts to analyze.
            reference_models_list (list, optional): List of reference model names. If None, 
                                                   uses data_generating_models_list.
            models_color_dict (dict, optional): Dictionary mapping model names to colors for 
                                               consistent visualization. 
            save_plot_folder (str, optional): Path to save generated figures. Defaults to 
                                             "04_Plots/Submission_Figures".
            formal_names_dict (dict, optional): Dictionary mapping internal model names to their 
                                               formal display names for publication.
            models_order_as_colors_dict (list, optional): List specifying the order of models 
                                                         when plotting colored bars/elements.
        """
        super().__init__(
            db_path=db_path,
            data_generating_models_list=data_generating_models_list,
            simulations_list=simulations_list,
            reference_models_list=reference_models_list
        )
        self.models_color_dict = models_color_dict
        self.n_triplets_list = n_triplets_list
        self.save_plot_folder = save_plot_folder
        self.formal_names_dict = formal_names_dict

        self.models_order_as_colors_dict = models_order_as_colors_dict
        # Create save folder if it doesn't exist
        if not os.path.exists(self.save_plot_folder):
            os.makedirs(self.save_plot_folder)
    
    def go_formal(self, model_name: Union[str, list, pd.DataFrame]):
        """
        Converts internal model names to their formal display names for publication.
        
        This method handles conversion between internal model identifiers and their formal, 
        publication-ready naming. It supports conversion of a single model name, a list of 
        model names, or an entire DataFrame with model names as index or columns. Special
        handling is included for Google ViT models to ensure consistent display formatting.
        
        Parameters:
            model_name (Union[str, list, pd.DataFrame]): The model name(s) to convert.
                - If str: Single model name to convert to formal name.
                - If list: List of model names to convert.
                - If DataFrame: DataFrame with model names as index/columns to rename.
        
        Returns:
            Union[str, list, pd.DataFrame]: The model name(s) in formal notation.
                - If input was str: Returns formal name as string.
                - If input was list: Returns list of formal names.
                - If input was DataFrame: Returns DataFrame with renamed index/columns.
        
        Raises:
            AssertionError: If formal_names_dict is not provided during initialization.
        """
        assert self.formal_names_dict is not None, "formal_names_dict is not provided"
        if isinstance(model_name, list):
            name_list = []
            for name in model_name:
                if name == "Google_ViT_Large_224" or name == "Google_ViT_Large":
                    name_list.append("ViT L/16")
                else:
                    name_list.append(self.formal_names_dict[name])
            return name_list
        elif isinstance(model_name, pd.DataFrame):
            # Create a copy to avoid modifying the original
            df = model_name.copy()
            
            # Create mapping dictionaries for index and columns
            index_mapping = {}
            for model in df.index:
                if model == "Google_ViT_Large_224" or model == "Google_ViT_Large":
                    index_mapping[model] = "ViT L/16"
                else:
                    index_mapping[model] = self.formal_names_dict[model]
                    
            column_mapping = {}
            for model in df.columns:
                if model == "Google_ViT_Large_224" or model == "Google_ViT_Large":
                    column_mapping[model] = "ViT L/16"
                else:
                    column_mapping[model] = self.formal_names_dict[model]
            
            # Rename index and columns using the mapping dictionaries
            df.rename(index=index_mapping, columns=column_mapping, inplace=True)
            return df
        else:
            if model_name == "Google_ViT_Large_224" or model_name == "Google_ViT_Large":
                return "ViT L/16"
            else:
                return self.formal_names_dict[model_name]
                
    def plot_spearman_correlation_wrong_rank(self, n_triplets: int, ranked_compare_df_data_path: str, ax = None):
        """
        Plot Spearman correlation between wrong rank and true rank.
        
        Args:
            n_triplets (int): Number of triplets to analyze
            ranked_compare_df_data_path (str): Path to the comparison data
            ax (matplotlib.axes.Axes, optional): Existing axis to plot on. If None, creates new figure.
        """
        spearman_corr, df_merged = self.spearman_correlation_wrong_rank(n_triplets, ranked_compare_df_data_path)
        
        if ax is None:
            # Create new figure and axis
            plt.clf()
            fig, ax = plt.subplots()
            save_plot = True
        else:
            save_plot = False
        
        # Create scatter plot
        ax.scatter(df_merged["False Rank"], df_merged["rank"], color='gray', alpha=0.7, s=20)
        
        # Add text labels with model-specific colors
        for i, row in df_merged.iterrows():
            model_name = row["Model"]
            # Handle special case for Google ViT
            color = (self.models_color_dict["Google_ViT_Large"] 
                    if model_name == "Google_ViT_Large_224" 
                    else self.models_color_dict[model_name])
            
            ax.annotate(
                self.go_formal([model_name])[0],
                (row["False Rank"], row["rank"]),
                fontsize=5,
                fontweight='bold',
                ha='center',
                va='bottom',
                xytext=(0, 5),
                textcoords='offset points',
                color=color
            )
        
        # Set labels and title with smaller font sizes
        ax.set_xlabel("Accuracy rank | not data generating model", fontsize=6)
        ax.set_ylabel("Accuracy rank | THINGS dataset", fontsize=6)
        ax.set_title(f"False Rank vs True Rank\nSpearman Correlation: {spearman_corr['False Rank']['rank']:.2f}", 
                    fontsize=8)
        
        # Set smaller tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=5)
        
        if save_plot:
            plt.savefig("04_Plots/spearman_correlation_wrong_rank.pdf", 
                        bbox_inches='tight', 
                        dpi=300)
            plt.close()
        
        return ax

    def create_figure_layout(self,
                    width_cm: float, 
                    height_cm: float,
                    gridspec_dict: dict,
                    subplot_specifics_dict: dict
                    ):
        """
        Creates a figure with precise layout control using GridSpec.
        
        This method builds a figure with highly customizable subplot layouts, removing 
        all default padding and margins for precise positioning of plot elements. The method
        is especially useful for creating publication-ready figures with specific size 
        requirements and complex subplot arrangements.
        
        Parameters:
            width_cm (float): Width of the figure in centimeters.
            height_cm (float): Height of the figure in centimeters.
            gridspec_dict (dict): Dictionary with parameters for matplotlib.gridspec.GridSpec
                                 (e.g., {'nrows': 2, 'ncols': 3}).
            subplot_specifics_dict (dict): Dictionary defining each subplot's position within
                                          the GridSpec. Keys are subplot names, values are
                                          dictionaries containing 'height_index' and 'width_index'
                                          which can be either integers or lists of integers
                                          defining the spanning of the subplot.
        
        Returns:
            tuple: A tuple containing:
                - fig (matplotlib.figure.Figure): The created figure object
                - subplots_dict (dict): Dictionary mapping subplot names to their axes objects
        
        Note:
            This method sets all spacing parameters to zero and removes padding, requiring
            explicit management of any desired spacing between elements.
        """
        width_inches = width_cm / 2.54
        height_inches = height_cm / 2.54
        # Create figure without default padding
        fig = plt.figure(figsize=(width_inches, height_inches), constrained_layout=False)
        # Create GridSpec with zero spacing
        gs = gridspec.GridSpec(
            **gridspec_dict,
            left=0,      
            right=1,     
            bottom=0,    
            top=1,       
            wspace=0,    
            hspace=0     
        )

        subplots_dict = {}
        for name, args in subplot_specifics_dict.items():
            h_index = args["height_index"]
            w_index = args["width_index"]
            # Convert list to slice if necessary
            if type(w_index) != int:
                w_index = slice(min(w_index), max(w_index) + 1)
            if type(h_index) != int:
                h_index = slice(min(h_index), max(h_index) + 1)
            ax = fig.add_subplot(gs[h_index, w_index])
            ax.set_frame_on(False)
            ax.grid(False)
            # Remove internal padding
            ax.margins(0, 0)
                
            subplots_dict[name] = ax

        # Set all spacing parameters to zero
        plt.subplots_adjust(
            left=0,    
            right=1,   
            bottom=0,  
            top=1,     
            wspace=0,  
            hspace=0   
        )
        
        
        return fig, subplots_dict
    
    def draw_bar_plot(
                    self,
                    ax,
                    y_data : Union[list, pd.DataFrame,Dict],
                    x_tick_labels_ordered_no_formal: Optional[list] = None,
                    horizontal: bool = True,
                    show_ticks_labels: bool = True,
                    show_ticks_y: bool = False,
                    zero_pad: bool = True,
                    tick_label_size: int =5,
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
        
        Parameters:
            ax (matplotlib.axes.Axes): The axes object to draw the plot on.
            y_data (Union[list, pd.DataFrame, Dict]): The data to plot as bar heights/lengths.
                - If DataFrame: Uses index as labels and values as bar heights.
                - If Dict: Uses keys as labels and values as bar heights.
                - If list: Uses values as bar heights; requires x_tick_labels_ordered_no_formal.
            x_tick_labels_ordered_no_formal (Optional[list]): List of model names for labels.
                If None and y_data is DataFrame/Dict, uses index/keys respectively.
            horizontal (bool): Whether to plot horizontal bars (True) or vertical bars (False).
                Defaults to True.
            show_ticks_labels (bool): Whether to show tick labels. Defaults to True.
            show_ticks_y (bool): Whether to show numeric y-axis ticks. Defaults to False.
            zero_pad (bool): Whether to set axis limits to start at zero. Defaults to True.
            tick_label_size (int): Font size for tick labels. Defaults to 5.
            error_data (Optional[Dict]): Dictionary mapping model names to error values for
                error bars. Defaults to None.
            show_error_bars (bool): Whether to display error bars. Defaults to True.
            error_capsize (int): Size of error bar caps. Defaults to 3.
            error_color (Optional[str]): Color for error bars. If None, uses bar colors.
                Defaults to None.
            error_lw (float): Line width for error bars. Defaults to 1.0.
            
        Returns:
            matplotlib.container.BarContainer: The container with all bar elements.
            
        Note:
            Bars with zero value are made transparent. Model names are converted to formal
            display names using go_formal before plotting.
        """
        if isinstance(y_data, pd.DataFrame):
            if x_tick_labels_ordered_no_formal is None:
                x_tick_labels_ordered_no_formal = y_data.index.tolist()

                x_tick_labels_ordered_no_formal = x_tick_labels_ordered_no_formal[::-1]
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
            assert x_tick_labels_ordered_no_formal is not None, "x_tick_labels_ordered_no_formal is required when y_data is a list"
            x_formal = self.go_formal(x_tick_labels_ordered_no_formal)
            y = y_data[::-1]
        # Create confidence interval for the error bars
        yerr = [np.nan if error_data[model] == 0 else error_data[model] for model in x_tick_labels_ordered_no_formal]

        
        if horizontal:
            bars = ax.barh(x_formal, y,left=0,xerr=yerr,capsize=2.5,error_kw={'linewidth': 0.2, 'capthick': 0.2})
            tick_labels = ax.get_yticklabels()
        else:
            bars = ax.bar(x_formal, y,yerr=yerr,capsize=2.5,error_kw={'linewidth': 0.2, 'capthick': 0.2})       
            tick_labels = ax.get_xticklabels()

        if self.models_color_dict is not None:
            for bar, model_name, tick_label, value in zip(bars, x_tick_labels_ordered_no_formal, tick_labels, y):
                if model_name == "Google_ViT_Large_224":
                    color = self.models_color_dict["Google_ViT_Large"]
                else:
                    color = self.models_color_dict[model_name]

                if value == 0:
                    bar.set_alpha(0)  # Make zero-value bars completely transparent
                else:
                    bar.set_color(color)
                tick_label.set_color(color)
                tick_label.set_fontsize(tick_label_size)


        
        if not show_ticks_labels:
            if horizontal:
                ax.set_yticks([])
            else:
                ax.set_xticks([])
                
                
        if show_ticks_y:
            if horizontal:
                ax.set_xticks(np.linspace(0, len(self.reference_models_list)-1, 5))
                ax.tick_params(axis='x', labelsize=tick_label_size)
                ax.set_xticklabels([f'{x:.1f}' for x in np.linspace(0, len(self.reference_models_list)-1, 4)])
            else:
                ax.set_yticks(np.linspace(0, len(self.reference_models_list)-1, 5))
                ax.tick_params(axis='y', labelsize=tick_label_size)
                ax.set_yticklabels([f'{y:.1f}' for y in np.linspace(0, len(self.reference_models_list)-1, 4)])
                
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
    def draw_confusion_matrix(self, 
                    ax, 
                    n_triplets, 
                    show_x_label: bool = True,
                    show_y_label: bool = True,
                    show_x_ticks: bool = True,
                    show_y_ticks: bool = True,
                    show_colorbar: bool = True,
                    colorbar_shrink: float = 0.6,
                    colorbar_aspect: float = 30,
                    panel_letter: str = None,
                    panel_letter_position: tuple = (-0.1, 1.1),
                    panel_letter_fontsize: float = 6,
                    panel_letter_weight: str = 'bold',
                    colorbar_ax: plt.Axes = None,
                    axis_label_size: float = None,
                    tick_label_size: float = None,
                    panel_letter_size: float = None,
                    ):
        """
        Creates a highly customized confusion matrix heatmap visualization.
        
        This method generates a publication-quality heatmap visualization of a confusion matrix
        showing model recovery results for a specific number of training triplets. The method
        includes extensive customization options for axis labels, ticks, colorbars, and panel
        lettering, making it suitable for scientific publications.
        
        Parameters:
            ax (matplotlib.axes.Axes): The axes object to draw the plot on.
            n_triplets (int): Number of training triplets to analyze and visualize.
            base_font_size (float): Base font size for scaling other text elements.
            show_x_label (bool): Whether to show the x-axis label. Defaults to True.
            show_y_label (bool): Whether to show the y-axis label. Defaults to True.
            show_x_ticks (bool): Whether to show x-axis tick labels. Defaults to True.
            show_y_ticks (bool): Whether to show y-axis tick labels. Defaults to True.
            show_colorbar (bool): Whether to show the colorbar. Defaults to True.
            colorbar_shrink (float): Shrink factor for colorbar height. Smaller values make
                                    the colorbar shorter. Defaults to 0.6.
            colorbar_aspect (float): Aspect ratio for colorbar. Smaller values make the
                                    colorbar wider. Defaults to 30.
            panel_letter (str, optional): Letter label for the panel (e.g., 'A', 'B').
                                         Defaults to None.
            panel_letter_position (tuple): Position of panel letter relative to axes coordinates.
                                          Defaults to (-0.1, 1.1).
            panel_letter_fontsize (float): Font size for panel letter. Defaults to 6.
            panel_letter_weight (str): Font weight for panel letter ('normal', 'bold', etc.).
                                      Defaults to 'bold'.
            colorbar_ax (plt.Axes, optional): Separate axes for the colorbar. If None, colorbar
                                             is placed alongside the main axes. Defaults to None.
            
        Returns:
            matplotlib.axes.Axes: The axes object with the confusion matrix plot.
            
        Notes:
            - The method automatically verifies data integrity by checking row sums.
            - A custom colormap is created with a distinctive zero color.
            - Model names are converted to formal display names using go_formal.
        """
        # Get the confusion matrix
        confusion_matrix = self.confusion_matrix(n_triplets)
        
        # Verify data integrity
        for model in self.data_generating_models_list:
            row_sum = confusion_matrix.loc[model].sum()
            assert row_sum == len(self.simulations_list), f"{model} have data problem"
        
        confusion_matrix = confusion_matrix.astype(int)

        # Calculate font sizes
        axis_label_size = axis_label_size
        tick_label_size = tick_label_size
        colorbar_fontsize = tick_label_size
        # Create custom colormap with distinctive zero color
        n_colors = 256
        colors_base = plt.cm.YlOrRd(np.linspace(0, 1, n_colors))
        zero_color = [0.95, 0.95, 1, 1]  # Very light blue for zero
        colors_base[0] = zero_color
        colors_base[1:] = plt.cm.YlOrRd(np.linspace(0.1, 1, n_colors-1))
        costum_cmap = LinearSegmentedColormap.from_list('custom_YlOrRd', colors_base, N=n_colors)
        
        # Get the maximum value for scaling
        vmax = float(confusion_matrix.to_numpy().max())
        
        # Create the heatmap
        sns.heatmap(self.go_formal(confusion_matrix),
                annot=False,
                cmap=costum_cmap,
                vmin=0,
                vmax=vmax,
                linewidths=0.1,
                linecolor='gray',
                cbar=show_colorbar,  # Control colorbar visibility
                cbar_kws={
                    'shrink': colorbar_shrink,  # Control colorbar height
                    'aspect': colorbar_aspect,   # Control colorbar width
                    'ticks': np.linspace(0, vmax, 10),
                    'format': '%d',
                    'drawedges': False,
                    'extend': 'both',
                    'extendfrac': 0.005,
                    'extendrect': True,
                },
                cbar_ax=colorbar_ax,
                square=False,
                ax=ax)
        
        if show_colorbar and colorbar_ax is not None:
            # Set label and font size
            colorbar_ax.set_ylabel("number of simulations", fontsize=axis_label_size, labelpad=2)
            # Move label to left
            colorbar_ax.yaxis.set_label_position("left")
            # Remove padding
        ax.margins(0, 0)
        ax.set_frame_on(False)
        ax.grid(False)
        # Remove all spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Set axis labels if enabled
        if show_x_label:
            ax.set_xlabel("recovered model", 
                        fontsize=axis_label_size, 
                        fontweight='bold',
                        labelpad=5)
        else:
            # Remove the x-axis label completely
            ax.set_xlabel("") 
        
        if show_y_label:
            ax.set_ylabel("data-generating model", 
                        fontsize=axis_label_size, 
                        fontweight='bold',
                        labelpad=5)
        else:
            # Remove the y-axis label completely
            ax.set_ylabel("")
        
        # Adjust tick labels
        ax.set_xticks(np.arange(len(confusion_matrix.columns)) + 0.5) 
        ax.set_yticks(np.arange(len(confusion_matrix.index)) + 0.5)
        # Handle x-axis tick labels
        if show_x_ticks:
            ax.set_xticklabels(self.go_formal(confusion_matrix).columns, 
                            rotation=90, 
                            ha='right', 
                            va='top',
                            fontsize=tick_label_size*0.5)
        else:
            ax.set_xticklabels([])
            # Also remove the tick marks completely
            ax.tick_params(axis='x', which='both', length=0)
        
        # Handle y-axis tick labels
        if show_y_ticks:
            ax.set_yticklabels(self.go_formal(confusion_matrix).index, 
                            fontsize=tick_label_size*0.5)
        else:
            ax.set_yticklabels([])
            # Also remove the tick marks completely
            ax.tick_params(axis='y', which='both', length=0)
        
        # Adjust tick parameters for visible ticks
        if show_x_ticks or show_y_ticks:
            ax.tick_params(axis='both', which='major', length=1, width=0.3)
        
        # Adjust colorbar font size if colorbar is shown
        if show_colorbar:
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=tick_label_size)
            cbar.ax.yaxis.set_ticks_position('right')  # Move ticks to left side of colorbar

        
        # Add panel letter if provided
        if panel_letter is not None:
            x, y = panel_letter_position
            ax.text(x, y, panel_letter, 
                   transform=ax.transAxes,
                   fontsize=panel_letter_fontsize,
                   fontweight=panel_letter_weight)
        
        return ax

    def draw_line_plot(self, 
                    ax, 
                    x_data, 
                    y_data, 
                    x_label, 
                    y_label, 
                    tick_label_size, 
                    axis_label_font_size,
                    panel_letter: str = None,
                    panel_letter_position: tuple = (-0.1, 0.98),
                    panel_letter_fontsize: float = 6,
                    panel_letter_weight: str = 'bold',
                    error_data: list = None,
                    error_alpha: float = 0.2,
                    CI: bool = False
                    ):
        """
        Creates a publication-quality line plot with error shading and precise formatting.
        
        This method generates a highly customized line plot with various visual enhancements
        for scientific publication, including error bands/patches, panel lettering, grid styling,
        and spine formatting. The plot supports both standard error bars and confidence intervals.
        
        Parameters:
            ax (matplotlib.axes.Axes): The axes object to draw the plot on.
            x_data (list or array): The data for the x-axis.
            y_data (list or array): The data for the y-axis.
            x_label (str): The label for the x-axis.
            y_label (str): The label for the y-axis.
            tick_label_size (int): The font size for the tick labels.
            panel_letter (str, optional): Letter label for the panel (e.g., 'A', 'B').
                                         Defaults to None.
            panel_letter_position (tuple): Position of panel letter relative to axes coordinates.
                                          Defaults to (-0.1, 0.98).
            panel_letter_fontsize (float): Font size for panel letter. Defaults to 6.
            panel_letter_weight (str): Font weight for panel letter. Defaults to 'bold'.
            error_data (list, optional): Error values to create error patch around the line.
                                        If CI=False, these values are treated as standard errors
                                        to be added/subtracted from y_data.
                                        If CI=True, these should be pairs of (lower, upper) bounds.
                                        Defaults to None.
            error_alpha (float): Alpha transparency value for the error patch. Defaults to 0.2.
            CI (bool): Whether error_data contains confidence intervals (True) or standard
                      errors (False). When True, error_data should be a list of tuples
                      (lower_bound, upper_bound). When False, error_data should be a list
                      of standard errors to add/subtract from y_data. Defaults to False.
            
        Returns:
            matplotlib.axes.Axes: The axes object with the line plot.
            
        Notes:
            - The method automatically sets y-axis limits from 0 to 1 for proportion/accuracy plots.
            - Grid lines are added with reduced opacity for readability.
            - Spines are selectively hidden (top and right) for a cleaner appearance.
        """
        ax.set_frame_on(True)
        ax.clear()
        
        # Plot with smaller markers and thinner lines
        line, = ax.plot(x_data, y_data, 'o-', markersize=2, linewidth=1)
        
        # Add error patches if error_data is provided
        if error_data is not None: 
 
            error_color = line.get_color()
            if CI:
                lower_bound = []
                upper_bound = []
                for i in range(len(error_data)):
                    lower_ci,upper_ci = error_data[i]
                    # Create arrays for upper and lower bounds
                    lower_bound.append(lower_ci)
                    upper_bound.append(upper_ci)
                lower_bound = np.array(lower_bound)
                upper_bound = np.array(upper_bound)
            else:
                lower_bound = np.array(y_data) - np.array(error_data)
                upper_bound = np.array(y_data) + np.array(error_data)
            # Clip lower bound to 0 to prevent negative values (since accuracy can't be negative)
            lower_bound = np.maximum(lower_bound, 0)
            
            # Create error patch
            ax.fill_between(
                range(len(x_data)),  # x values (same as the line plot)
                lower_bound,         # Lower bound of the error
                upper_bound,         # Upper bound of the error
                alpha=error_alpha,   # Transparency level
                color=error_color,   # Use the same color as the line
                linewidth=0          # No border for the patch
            )
        
        # Set labels with bold y-label
        ax.set_xlabel(x_label, fontsize=axis_label_font_size, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=axis_label_font_size, fontweight='bold')
        
        # Set axis limits
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, len(x_data) - 0.5)
        ax.set_yticks([0,0.2,0.5,0.8,1])
        # ax.set_yticklabels([f'{x:.1f}' for x in list(np.linspace(0, 1, 5))])
        lines_width = 0.5
        # Handle spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_bounds(0, 1)  # Set y-axis spine bounds
        ax.spines['bottom'].set_bounds(-0.5, len(x_data) - 0.5)  # Set x-axis spine bounds
        ax.spines['left'].set_linewidth(lines_width)
        ax.spines['bottom'].set_linewidth(lines_width)
        
        # Add light grid
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=lines_width)
        
        # Adjust tick parameters
        ax.tick_params(axis='both', which='major', labelsize=tick_label_size, width=lines_width)
        
        # Rotate x-axis labels for better readability if needed
        # plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # Uncomment to rotate labels
        
        # Add panel letter if provided
        if panel_letter is not None:
            x, y = panel_letter_position
            ax.text(x, y, panel_letter, 
                   transform=ax.transAxes,
                   fontsize=panel_letter_fontsize,
                   fontweight=panel_letter_weight)
        
        # Make plot more compact
        #ax.margins(x=0.02)  # Reduce horizontal margins
        
        return ax
    
    def add_line_plot_letters(self, ax, letters_x_label_dict, letter_size, line_index=0):
        """
        Adds reference letters at specific points on a line plot with connecting dotted lines.
        
        This method is useful for annotating significant data points on a line plot with 
        reference letters (e.g., for cross-referencing within a paper). For each specified
        x-label position, a letter is placed above the corresponding data point, connected
        by a dotted line to enhance visibility.
        
        Parameters:
            ax (matplotlib.axes.Axes): The axes object containing the line plot to annotate.
            letters_x_label_dict (dict): Dictionary mapping x-labels to their corresponding 
                                        letters. Example: {'200': 'B', '1K': 'C', '5K': 'D'}
                                        where keys are x-tick labels and values are the letters
                                        to display above those points.
            letter_size (float): Font size for the annotation letters.
            line_index (int): Index of the line to annotate if multiple lines exist in the plot.
                             Defaults to 0 (first line).
        
        Returns:
            None
        
        Notes:
            - The method searches for exact matches between keys in letters_x_label_dict and
              the x-axis tick labels.
            - Raises a warning if specified x-labels aren't found or the line index is invalid.
            - Letters are positioned slightly above the data points with a dotted connection line.
        """
        assert isinstance(ax, plt.Axes), "ax must be a matplotlib Axes object"
        
        try:
            # Get the specified line object
            line = ax.get_lines()[line_index]
            
            # Get data from the line
            x_positions = line.get_xdata()  # These are the actual x-coordinates
            y_values = line.get_ydata()     # These are the actual y-values
            x_labels = [label.get_text() for label in ax.get_xticklabels()]
            
            for x_label, letter in letters_x_label_dict.items():
                try:
                    # Find the index of the x_label in x_labels
                    idx = x_labels.index(x_label)
                    x_pos = x_positions[idx]
                    y_pos = y_values[idx]
                    
                    # Add dotted line
                    ax.vlines(x_pos, y_pos, y_pos + 0.1, 
                             linestyles=':', 
                             colors='gray', 
                             linewidth=0.8)
                    
                    # Add letter
                    ax.text(x_pos, y_pos + 0.12, letter,
                           ha='center',
                           va='bottom',
                           fontsize=letter_size,
                           fontweight='bold')
                    
                except ValueError:
                    print(f"Warning: Could not find {x_label} in x_labels")
                
        except IndexError:
            print(f"Warning: Could not find line at index {line_index}")

    def create_figure_1(self,
                                                        width_cm: float,
                                                        height_cm: float,
                                                        panel_letters_dict: dict = None,
                                                        gridspec_dict: dict = None,
                                                        subplots_dict: dict = None,
                                                        conf_size_dict: dict = None,
                                                        line_plot_letters_labels_dict: dict = None,
                                                        training_triplets_N = 4200000,
                                                        show_error_bars: bool = True,
                                                        error_alpha: float = 0.2,
                                                        CI: bool = False,
                                                        axis_label_size: int = 4,
                                                        tick_label_size: int = 5,
                                                        panel_letter_size: int = 7,
                                                        ):
        """
        Creates Figure 1 showing model recovery accuracy as a function of training triplets.
        
        This method generates a publication-ready figure with multiple panels:
        1) A line plot showing model recovery accuracy vs. number of training triplets
        2) Confusion matrices at different triplet counts (small, medium, large)
        The figure is designed for use in scientific publications with precise size control,
        panel lettering, and coordinated formatting across panels.
        
        Parameters:
            width_cm (float): Width of the figure in centimeters.
            height_cm (float): Height of the figure in centimeters.
            panel_letters_dict (dict, optional): Dictionary specifying panel letters and their 
                                               positions for each subplot. Expected structure:
                                               {'line_plot': {'letter': 'A', 'position': (-0.1, 1.1),
                                                           'fontsize': 7, 'weight': 'bold'},
                                                'conf_matrix_small': {...}, ...}
            gridspec_dict (dict, optional): Dictionary with parameters for matplotlib.gridspec.GridSpec.
            subplots_dict (dict, optional): Dictionary defining subplot positions within GridSpec.
            conf_size_dict (dict, optional): Dictionary mapping size labels ('small', 'medium', 'large')
                                           to n_triplets values for the confusion matrices.
            line_plot_letters_labels_dict (dict, optional): Dictionary mapping x-labels to lettering
                                                          for cross-referencing with other figures.
                                                          Example: {'200': 'B', '1K': 'C', '5K': 'D'}
            training_triplets_N (int): Number of training triplets used. Defaults to 4200000.
            show_error_bars (bool): Whether to show error bars (binomial error) on the line plot.
                                   Defaults to True.
            error_alpha (float): Alpha transparency value for error bands. Defaults to 0.2.
            CI (bool): Whether to use confidence intervals (True) or standard errors (False).
                      Defaults to False.
        
        Returns:
            None: The figure is saved to disk at {save_plot_folder}/figure_1.pdf
            
        Notes:
            - The method automatically creates analysis folders and saves analysis data
              as a CSV for reproducibility.
            - Different confusion matrix sizes have different formatting options (e.g.,
              only the 'small' matrix shows both x and y labels).
        """
        save_plot_folder = f"{self.save_plot_folder}"
        if not os.path.exists(save_plot_folder):
            os.makedirs(save_plot_folder)
        fig, ax = self.create_figure_layout(
            width_cm=width_cm,
            height_cm=height_cm,
            gridspec_dict=gridspec_dict,
            subplot_specifics_dict=subplots_dict
        )
        
        
        # Ensure both subplots start at the same position
        for ax_name in ax.keys():
            if 'conf_matrix' in ax_name:
                # Move the colorbar to ensure it doesn't affect alignment
                ax[ax_name].set_anchor('W')  # Anchor to the west (left) side
            elif 'n_triplets' in ax_name:
                ax[ax_name].set_anchor('W')  # Anchor to the west (left) side
            elif 'mean_rank' in ax_name:
                ax[ax_name].set_anchor('W')  # Anchor to the west (left) side
        # Convert numpy int64 indices to regular Python integers
        sorted_indices = [int(i) for i in np.argsort(self.n_triplets_list)]
        sorted_n_triplets = [self.n_triplets_list[i] for i in sorted_indices]
        total_model_recovery_accuracy = [self.get_total_model_recovery_accuracy(n_triplets) 
                                    for n_triplets in sorted_n_triplets]
        binomial_error = [self.get_model_recovery_binomial_error(n_triplets,CI=CI) 
                                    for n_triplets in sorted_n_triplets]
        # Save the total model recovery accuracy to a CSV file
        analysis_save_path = os.path.join(self.save_plot_folder,"analysis","Figure_1_analysis.csv")
        os.makedirs(os.path.dirname(analysis_save_path),exist_ok=True)
        recovery_df = pd.DataFrame(total_model_recovery_accuracy,index=sorted_n_triplets,columns=['model recovery accuracy'])
        error_df = pd.DataFrame(binomial_error,index=sorted_n_triplets,columns=["low_CI","High_CI"])
        figure_1_df = pd.concat([recovery_df,error_df],axis=1)
        figure_1_df.to_csv(analysis_save_path)
        
        
        # Format x-axis labels
        x_labels = []
        for n in sorted_n_triplets:
            if n >= 1_000_000:
                x_labels.append(f"{n/1_000_000:.1f}M")
            elif n >= 1000:
                x_labels.append(f"{n/1000:.1f}K")
            else:
                x_labels.append(str(n))

        # Calculate font sizes based on figure dimensions
        width_inches = width_cm / 2.54
        height_inches = height_cm / 2.54


        axis_label_size = axis_label_size
        tick_label_size = tick_label_size
        panel_letter_size = panel_letter_size
        for ax_name in ax.keys():
            if 'n_triplets' in ax_name:
                self.draw_line_plot(ax[ax_name], 
                                    x_data=x_labels, 
                                    y_data=total_model_recovery_accuracy, 
                                    x_label="number of training triplets", 
                                    y_label="model recovery accuracy", 
                                    tick_label_size=tick_label_size,
                                    axis_label_font_size=axis_label_size,
                                    panel_letter=panel_letters_dict['line_plot']['letter'],
                                    panel_letter_position=panel_letters_dict['line_plot']['position'],
                                    panel_letter_fontsize=panel_letters_dict['line_plot']['fontsize'],
                                    panel_letter_weight=panel_letters_dict['line_plot']['weight'],
                                    error_data=binomial_error if show_error_bars else None,
                                    error_alpha=error_alpha,
                                    CI=CI
                                    )
                if line_plot_letters_labels_dict is not None:
                    self.add_line_plot_letters(ax[ax_name], 
                                    letters_x_label_dict=line_plot_letters_labels_dict,
                                    letter_size=panel_letters_dict['line_plot']['fontsize'],
                                    line_index=0
                                    ) 
            elif 'conf_matrix' in ax_name:
                size = ax_name.split('_')[-1]
                n_triplets = conf_size_dict[size]
                if size == "small":
                    show_x_label = True
                    show_y_label = True
                    show_x_ticks = True
                    show_y_ticks = True
                    show_colorbar = False
                    # panel_letter_size = None
                    # panel_letter = "B"
                elif size == "medium":
                    show_x_label = False
                    show_y_label = False
                    show_x_ticks = True
                    show_y_ticks = False
                    show_colorbar = False
                    # panel_letter = "C"
                elif size == "large":
                    show_x_label = False
                    show_y_label = False
                    show_x_ticks = True
                    show_y_ticks = False
                    show_colorbar = True
                    colorbar_shrink = 0.9
                    colorbar_aspect = 50
                    # panel_letter = "D"
                self.draw_confusion_matrix(ax[ax_name], 
                                    n_triplets=n_triplets, 
                                    show_x_label=show_x_label,
                                    show_y_label=show_y_label,
                                    show_x_ticks=show_x_ticks,
                                    show_y_ticks=show_y_ticks,
                                    show_colorbar=show_colorbar,
                                    colorbar_shrink=colorbar_shrink if 'colorbar_shrink' in locals() else 0.9,
                                    colorbar_aspect=colorbar_aspect if 'colorbar_aspect' in locals() else 50,
                                    panel_letter=panel_letters_dict[f'conf_matrix_{size}']['letter'],
                                    panel_letter_fontsize=panel_letters_dict[f'conf_matrix_{size}']['fontsize'],
                                    panel_letter_weight=panel_letters_dict[f'conf_matrix_{size}']['weight'],
                                    colorbar_ax = ax[f'colorbar'] if 'colorbar' in ax.keys() else None,
                                    axis_label_size=axis_label_size,
                                    tick_label_size=tick_label_size,
                                    )
        if "PCA_500" in save_plot_folder:
            figure_name = "Figure_S3"
        else:
            figure_name = "Figure_1"
        plt.savefig(f"{save_plot_folder}/{figure_name}_4.pdf",
                bbox_inches=None,
                dpi=600,
                format='pdf')
        plt.close()
    
    def create_figure_3(self,
                    width_cm,
                    height_cm,
                    gridspec_dict,
                    subplots_dict,
                    training_triplets_N,
                    panel_letters_dict,
                    show_error_bars =True,
                    rank_error_capsize =3,
                    rank_error_lw =0.8, 
                    rank_error_color = "black",
                    axis_label_size = 4,
                    tick_label_size = 5,
                    panel_letter_size = 7,
                    ):
        """
        Creates Figure 3 showing model ranking analysis with paired bar plots.
        
        This method generates a publication-ready figure showing two bar plots:
        1) The mean rank of each model when it is the correct (data-generating) model
        2) The mean rank of each model when it is NOT the data-generating model
        
        The figure visualizes how well different models perform in model recovery tasks,
        highlighting both their accuracy when they are the true model and when they are
        being used as a reference for other models.
        
        Parameters:
            width_cm (float): Width of the figure in centimeters.
            height_cm (float): Height of the figure in centimeters.
            gridspec_dict (dict): Dictionary with parameters for matplotlib.gridspec.GridSpec.
            subplots_dict (dict): Dictionary defining subplot positions within GridSpec.
            training_triplets_N (int): Number of training triplets to use for the analysis.
            panel_letters_dict (dict): Dictionary specifying panel letters and their 
                                      positions for each subplot. Expected structure:
                                      {'mean_rank_correct': {'letter': 'A', 'position': (-0.1, 1.1),
                                                         'fontsize': 7, 'weight': 'bold'},
                                       'mean_rank_opposite': {...}}
            show_error_bars (bool): Whether to show standard error bars on the bar plots.
                                   Defaults to True.
            rank_error_capsize (int): Size of error bar caps. Defaults to 3.
            rank_error_lw (float): Line width for error bars. Defaults to 0.8.
            rank_error_color (str): Color for error bars. Defaults to "black".
            
        Returns:
            None: The figure is saved to disk at {save_plot_folder}/figure_3.pdf
            
        Notes:
            - Rank values are inverted so that higher values (bars) indicate better ranking
              (1 = highest rank, N = lowest rank).
            - A vertical line is added at the midpoint to indicate chance level performance.
            - Axis ticks are specially formatted with "ranked first", "chance level", and
              "ranked last" labels at appropriate positions.
            - Analysis data is saved as a CSV for reproducibility.
        """


        if not os.path.exists(self.save_plot_folder):
            os.makedirs(self.save_plot_folder)
        
        fig, ax = self.create_figure_layout(
            width_cm=width_cm,
            height_cm=height_cm,
            gridspec_dict=gridspec_dict,
            subplot_specifics_dict=subplots_dict,
        )
        

        # Ensure both subplots start at the same position
        for ax_name in ax.keys():
            ax[ax_name].set_anchor('W')  # Anchor to the west (left) side
        # Convert numpy int64 indices to regular Python integers
        sorted_indices = [int(i) for i in np.argsort(self.n_triplets_list)]
        sorted_n_triplets = [self.n_triplets_list[i] for i in sorted_indices]
        

        
        df, corect_ranks_dict, false_ranks_dict, correct_ranks_dict_std, false_ranks_dict_std = self.calculate_models_mean_rank_correct_and_incorrect(training_triplets_N,return_df=True)
        # Invert the y data so that a higher rank indicates better performance
        corect_ranks_dict = {k: len(self.reference_models_list) - v + 1 for k, v in corect_ranks_dict.items()}
        false_ranks_dict = {k: len(self.reference_models_list) - v + 1 for k, v in false_ranks_dict.items()}
        # Save corect_ranks_dict and false_ranks_dict to a CSV file
        analysis_save_path = os.path.join(self.save_plot_folder,"analysis","Figure_3_analysis.csv")
        os.makedirs(os.path.dirname(analysis_save_path),exist_ok=True)
        df.to_csv(analysis_save_path)
        
        for ax_name in ax.keys():
            
            if 'correct' in ax_name:
                self.draw_bar_plot(ax[ax_name],
                                    y_data=corect_ranks_dict, 
                                    x_tick_labels_ordered_no_formal=self.models_order_as_colors_dict,
                                    error_data=correct_ranks_dict_std if show_error_bars else None,  # Add error data
                                    show_error_bars=show_error_bars,  # Control error bar visibility
                                    error_capsize=rank_error_capsize,  # Set error bar cap size
                                    error_lw=rank_error_lw,  # Set error bar line width
                                    error_color=rank_error_color,
                                    tick_label_size=tick_label_size,# Set error bar color
                                    )
                ## Add panel letter
                ax[ax_name].text(x=panel_letters_dict['mean_rank_correct']['position'][0],
                                    y=panel_letters_dict['mean_rank_correct']['position'][1],
                                    s=panel_letters_dict['mean_rank_correct']['letter'],
                                    fontsize=panel_letters_dict['mean_rank_correct']['fontsize'],
                                    fontweight=panel_letters_dict['mean_rank_correct']['weight']
                                    )
                ## Add custom Ticks 
            elif 'opposite' in ax_name:
                self.draw_bar_plot(ax[ax_name],
                                    y_data=false_ranks_dict, 
                                    x_tick_labels_ordered_no_formal=self.models_order_as_colors_dict,
                                    error_data=false_ranks_dict_std if show_error_bars else None,  # Add error data
                                    show_error_bars=show_error_bars,  # Control error bar visibility
                                    error_capsize=rank_error_capsize,  # Set error bar cap size
                                    error_lw=rank_error_lw,  # Set error bar line width
                                    error_color=rank_error_color  # Set error bar color
                                    )
                                    
                ## Add panel letter
                ax[ax_name].text(x=panel_letters_dict['mean_rank_opposite']['position'][0],
                                    y=panel_letters_dict['mean_rank_opposite']['position'][1],
                                    s=panel_letters_dict['mean_rank_opposite']['letter'],
                                    fontsize=panel_letters_dict['mean_rank_opposite']['fontsize'],
                                    fontweight=panel_letters_dict['mean_rank_opposite']['weight']
                                )
                #add custom ticks
            ticks = np.linspace(1,len(self.reference_models_list),len(self.reference_models_list))
            ax[ax_name].set_xticks(ticks)
            tick_labels = ax[ax_name].get_xticklabels()
            tick_list = ["" for _ in range(ticks.shape[0])]
            tick_list[0] = "ranked last"
            tick_list[int((len(tick_list)-1)/2)] = "chance\nlevel"
            tick_list[-1] = "ranked first"
            ax[ax_name].set_xticklabels(tick_list)
            ax[ax_name].tick_params(axis='both', length=1, width=0.5, labelsize=tick_label_size)
            ax[ax_name].axvline(x=len(self.reference_models_list)/2, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax[ax_name].set_xlim(0,len(self.reference_models_list))
            # Show frame and spines
            ax[ax_name].set_frame_on(True)
            ax[ax_name].spines['top'].set_visible(False)
            ax[ax_name].spines['right'].set_visible(False)
            ax[ax_name].spines['bottom'].set_visible(True)
            ax[ax_name].spines['left'].set_visible(True)
            ax[ax_name].spines['bottom'].set_linewidth(0.5)
            ax[ax_name].spines['left'].set_linewidth(0.5)
        if "PCA_500" in self.save_plot_folder:
            figure_name = "Figure_S3_model_ranking"
        else:
            figure_name = "Figure_3"
        plt.savefig(f"{self.save_plot_folder}/{figure_name}.pdf",
                    bbox_inches=None,
                    dpi=600,
                    format='pdf')
        plt.close()

    def one_model_metrics_plot_for_N_triplets(
                                                self, data_generating_model_name: str, width_cm: float, height_cm: float,
                                                panel_letter_precision: str = None, panel_letter_recall: str = None,
                                                panel_letter_accuracy: str = None, panel_letter_f1: str = None
                                                ):
        """
        Generates detailed performance metric plots for a single model across different triplet counts.
        
        This method creates two publication-quality figures for a given data generating model:
        1) A figure with precision and recall metrics as functions of the number of training triplets
        2) A figure with accuracy and F1 score metrics as functions of the number of training triplets
        
        Each figure contains two vertically stacked subplots with coordinated formatting and
        optional panel lettering suitable for scientific publications.
        
        Parameters:
            data_generating_model_name (str): Name of the model to analyze.
            width_cm (float): Width of each figure in centimeters.
            height_cm (float): Height of each figure in centimeters.
            panel_letter_precision (str, optional): Panel letter for the precision subplot.
            panel_letter_recall (str, optional): Panel letter for the recall subplot.
            panel_letter_accuracy (str, optional): Panel letter for the accuracy subplot.
            panel_letter_f1 (str, optional): Panel letter for the F1 score subplot.
            
        Returns:
            None: The figures are saved to disk at:
                 {save_plot_folder}/model_metrics/{data_generating_model_name}/
                 with filenames:
                 - {data_generating_model_name}_precision_recall.pdf
                 - {data_generating_model_name}_accuracy_f1.pdf
                 
        Notes:
            - The method automatically computes all metrics for each triplet count in n_triplets_list.
            - Metrics are calculated using get_model_precision_recall_f1_accuracy.
            - X-axis labels are formatted with K/M suffixes for thousands/millions.
            - The plots feature coordinated styling with gridlines and consistent formatting.
        """
        # Convert numpy int64 indices to regular Python integers
        sorted_indices = [int(i) for i in np.argsort(self.n_triplets_list)]
        sorted_n_triplets = [self.n_triplets_list[i] for i in sorted_indices]
        precision_list, recall_list, accuracy_list, f1_list = [], [], [], []

        for n in tqdm(sorted_n_triplets, desc=f"Metrics for {data_generating_model_name}", unit="triplets",position=1):
            metrics = self.get_model_precision_recall_f1_accuracy(data_generating_model_name, n)
            precision_list.append(metrics["precision"])
            recall_list.append(metrics["recall"])
            accuracy_list.append(metrics["accuracy"])
            f1_list.append(metrics["f1"])

        # Format x-axis labels
        x_labels = []
        for n in sorted_n_triplets:
            if n >= 1_000_000:
                x_labels.append(f"{n/1_000_000:.1f}M")
            elif n >= 1000:
                x_labels.append(f"{n/1000:.0f}K")
            else:
                x_labels.append(str(n))

        # Calculate font sizes based on figure dimensions
        width_inches = width_cm / 2.54
        height_inches = height_cm / 2.54
        base_font_size = min(width_inches, height_inches) * 1.5
        
        axis_label_size = 10
        tick_label_size = 5
        panel_letter_size = 7

        # Create folder for model metrics plots
        save_folder = os.path.join(self.save_plot_folder, "model_metrics", data_generating_model_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Plot Precision and Recall in one figure with two subplots
        fig1 = plt.figure(constrained_layout=True)
        fig1.set_size_inches(width_inches, height_inches)
        gs1 = fig1.add_gridspec(2, 1, height_ratios=[1, 1])
        ax1 = fig1.add_subplot(gs1[0])
        ax2 = fig1.add_subplot(gs1[1])

        ax1.plot(range(len(sorted_n_triplets)), precision_list, 'o-')
        ax1.set_xticks(range(len(sorted_n_triplets)))
        ax1.set_xticklabels(x_labels, rotation=45, ha='right')
        ax1.set_ylabel("Precision", fontsize=axis_label_size)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.tick_params(axis='both', which='major', labelsize=tick_label_size)
        if panel_letter_precision:
            ax1.text(-0.1, 1.1, panel_letter_precision, transform=ax1.transAxes,
                     fontsize=panel_letter_size, fontweight='bold')

        ax2.plot(range(len(sorted_n_triplets)), recall_list, 'o-')
        ax2.set_xticks(range(len(sorted_n_triplets)))
        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
        ax2.set_xlabel("Number of training triplets", fontsize=axis_label_size)
        ax2.set_ylabel("Recall", fontsize=axis_label_size)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.tick_params(axis='both', which='major', labelsize=tick_label_size)
        if panel_letter_recall:
            ax2.text(-0.1, 1.1, panel_letter_recall, transform=ax2.transAxes,
                     fontsize=panel_letter_size, fontweight='bold')

        fig1.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, hspace=0.1, wspace=0.1)
        plt.savefig(os.path.join(save_folder, f"{data_generating_model_name}_precision_recall.pdf"),
                    bbox_inches=None, dpi=600, format='pdf')
        plt.close(fig1)

        # Plot Accuracy and F1 in a second figure with two subplots
        fig2 = plt.figure(constrained_layout=True)
        fig2.set_size_inches(width_inches, height_inches)
        gs2 = fig2.add_gridspec(2, 1, height_ratios=[1, 1])
        ax3 = fig2.add_subplot(gs2[0])
        ax4 = fig2.add_subplot(gs2[1])

        ax3.plot(range(len(sorted_n_triplets)), accuracy_list, 'o-')
        ax3.set_xticks(range(len(sorted_n_triplets)))
        ax3.set_xticklabels(x_labels, rotation=45, ha='right')
        ax3.set_ylabel("Accuracy", fontsize=axis_label_size)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.tick_params(axis='both', which='major', labelsize=tick_label_size)
        if panel_letter_accuracy:
            ax3.text(-0.1, 1.1, panel_letter_accuracy, transform=ax3.transAxes,
                     fontsize=panel_letter_size, fontweight='bold')

        ax4.plot(range(len(sorted_n_triplets)), f1_list, 'o-')
        ax4.set_xticks(range(len(sorted_n_triplets)))
        ax4.set_xticklabels(x_labels, rotation=45, ha='right')
        ax4.set_xlabel("Number of Training Triplets", fontsize=axis_label_size)
        ax4.set_ylabel("F1 Score", fontsize=axis_label_size)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.tick_params(axis='both', which='major', labelsize=tick_label_size)
        if panel_letter_f1:
            ax4.text(-0.1, 1.1, panel_letter_f1, transform=ax4.transAxes,
                     fontsize=panel_letter_size, fontweight='bold')

        fig2.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, hspace=0.1, wspace=0.1)
        plt.savefig(os.path.join(save_folder, f"{data_generating_model_name}_accuracy_f1.pdf"),
                    bbox_inches=None, dpi=600, format='pdf')
        plt.close(fig2)

    def all_models_metrics_plots_for_N_triplets(self, width_cm: float, height_cm: float,
                                precision_panel_letter: str = None, recall_panel_letter: str = None,
                                accuracy_panel_letter: str = None, f1_panel_letter: str = None):
        """
        Generates performance metric plots for all data-generating models across all triplet counts.
        
        This method iterates through all models in data_generating_models_list and creates
        performance plots for each one, effectively executing one_model_metrics_plot_for_N_triplets
        for every model. It generates a comprehensive set of visualizations showing how each
        model performs as the data-generating model at different training triplet counts.
        
        Parameters:
            width_cm (float): Width of each figure in centimeters.
            height_cm (float): Height of each figure in centimeters.
            precision_panel_letter (str, optional): Panel letter for all precision subplots.
            recall_panel_letter (str, optional): Panel letter for all recall subplots.
            accuracy_panel_letter (str, optional): Panel letter for all accuracy subplots.
            f1_panel_letter (str, optional): Panel letter for all F1 score subplots.
            
        Returns:
            None: Figures are saved to disk at:
                 {save_plot_folder}/model_metrics/{model_name}/
                 for each model in data_generating_models_list.
                 
        Notes:
            - This is a batch operation that may take significant time with many models.
            - Progress is displayed with a tqdm progress bar showing model-by-model completion.
            - Each model will have two plots: one for precision/recall and one for accuracy/F1.
            - The panel lettering is consistent across all models for easier comparison.
        """
        for model in tqdm(self.data_generating_models_list,
                        desc="Creating metric plots for all models", 
                        unit="model",
                        position=0):
            self.one_model_metrics_plot_for_N_triplets(
                data_generating_model_name=model,
                width_cm=width_cm,
                height_cm=height_cm,
                panel_letter_precision=precision_panel_letter,
                panel_letter_recall=recall_panel_letter,
                panel_letter_accuracy=accuracy_panel_letter,
                panel_letter_f1=f1_panel_letter
            )

    def one_confusion_matrix(self,
                            n_triplets: int,
                            height_cm: float,
                            width_cm: float,
                            panel_letter: str = None
                        ):
        """
        Creates a standalone confusion matrix figure for a specific triplet count.
        
        This method generates a publication-quality visualization of a confusion matrix
        showing model recovery results for a specific number of training triplets. Unlike
        the draw_confusion_matrix method which adds a confusion matrix to an existing axes,
        this method creates a complete standalone figure with appropriate formatting.
        
        Parameters:
            n_triplets (int): Number of training triplets to visualize the confusion matrix for.
            height_cm (float): Height of the figure in centimeters.
            width_cm (float): Width of the figure in centimeters.
            panel_letter (str, optional): Letter label for the panel (e.g., 'A', 'B').
                                         Defaults to None.
            
        Returns:
            matplotlib.figure.Figure: The figure containing the confusion matrix.
            
        Notes:
            - The method automatically verifies data integrity by checking row sums.
            - A custom colormap is created with a distinctive zero color.
            - Model names are converted to formal display names using go_formal.
            - The resulting figure can be saved separately or embedded in other contexts.
        """
        # Get the confusion matrix
        confusion_matrix = self.confusion_matrix(n_triplets)
        
        # Verify data integrity
        for model in self.data_generating_models_list:
            row_sum = confusion_matrix.loc[model].sum()
            assert row_sum == len(self.simulations_list), f"{model} have data problem"
        
        confusion_matrix = confusion_matrix.astype(int)
        
        # Convert cm to inches
        width_inches = width_cm / 2.54
        height_inches = height_cm / 2.54

        # Calculate font sizes
        base_font_size = min(width_inches, height_inches) * 1.5
        axis_label_size = 10
        tick_label_size = 5
        colorbar_fontsize = 6
        
        # Create figure with automatic layout
        fig, ax = plt.subplots(figsize=(width_inches, height_inches), 
                            constrained_layout=True)
        
        # Create custom colormap with distinctive zero color
        n_colors = 256
        colors_base = plt.cm.YlOrRd(np.linspace(0, 1, n_colors))
        zero_color = [0.95, 0.95, 1, 1]  # Very light blue for zero
        colors_base[0] = zero_color
        colors_base[1:] = plt.cm.YlOrRd(np.linspace(0.1, 1, n_colors-1))
        custom_cmap = LinearSegmentedColormap.from_list('custom_YlOrRd', colors_base, N=n_colors)
        
        # Get the maximum value for scaling
        vmax = float(confusion_matrix.to_numpy().max())
        
        # Create the heatmap
        sns.heatmap(self.go_formal(confusion_matrix),
                annot=False,
                cmap=custom_cmap,
                vmin=0,
                vmax=vmax,
                linewidths=0.2,
                linecolor='gray',
                cbar_kws={
                    'shrink': 0.6,
                    'aspect': 30,
                    'ticks': np.linspace(0, (int(vmax/2)*2), 10),
                    'format': '%d'
                },
                square=False,
                ax=ax)
        
        # Add panel letter if provided
        if panel_letter:
            fig.text(0.02, 0.95, panel_letter, 
                    fontsize=6,
                    fontweight='bold',
                    transform=fig.transFigure)
        
        # Set  and labels

        
        ax.set_xlabel("Recovered Model", 
                    fontsize=8, 
                    fontweight='bold',
                    labelpad=2)
        
        ax.set_ylabel("Data Generating Model", 
                    fontsize=8, 
                    fontweight='bold',
                    labelpad=1)
        
        # Adjust tick labels
        ax.set_xticks(np.arange(len(confusion_matrix.columns)) + 0.5)
        ax.set_yticks(np.arange(len(confusion_matrix.index)) + 0.5)
        
        ax.set_xticklabels(self.go_formal(confusion_matrix).columns, 
                           rotation=45, 
                           ha='right', 
                           va='top',
                           fontsize=5)
        
        ax.set_yticklabels(self.go_formal(confusion_matrix).index, 
                           fontsize=5)
        
        # Adjust tick parameters
        ax.tick_params(axis='both', which='major', length=1, width=0.3)
        
        # Adjust colorbar font size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6)
        
        return fig

    def one_precision_recall_bar_plot(self,
                                n_triplets: int,
                                width_cm: float,
                                height_cm: float,
                                gridspec_dict: dict, 
                                subplots_dict: dict,
                                panel_letter: str = None,
                                at_K: bool = False        
                                ):
        """
        Creates a figure with precision and recall bar plots for all models at a specific triplet count.
        
        This method generates a publication-quality figure with paired bar plots showing 
        precision and recall metrics for all models at a specific number of training triplets.
        Typically, the figure is divided into sections for different model architectures
        (e.g., CNN vs. Transformer) to facilitate comparison within architecture groups.
        
        Parameters:
            n_triplets (int): Number of training triplets for analysis.
            width_cm (float): Width of the figure in centimeters.
            height_cm (float): Height of the figure in centimeters.
            gridspec_dict (dict): Dictionary with parameters for matplotlib.gridspec.GridSpec.
            subplots_dict (dict): Dictionary defining subplot positions within GridSpec.
            panel_letter (str, optional): Letter label for the panel (e.g., 'A', 'B').
                                         Defaults to None.
            at_K (bool): Whether to calculate precision and recall at K (True) or use standard
                        precision and recall (False). When True, uses get_k_mean_precision_recall.
                        When False, uses get_model_precision_recall_f1_accuracy. Defaults to False.
            
        Returns:
            None: The figure is saved to disk at:
                 {save_plot_folder}/precision_recall_bars[_at_K]/precision_recall_bars_{n_triplets}.pdf
                 
        Notes:
            - Bars are color-coded by model according to models_color_dict.
            - Models are grouped by architecture type for easier comparison.
            - The 'at K' variant calculates precision/recall across multiple K values (top-K
              ranked models), averaged for each reference model.
        """
        # Create save folder if it doesn't exist
        if at_K:
            save_folder = f"{self.save_plot_folder}/precision_recall_bars_at_K"
        else:
            save_folder = f"{self.save_plot_folder}/precision_recall_bars"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Create the Figure layout
        fig, axs = self.create_figure_layout(
                                width_cm=width_cm,
                                height_cm=height_cm,
                                gridspec_dict=gridspec_dict,
                                subplot_specifics_dict=subplots_dict
                                )
        base_font_size = min(width_cm, height_cm) / 2.54 * 2  # Convert cm to inches and scale
        ticks_label_size = base_font_size * 0.8
        # Add panel letter if provided
        if panel_letter:
            # Calculate font size based on figure dimensions
            panel_letter_size = base_font_size * 1.4
            
            # Add text in the upper left corner of the figure
            fig.text(0.02, 0.96, panel_letter, 
                    fontsize=panel_letter_size,
                    fontweight='bold',
                    transform=fig.transFigure)
        
        # Get metrics for all models
        precision_dict, recall_dict = {}, {}    
        # Collect data
        if at_K:
            results = self.get_k_mean_precision_recall(n_triplets)
            
        for architecture in self.model_arch_dict.keys():
            precision_dict[architecture], recall_dict[architecture] = {}, {}
            for model in self.model_arch_dict[architecture].keys():       
                if at_K:
                    metrics = results[model]
                else:
                    metrics = self.get_model_precision_recall_f1_accuracy(model, n_triplets)
                precision_dict[architecture][model] = metrics["precision"]
                recall_dict[architecture][model] = metrics["recall"]

        architecture_list = list(self.model_arch_dict.keys())
        
        # Draw the plots with appropriate tick settings
        for ax_name in axs.keys():
            if "precision" in ax_name:
                if architecture_list[0] in ax_name:
                    self.draw_bar_plot(axs[ax_name], precision_dict[architecture_list[0]], show_ticks_labels=True,show_ticks_y=True,tick_label_size=ticks_label_size,)
                elif architecture_list[1] in ax_name:
                    self.draw_bar_plot(axs[ax_name], precision_dict[architecture_list[1]], show_ticks_labels=True,tick_label_size=ticks_label_size,)
                    axs[ax_name].set_title("Precision" if not at_K else "Precision@K", fontsize=base_font_size )
            elif "recall" in ax_name:
                if architecture_list[0] in ax_name:
                    self.draw_bar_plot(axs[ax_name], recall_dict[architecture_list[0]],show_ticks_y=True,tick_label_size=ticks_label_size)
                elif architecture_list[1] in ax_name:
                    self.draw_bar_plot(axs[ax_name], recall_dict[architecture_list[1]],tick_label_size=ticks_label_size)
                    axs[ax_name].set_title("Recall" if not at_K else "Recall@K", fontsize=base_font_size )

        # Save the figure
        plt.savefig(f"{save_folder}/precision_recall_bars_{n_triplets}.pdf",
                    dpi=600, format='pdf')
        plt.close()
    
    def create_and_save_precision_recall_bar_plots_for_all_n_triplets(self,
                                                                height_cm: float,
                                                                width_cm: float,
                                                                gridspec_dict: dict,
                                                                subplots_dict: dict,
                                                                panel_letter: str = None,
                                                                at_K: bool = False
                                                                ):
        """
        Generates precision and recall bar plots for all models across all triplet counts.
        
        This method creates a comprehensive set of bar plot figures showing precision and 
        recall metrics for all models at each triplet count in n_triplets_list. It's a 
        batch operation that effectively runs one_precision_recall_bar_plot for each 
        triplet count, creating a sequence of figures that can show performance trends.
        
        Parameters:
            height_cm (float): Height of each figure in centimeters.
            width_cm (float): Width of each figure in centimeters.
            gridspec_dict (dict): Dictionary with parameters for matplotlib.gridspec.GridSpec.
            subplots_dict (dict): Dictionary defining subplot positions within GridSpec.
            panel_letter (str, optional): Letter label for the panel on each figure.
                                         Defaults to None.
            at_K (bool): Whether to calculate precision and recall at K (True) or standard
                        precision and recall (False). Defaults to False.
            
        Returns:
            None: Figures are saved to disk at:
                 {save_plot_folder}/precision_recall_bars[_at_K]/precision_recall_bars_{n_triplets}.pdf
                 for each n_triplets in n_triplets_list.
                 
        Notes:
            - This is a batch operation that creates multiple figures, one per triplet count.
            - Progress is displayed with a tqdm progress bar showing completion by triplet count.
            - The output directory changes based on the at_K parameter.
        """
        for n_triplets in tqdm(self.n_triplets_list,
                            desc="Creating precision recall bar plots",
                            unit="plot"):
            self.one_precision_recall_bar_plot(n_triplets, height_cm, width_cm, gridspec_dict, subplots_dict, panel_letter,at_K)
    
    def create_and_save_confusion_matrix_plots_for_all_n_triplets(self,
                                                                height_cm: float,
                                                                width_cm: float,
                                                                panel_letter: str = None
                                                                ):
        """
        Generates confusion matrix plots for all triplet counts.
        
        This method creates a comprehensive set of confusion matrix figures, one for
        each triplet count in n_triplets_list. It provides a batch approach to visualizing
        how model recovery performance changes across different training set sizes.
        
        Parameters:
            height_cm (float): Height of each figure in centimeters.
            width_cm (float): Width of each figure in centimeters.
            panel_letter (str, optional): Letter label for the panel on each figure.
                                         Defaults to None.
            
        Returns:
            None: Figures are saved to disk at:
                 {save_plot_folder}/confusion_matrices/confusion_matrix_{n_triplets}.pdf
                 for each n_triplets in n_triplets_list.
                 
        Notes:
            - This is a batch operation that creates multiple figures, one per triplet count.
            - Progress is displayed with a tqdm progress bar showing completion by triplet count.
            - Each confusion matrix shows recovery performance for all model pairs at a specific
              number of training triplets.
        """
        confusion_matrices_folder = f"{self.save_plot_folder}/confusion_matrices"
        if not os.path.exists(confusion_matrices_folder):
            os.makedirs(confusion_matrices_folder)
            
        for n_triplets in tqdm(self.n_triplets_list,
                            desc="Creating confusion matrix plots",
                            unit="plot"):
            self.one_confusion_matrix(n_triplets, height_cm, width_cm, panel_letter)
            plt.savefig(f"{confusion_matrices_folder}/confusion_matrix_{n_triplets}.pdf",
                    bbox_inches=None,  # Changed from 'tight' to None
                    dpi=600,
                    format='pdf')
            plt.close()

    def create_and_save_MRR_plot_for_all_models(self,
                                               width_cm: float,
                                               height_cm: float,
                                               panel_letter: str = None,
                                               gridspec_dict: dict = None,
                                               subplots_dict: dict = None):
        """
        Generates Mean Reciprocal Rank (MRR) bar plots for all models at all triplet counts.
        
        This method creates a set of publication-quality bar plot figures showing the Mean
        Reciprocal Rank (MRR) for all models at each triplet count. MRR is a measure of how
        highly a model ranks itself when it is the true data-generating model, providing
        insight into model identifiability. The figures are typically split by architecture
        groups for easier comparison.
        
        Parameters:
            width_cm (float): Width of each figure in centimeters.
            height_cm (float): Height of each figure in centimeters.
            panel_letter (str, optional): Letter label for the panel on each figure.
                                         Defaults to None.
            gridspec_dict (dict, optional): Dictionary with parameters for matplotlib.gridspec.GridSpec.
            subplots_dict (dict, optional): Dictionary defining subplot positions within GridSpec.
            
        Returns:
            None: Figures are saved to disk at:
                 {save_plot_folder}/mrr_plots/mrr_plot_{n_triplets}.pdf
                 for each n_triplets in n_triplets_list.
                 
        Notes:
            - This is a batch operation that creates multiple figures, one per triplet count.
            - Progress is displayed with a tqdm progress bar showing completion by triplet count.
            - MRR values range from 0 to 1, with higher values indicating better model identifiability.
            - Models are grouped by architecture type (typically in self.model_arch_dict).
            - Special formatting is applied to axis spines, with the first architecture subplot
              showing full axes and the second showing only the left spine.
        """
        # Create save folder if it doesn't exist
        save_folder = f"{self.save_plot_folder}/mrr_plots"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for n_triplets in tqdm(self.n_triplets_list, desc="Creating MRR plots", unit="plot"):
            # Create the Figure layout
            fig, axs = self.create_figure_layout(
                width_cm=width_cm,
                height_cm=height_cm,
                gridspec_dict=gridspec_dict,
                subplot_specifics_dict=subplots_dict
            )

            base_font_size = min(width_cm, height_cm) / 2.54 * 2  # Convert cm to inches and scale
            ticks_label_size = base_font_size * 0.8
            lines_width = 0.4

            # Add panel letter if provided
            if panel_letter:
                panel_letter_size = base_font_size * 1.4
                fig.text(0.02, 0.96, panel_letter, 
                        fontsize=panel_letter_size,
                        fontweight='bold',
                        transform=fig.transFigure)

            # Calculate MRR for each architecture type and model
            mrr_dict = {}
            for architecture in self.model_arch_dict.keys():
                mrr_dict[architecture] = {}
                for model in self.model_arch_dict[architecture].keys():
                    mrr = self.calculate_model_MRR(n_triplets)[model]
                    mrr_dict[architecture][model] = mrr

            # Draw the plots
            architecture_list = list(self.model_arch_dict.keys())
            for ax_name, ax in axs.items():
                if architecture_list[0] in ax_name:  # First architecture type (e.g., CNN)
                    self.draw_bar_plot(
                        axs[ax_name],
                        mrr_dict[architecture_list[0]],
                        show_ticks_labels=True,
                        show_ticks_y=True,
                        tick_label_size=ticks_label_size
                    )
                    ax.set_frame_on(True)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_linewidth(lines_width)
                    ax.spines['left'].set_linewidth(lines_width)
                    ax.tick_params(axis='both', which='major', length=2, width=lines_width)
                elif architecture_list[1] in ax_name:  # Second architecture type (e.g., Transformer)
                    self.draw_bar_plot(
                        axs[ax_name],
                        mrr_dict[architecture_list[1]],
                        show_ticks_labels=True,
                        tick_label_size=ticks_label_size
                    )
                    ax.set_frame_on(True)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(True)
                    ax.spines['left'].set_linewidth(lines_width)
                    ax.set_title(f"Mean Reciprocal Rank (MRR) {n_triplets}", fontsize=base_font_size)
                    ax.tick_params(axis='both', which='major', length=2, width=lines_width)
    
            # Save the figure
            plt.savefig(f"{save_folder}/mrr_plot_{n_triplets}.pdf",
                        dpi=600,
                        format='pdf')
            plt.close()
            
    def create_mean_accuracy_heatmap(self, n_triplets: int, width_cm: float, height_cm: float,
                                   save_path: str = None, title: str = None,
                                   show_colorbar: bool = True, panel_letter: str = None):
        """
        Creates a heatmap showing mean accuracy across simulations for each model pair.
        
        This method generates a publication-quality heatmap visualization showing the mean
        test accuracy for each reference model (columns) when predicting triplets generated
        by each data-generating model (rows). The heatmap provides insight into how well
        different models can approximate other models' behavior, with diagonal elements
        representing self-recovery accuracy.
        
        Parameters:
            n_triplets (int): Number of training triplets to analyze.
            width_cm (float): Width of the figure in centimeters.
            height_cm (float): Height of the figure in centimeters.
            save_path (str, optional): Path to save the figure. If None, uses default location
                                      under {save_plot_folder}/mean_accuracy_heatmaps/.
            title (str, optional): Title for the heatmap. Defaults to None.
                                  Note: Title display is deprecated and will be ignored.
            show_colorbar (bool): Whether to show the colorbar. Defaults to True.
            panel_letter (str, optional): Letter label for the panel (e.g., 'A', 'B').
                                         Defaults to None.
            
        Returns:
            pd.DataFrame: The DataFrame containing mean accuracy values for all model pairs.
            
        Notes:
            - The method automatically saves the accuracy data as a CSV file for reproducibility.
            - A custom colormap is used with special emphasis on the 0.5-0.67 range,
              which is important for highlighting differences near chance level.
            - Diagonal elements (representing self-recovery) are bolded in the heatmap annotation.
            - Model names are converted to formal display names using go_formal.
        """
        # Convert dimensions to inches
        width_inches = width_cm / 2.54
        height_inches = height_cm / 2.54

        # Calculate base font sizes (slightly adjusted for better readability)
        base_font_size = min(width_inches, height_inches) * 1.4 # Slightly smaller base
        axis_label_size = 10 # Update to use constant font size
        tick_label_size = 5 # Update to use constant font size
        colorbar_fontsize = 6 # Update to use constant font size

        # Create an empty DataFrame to store the mean accuracies
        mean_acc_df = pd.DataFrame(
            index=self.data_generating_models_list,
            columns=self.reference_models_list,
            dtype=float  # Ensure DataFrame has numeric dtype
        ).fillna(0.0)  # Fill with zeros initially

        # For each data generating model and reference model pair
        for dgm in self.data_generating_models_list:
            results_dict = self.get_model_simulations_results(dgm, n_triplets)

            # For each reference model, calculate the mean accuracy across all simulations
            for ref_model in self.reference_models_list:
                # Get accuracies where this reference model was used
                accuracies = []
                for sim_idx, sim_results in results_dict.items():
                    if ref_model in sim_results:
                        accuracies.append(sim_results[ref_model])

                # Calculate mean if accuracies exist, otherwise use 0
                if accuracies:
                    mean_acc = float(np.mean(accuracies))
                    mean_acc_df.loc[dgm, ref_model] = mean_acc
        # Save mean_acc_df to csv
        # Make sure the folder exists
        
        os.makedirs(os.path.dirname(f"{self.save_plot_folder}mean_accuracy_heatmaps/mean_accuracy_heatmap_{n_triplets}.csv"),exist_ok=True)
        mean_acc_df.to_csv(f"{self.save_plot_folder}mean_accuracy_heatmaps/mean_accuracy_heatmap_{n_triplets}.csv",index=True)
        # Create figure with constrained layout to help with spacing
        fig, ax = plt.subplots(figsize=(width_inches, height_inches), constrained_layout=True)

        # Get formal names for models
        formal_index = [self.go_formal(model) for model in mean_acc_df.index]
        formal_columns = [self.go_formal(model) for model in mean_acc_df.columns]

        # Create a custom colormap emphasizing the 0.5-0.67 range
        colors = ['#a50026', # Dark Red at 0.0
                  '#f46d43', # Orange-Red at 0.3
                  '#ffffbf', # Pale Yellow at 0.5 (neutral)
                  '#1a9850', # More Saturated Green at 0.67 <--- Sharper transition here
                  '#006837'] # Dark Green at 1.0
        nodes = [0.0, 0.3, 0.5, 0.67, 1.0]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_RdYlGn_sharp", list(zip(nodes, colors)) # New name
        )
        # Use standard normalization
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

        # Create a custom formatter function to remove leading zeros
        def format_value(val):
            if val == 0:
                return '.00'
            elif val == 1:
                return '1.00'
            else:
                # Format to 2 decimal places, remove leading zero if present
                return f'{val:.2f}'.lstrip('0')


        # Create the heatmap
        hm = sns.heatmap(
            mean_acc_df.values,  # Use numeric values directly
            annot=True,
            fmt='.2f', # Use standard format initially, will be reformatted
            annot_kws={'size': tick_label_size * 1.0}, # Make annotations slightly bigger (same as ticks)
            cmap=custom_cmap, # Use the custom colormap
            norm=norm, # Apply standard normalization
            vmin=0.0, # Explicitly set vmin/vmax
            vmax=1.0,
            linewidths=0.1,
            linecolor='gray',
            cbar=show_colorbar,
            cbar_kws={
                'shrink': 0.6,
                'aspect': 30,
                'label': 'Mean Accuracy',
                'format': '%.2f'  # Use standard format for colorbar
            },
            square=True,
            ax=ax
        )

        # Update annotation texts using the custom formatter
        for text in hm.texts:
            text.set_text(format_value(float(text.get_text())))

        # Bold the diagonal annotations
        n = len(mean_acc_df) # Number of rows/columns
        for i in range(n):
            idx = i * n + i # Calculate index in flattened list (assuming row-major)
            if idx < len(hm.texts):
                hm.texts[idx].set_fontweight('bold')

        # Manually set the tick labels after creating the heatmap
        ax.set_xticklabels(formal_columns, rotation=45, ha='right', fontsize=5)
        # Reduce potential overlap for y-axis ticks
        ax.set_yticklabels(formal_index, rotation=0, fontsize=5, va='center') # Adjust vertical alignment

        # Set labels with increased padding
        ax.set_xlabel("Reference Model", fontsize=10, fontweight='bold', labelpad=10) # Increased padding
        ax.set_ylabel("Data Generating Model", fontsize=10, fontweight='bold', labelpad=10) # Increased padding

        # Adjust colorbar font size if shown
        if show_colorbar:
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=5)
            cbar.set_label('Mean Accuracy', fontsize=6, labelpad=5) # Update legend font size

        # Add panel letter if provided
        if panel_letter:
            # Adjust position slightly if needed due to layout changes
            plt.text(
                -0.1, 1.05, panel_letter,
                transform=ax.transAxes,
                fontsize=6, # Use consistent panel letter font size
                fontweight='bold'
            )

        # Save the figure if path is provided
        if save_path:
            # Use bbox_inches='tight' carefully with constrained_layout
            # Make sure the folder exists
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, format='pdf')
        else:
            # Create default save path
            default_save_folder = f"{self.save_plot_folder}/mean_accuracy_heatmaps"
            if not os.path.exists(default_save_folder):
                os.makedirs(default_save_folder)
            plt.savefig(f"{default_save_folder}/mean_accuracy_heatmap_{n_triplets}.pdf",
                        dpi=300, format='pdf')

        plt.close(fig) # Close the specific figure object

        # Return the DataFrame for further analysis
        return mean_acc_df

def create_colors_list(n_colors, start=0, rot=0.5):
    """
    Creates a list of distinct colors using seaborn's cubehelix color palette.
    
    This function generates a list of distinct, visually pleasing colors suitable for
    differentiating between multiple models in visualization plots. It leverages the
    cubehelix color system which creates perceptually uniform color palettes that work
    well for scientific visualization.
    
    Parameters:
        n_colors (int): Number of distinct colors to generate.
        start (float): Starting position in the color wheel (0-3). Different start values
                      create different hue ranges. Defaults to 0.
        rot (float): Number of rotations through the color wheel. Higher values increase
                    the range of hues used. Defaults to 0.5.
    
    Returns:
        list: List of hex color codes that can be used for matplotlib plots.
        
    Notes:
        - Different start values are useful for creating distinct color groups for
          different types of models (e.g., supervised, self-supervised, multimodal).
        - The function uses fixed dark (0.3) and light (0.85) values to ensure good
          visibility against both light and dark backgrounds.
    """
    
    palette = sns.cubehelix_palette(
        n_colors= n_colors,
        start=start,
        rot=rot,
        dark=0.3,
        light=0.85
    )
    
    colors_list = [mcolors.rgb2hex(color) for color in palette]
    return colors_list

##############################
# Main Execution Script
##############################
@hydra.main(version_base=None, config_path='../scripts_configurations',config_name="figures_1_and_3")
def main(cfg: DictConfig):
    """
    Main function for generating Figures 1 and 3 for the paper.
    
    This function follows the SAME pattern as create_Figure_2.py for consistency.
    It uses Hydra configuration management to load parameters and settings,
    derives model selection and ordering from cfg.models_objective, generates
    colors using cubehelix palettes, and creates figures using ModelRecoveryVisualizer.
    
    Parameters:
        cfg (DictConfig): Hydra configuration object containing:
                         - experiment_name: Name of experiment (selects which config to use)
                         - Per-experiment configuration (accessed via cfg[experiment_name]):
                           * db_path: Path to simulation database
                           * use_additional_models: Whether experiment includes all 30 models
                           * N_simulations: Number of simulation runs
                           * n_triplets: Training set sizes
                         - models: List of 20 base models (from shared_args)
                         - new_models: List of 10 additional models (from shared_args)
                         - create_figure_1: Boolean flag to generate Figure 1 (default: True)
                         - create_figure_3: Boolean flag to generate Figure 3 (default: True)
                         - models_objective: Dict mapping models to training objectives
                         - formal_names_dict: Mapping from internal to publication names
                         - architecture_dict: Mapping from models to CNN/Transformer
                         - Figures_settings_dict: Detailed settings for each figure
    
    Returns:
        None: The figures are saved to disk at Plots/{experiment_name}/
        
    Notes:
        - use_additional_models its an EXPERIMENT-SPECIFIC flag read from cfg[experiment_name]
        - models_list is derived based on use_additional_models flag
        - It must match what models were included in the simulation database
        - Model ordering is derived from cfg.models_objective dictionary keys
        - Color counts are computed dynamically from models_objective values
        - Colors are assigned using cubehelix palettes (same as Figure 2)
        - Each experiment_name points to a different SQLite database
        - Figure generation can be controlled independently via create_figure_1/create_figure_3
    """
    # ==========================================================================
    # Step 1: Model Selection Based on Experiment Configuration
    # ==========================================================================
    # Get experiment configuration first to access use_additional_models
    experiment_name = cfg.experiment_name
    experiment_CONFIG_dict = cfg[experiment_name]
    
    # Read use_additional_models from the experiment configuration
    # (not from global config - it's experiment-specific!)
    use_additional_models = experiment_CONFIG_dict.get('use_additional_models', False)
    
    # Determine which models to use based on experiment's flag
    models_list = (
        cfg.models if not use_additional_models 
        else cfg.models + cfg.new_models
    )
    
    # ==========================================================================
    # Step 2: Filter Models Objective Dictionary
    # ==========================================================================
    # Keep only the objectives for models we're using
    models_objective_dict = {
        key: value for key, value in cfg.models_objective.items() 
        if key in models_list
    }
    
    # ==========================================================================
    # Step 3: Create Ordered Model List from Objectives Dictionary
    # ==========================================================================
    # Order is determined by the keys() of models_objective dict
    # This automatically groups models by objective type
    models_ordered = list(models_objective_dict.keys())
    
    # ==========================================================================
    # Step 4: Count Objectives Dynamically (Same as Figure 2)
    # ==========================================================================
    # Count how many models of each type we have
    from collections import Counter
    objectives_count = Counter(models_objective_dict.values())
    
    # Map to standardized keys (Figure 2 pattern)
    objectives_count_dict = {
        "supervised": objectives_count.get("Supervised", 0),
        "self_supervised": objectives_count.get("Self-supervised", 0),
        "multimodal": objectives_count.get("Image & Text Alignment", 0)
    }
    
    # ==========================================================================
    # Step 5: Filter Formal Names and Architecture Dictionaries
    # ==========================================================================
    # Keep only entries for models we're using
    formal_models_names_dict = {
        key: value for key, value in cfg.formal_names_dict.items() 
        if key in models_list
    }
    
    architecture_dict = {
        key: value for key, value in cfg.architecture_dict.items() 
        if key in models_list
    }
    
    # ==========================================================================
    # Step 6: Create Color Palettes (Exact Same Logic as Figure 2)
    # ==========================================================================
    models_colors_dict = {}
    N_models = len(models_ordered)
    
    # Generate color palettes for each objective type
    supervised_colors_list = create_colors_list(N_models, start=0, rot=0.5)
    self_supervised_colors_list = create_colors_list(N_models, start=1, rot=0.5)
    multimodal_colors_list = create_colors_list(N_models, start=2, rot=0.5)
    
    # ==========================================================================
    # Step 7: Assign Colors Based on Position (Same as Figure 2)
    # ==========================================================================
    N_supervised = objectives_count_dict["supervised"]
    N_self_supervised = objectives_count_dict["self_supervised"]
    
    # Assign colors based on position in models_ordered list
    for i, model in enumerate(models_ordered):
        if i < N_supervised:
            models_colors_dict[model] = supervised_colors_list[i]
        elif i < N_supervised + N_self_supervised:
            models_colors_dict[model] = self_supervised_colors_list[i-N_supervised]
        else:
            models_colors_dict[model] = multimodal_colors_list[i-N_supervised-N_self_supervised]
    
    # ==========================================================================
    # Step 8: Populate Gradient Legend (Same as Figure 2)
    # ==========================================================================
    legend_gradient_dict = {}
    for key in ["Supervised", "Self-supervised", "Image & Text Alignment"]:
        if key == "Supervised":
            legend_gradient_dict[key] = supervised_colors_list[:N_supervised]
        elif key == "Self-supervised":
            legend_gradient_dict[key] = self_supervised_colors_list[:N_self_supervised]
        else:
            legend_gradient_dict[key] = multimodal_colors_list[:objectives_count_dict["multimodal"]]
    
    # ==========================================================================
    # Step 9: Extract Experiment Configuration (Already Done in Step 1)
    # ==========================================================================
    # We already extracted these in Step 1, just get remaining parameters
    results_db_path = experiment_CONFIG_dict["db_path"]
    N_simulations = experiment_CONFIG_dict["N_simulations"]
    simulations_list = list(range(N_simulations)) if N_simulations is not None else None
    n_triplets = experiment_CONFIG_dict["n_triplets"]
    
    # Use models_ordered as the data-generating models list
    data_generating_models_list = models_ordered
    
    # ==========================================================================
    # Step 10: Create Visualizer Instance
    # ==========================================================================
    visualizer = ModelRecoveryVisualizer(
        db_path=results_db_path,
        data_generating_models_list=data_generating_models_list,
        simulations_list=simulations_list,
        n_triplets_list=n_triplets,
        reference_models_list=data_generating_models_list,
        models_color_dict=models_colors_dict,
        save_plot_folder=f"Plots/{experiment_name}",
        formal_names_dict=formal_models_names_dict,
        models_order_as_colors_dict=list(models_colors_dict.keys())
    )

    # ==========================================================================
    # Step 11: Extract Figure Settings and Generate Figures
    # ==========================================================================
    Figures_settings_dict = cfg.Figures_settings_dict
    
    # Figure 1 parameters
    fig_1_params = {   
            "width_cm": cfg.width_size_figure,
            "height_cm": Figures_settings_dict["Figure_1"]["height_cm"],
            "panel_letters_dict": Figures_settings_dict["Figure_1"]["panel_letters_dict"],
            "gridspec_dict": Figures_settings_dict["Figure_1"]["gridspec_dict"],
            "subplots_dict": Figures_settings_dict["Figure_1"]["subplots_dict"],   
            "conf_size_dict": Figures_settings_dict["Figure_1"]["conf_size_dict"],
            "line_plot_letters_labels_dict": Figures_settings_dict["Figure_1"]["line_plot_letters_labels_dict"],
            "axis_label_size": cfg.axis_labels_fontsize,
            "tick_label_size": cfg.tick_labels_fontsize,
            "panel_letter_size": cfg.panel_letter_fontsize,
            "CI": Figures_settings_dict["Figure_1"]["CI"]
    }
    
    # Figure 3 parameters
    fig_3_params = {
        "width_cm": cfg.width_size_figure,
        "height_cm": Figures_settings_dict["Figure_3"]["height_cm"],
        "gridspec_dict": Figures_settings_dict["Figure_3"]["gridspec_dict"],
        "subplots_dict": Figures_settings_dict["Figure_3"]["subplots_dict"],
        "training_triplets_N": Figures_settings_dict["Figure_3"]["training_triplets_N"],
        "panel_letters_dict": Figures_settings_dict["Figure_3"]["panel_letters_dict"],
        "axis_label_size": cfg.axis_labels_fontsize,
        "tick_label_size": cfg.tick_labels_fontsize,
        "panel_letter_size": cfg.panel_letter_fontsize
    }   

    # Generate figures
    print(f"\n{'='*60}")
    print(f"CREATING FIGURES 1 AND 3")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Database: {results_db_path}")
    print(f"Models: {len(models_ordered)} ({'all 30' if use_additional_models else '20 base'})")
    print(f"  use_additional_models: {use_additional_models} (from experiment config)")
    print(f"Simulations: {N_simulations}")
    print(f"Training sizes: {len(n_triplets)} levels ({min(n_triplets)} to {max(n_triplets)} triplets)")
    print(f"Output folder: Plots/{experiment_name}/")
    print(f"Figure 1: {'Enabled' if cfg.get('create_figure_1', True) else 'Disabled'}")
    print(f"Figure 3: {'Enabled' if cfg.get('create_figure_3', True) else 'Disabled'}")
    print(f"{'='*60}\n")
    
    # Generate Figure 1 if enabled
    if cfg.get('create_figure_1', True):
        print("Generating Figure 1...")
        visualizer.create_figure_1(**fig_1_params)
        print(f"✓ Figure 1 saved: Plots/{experiment_name}/Figure_1.pdf\n")
    else:
        print("Skipping Figure 1 (disabled in configuration)\n")
    
    # Generate Figure 3 if enabled
    if cfg.get('create_figure_3', True):
        print("Generating Figure 3...")
        visualizer.create_figure_3(**fig_3_params)
        print(f"✓ Figure 3 saved: Plots/{experiment_name}/Figure_3.pdf\n")
    else:
        print("Skipping Figure 3 (disabled in configuration)\n")
    
    print(f"{'='*60}")
    print(f"FIGURES GENERATION COMPLETE")
    print(f"{'='*60}")
    generated_figures = []
    if cfg.get('create_figure_1', True):
        generated_figures.append(f"Figure 1: Plots/{experiment_name}/Figure_1.pdf")
    if cfg.get('create_figure_3', True):
        generated_figures.append(f"Figure 3: Plots/{experiment_name}/Figure_3.pdf")
    
    if generated_figures:
        for fig in generated_figures:
            print(f"  {fig}")
    else:
        print("  No figures generated (all disabled)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
