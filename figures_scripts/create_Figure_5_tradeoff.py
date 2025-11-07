"""
===============================================================================
Figure 5: Transformation Flexibility vs. Predictive Accuracy Tradeoff
===============================================================================

This script generates Figure 5 of the paper, which visualizes the fundamental
tradeoff between transformation flexibility and model identifiability in
behavioral alignment. The figure demonstrates that more flexible alignment
metrics achieve higher predictive accuracy but lower model recovery accuracy.

Scientific Context:
------------------
When aligning neural network representations to behavioral data, we can use
different transformation flexibilities (W matrices with varying constraints).
This raises a key question: Does increased flexibility improve prediction at
the cost of model identifiability?

Figure Overview:
---------------
The figure is a scatter plot showing:
- X-axis: Model recovery accuracy (% correct model identifications)
- Y-axis: Mean predictive accuracy (performance on held-out triplets)
- Colors: Constraint types (coolwarm gradient from rigid to flexible)
- Markers: Training set sizes (different shapes)
- Lines: Connect points of same constraint type (dotted)



Constraint Types (from rigid to flexible):
------------------------------------------
1. zero_shot: Identity matrix (W = I), no learning
   - Lowest flexibility, highest identifiability
   - Baseline showing raw model similarity
   
2. diagonal: Element-wise scaling only
   - W ∈ Diag_p(ℝ) - only p parameters
   - Moderate flexibility, good identifiability
   
3. Rectangular_30: Low-rank projection
   - W ∈ ℝ^(p×30) - reduces to 30 dimensions
   - Medium flexibility, medium identifiability
   
4. full_W_L1: Full matrix with L1 regularization
   - W ∈ ℝ^(p×p) with sparse penalty
   - High flexibility, lower identifiability
   
5. full_W: Full unconstrained matrix
   - W ∈ ℝ^(p×p) with eye_distance regularization
   - Maximum flexibility, lowest identifiability

Implementation Details:
----------------------
- Uses Hydra for configuration (scripts_configurations/Figure_5.yaml)
- Loads pre-computed results from figure_5_analysis.py
- Creates publication-quality PDF with precise GridSpec layout
- Three separate legends: flexibility, regularization, train set size
- Smart tick generation focuses on data-dense regions

Data Flow:
---------
1. Load final_results_df.csv (aggregated metrics)
2. Filter to specified constraint types
3. Create scatter plot with color/marker encoding
4. Add connecting lines per constraint type
5. Position three legends using GridSpec
6. Save high-resolution PDF

Configuration:
-------------
All parameters controlled via scripts_configurations/Figure_5.yaml:
- constraint_types_to_plot: Which constraints to include
- figure_dimensions: Width/height in cm
- font_sizes: Axis labels, ticks, legend
- plot_styling: Marker sizes, line styles, colors
- output: Save location and format

Usage:
------
# Generate with default settings
python Figure_5_tradeoff.py

# Show only endpoint constraints
python Figure_5_tradeoff.py constraint_types_to_plot="['zero_shot','full_W']"

# Enable error bars
python Figure_5_tradeoff.py show_error_bars=true

# Custom output
python Figure_5_tradeoff.py output.filename="custom_figure.pdf"

Requirements:
------------
- Results/flexibility_accuracy/final_results_df.csv (from figure_5_analysis.py)
- matplotlib, numpy, pandas, sklearn (for KMeans clustering)
- Hydra for configuration management

Output:
-------
Plots/Figure_5.pdf - Publication-quality PDF figure

===============================================================================
"""

import os
import sys
from pathlib import Path
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(
    PARENT_DIR
)  # We want the script to work with relative paths as if script is one level up
sys.path.append(str(PARENT_DIR))
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from typing import List
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.text import Text
import hydra
from omegaconf import DictConfig, OmegaConf

# =============================================================================
# Matplotlib Configuration
# =============================================================================

# Configure matplotlib for publication-quality output
# Uses Arial for sans-serif text and Computer Modern for mathematical equations
mpl.rcParams.update({
    # Use Arial for all text (standard for scientific publications)
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    # Use Computer Modern for math (LaTeX-quality equations)
    'mathtext.fontset': 'cm',
})


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_data(file_path, constraints_list: Optional[List[str]] = None):
    """
    Load aggregated results from CSV file and optionally filter by constraints.
    
    This function loads the pre-computed results from figure_5_analysis.py,
    which contains test accuracies, model recovery accuracies, and metadata
    for different constraint types and training set sizes.
    
    Parameters
    ----------
    file_path : str
        Path to the final_results_df.csv file containing aggregated metrics.
        Expected to be in Results/flexibility_accuracy/ directory.
        
    constraints_list : List[str], optional
        List of constraint type names to include in the output.
        If None, all constraints in the file are included.
        Example: ['zero_shot', 'diagonal', 'full_W']
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - constraint: str, transformation constraint type
        - train_set_size: int, number of training triplets
        - test_acc: float, mean test accuracy across models
        - test_acc_std: float, standard deviation of test accuracy
        - model_recovery_acc: float, percentage of correct identifications
        - max_acc_model: str, name of best-performing model
        - max_acc_model_acc: float, accuracy of best model
        - min_acc_model: str, name of worst-performing model
        - min_acc_model_acc: float, accuracy of worst model
        
    Raises
    ------
    FileNotFoundError
        If the specified file doesn't exist. This likely means
        figure_5_analysis.py hasn't been run yet.
        
    Notes
    -----
    The input CSV is generated by running:
        python analysis_scripts/figure_5_analysis.py
    
    Each row represents one (constraint_type, training_size) combination.
    The test_acc values are averaged across all neural network models,
    while model_recovery_acc comes from simulation database queries.
    
    Example
    -------
    >>> data = load_data("Results/flexibility_accuracy/final_results_df.csv")
    >>> print(data[data['constraint'] == 'full_W'].head())
    >>> 
    >>> # Filter to specific constraints
    >>> data = load_data("Results/flexibility_accuracy/final_results_df.csv",
    ...                  constraints_list=['zero_shot', 'full_W'])
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File {file_path} not found.\n"
            f"Please run 'python analysis_scripts/figure_5_analysis.py' first to generate the data."
        )
    
    data = pd.read_csv(file_path)
    
    # Filter by constraints if specified
    if constraints_list is not None:
        data = data[data['constraint'].isin(constraints_list)]
        
        # Warn if some requested constraints are not in the data
        missing = set(constraints_list) - set(data['constraint'].unique())
        if missing:
            print(f"Warning: Requested constraints not found in data: {missing}")
    
    return data


def setup_figure(width_cm, height_cm, aspect_ratio_data=None):
    """
    Set up the figure with the specified dimensions and GridSpec layout.
    
    This function creates the base figure and defines a precise grid layout
    for positioning the main plot and three separate legends. The layout is
    optimized for publication-quality single-column figures.
    
    Parameters
    ----------
    width_cm : float
        Width of the figure in centimeters (typically half_width_size for 
        single-column format = 6.985 cm)
    height_cm : float
        Height of the figure in centimeters (adjusted for aspect ratio,
        typically ~6 cm)
    aspect_ratio_data : tuple, optional
        A tuple (x_range, y_range) of the data extent. Currently not used
        but reserved for future aspect ratio calculations.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure with specified dimensions
    gs : matplotlib.gridspec.GridSpec
        The GridSpec object defining the layout grid with 5 rows × 9 columns
        
    Notes
    -----
    The GridSpec layout is structured as:
    
    Row 0: Top margin (0.2 units)
    Row 1: Main scatter plot (4 units - largest area)
    Row 2: Spacing between plot and legends (1 unit)
    Row 3: Three legends side-by-side (1.1 units)
    Row 4: Bottom margin (0.3 units)
    
    Columns are distributed to accommodate:
    - Main plot spanning columns 1-7
    - Flexibility legend in column 2
    - Regularization legend in column 4
    - Train set size legend in column 6
    - Margins in columns 0 and 8
    
    The layout uses hspace=0 and wspace=0 for precise control, with all
    spacing handled by the height_ratios and width_ratios.
    
    Example
    -------
    >>> fig, gs = setup_figure(width_cm=6.985, height_cm=6)
    >>> ax_main = fig.add_subplot(gs[1, 1:8])  # Main plot
    >>> ax_legend1 = fig.add_subplot(gs[3, 2])  # First legend
    """
    # Convert cm to inches (matplotlib uses inches internally)
    # 1 inch = 2.54 cm
    width_inches = width_cm / 2.54
    height_inches = (height_cm + 1) / 2.54  # +1 cm for legend space
    
    # Create figure with exact dimensions
    fig = plt.figure(figsize=(width_inches, height_inches))

    # Define width ratios for 9 columns
    # Optimized for main plot + three legends side-by-side
    width_ratios = [0.8, 0.6, 0.8, 0.3, 0.8, 0.8, 0.8, 0.6, 0.1]
    
    # Define height ratios for 5 rows
    # Main plot gets most space (4 units), legends get 1.1 units
    height_ratios = [0.2, 4, 1, 1.1, 0.3]

    # Create gridspec: 5 rows × 9 columns
    gs = gridspec.GridSpec(
        5, 9, 
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0,  # No additional horizontal spacing between cells
        wspace=0   # No additional vertical spacing between cells
    )
    
    return fig, gs


def create_smart_ticks(values, min_val=0, n_clusters=3, ticks_per_cluster=3, min_ticks=5, max_ticks=10):
    """
    Create smart tick positions that focus on dense regions of data.
    
    This function identifies clusters in the data values and places more tick marks
    in regions with higher data density. This improves readability of the plot by
    ensuring important regions have sufficient tick resolution.
    
    The algorithm uses KMeans clustering to identify where data points concentrate,
    then distributes ticks both uniformly across the range and densely around
    cluster centers.
    
    Parameters
    ----------
    values : array-like
        The data values to create ticks for (e.g., model recovery accuracies)
    min_val : float, default=0
        Minimum value for the ticks (usually 0 for percentage data)
    n_clusters : int, default=3
        Number of clusters to identify in the data using KMeans
    ticks_per_cluster : int, default=3
        Number of additional ticks to place within each dense cluster region
    min_ticks : int, default=5
        Minimum number of ticks to create (ensures basic readability)
    max_ticks : int, default=10
        Maximum number of ticks to create (prevents overcrowding)
    
    Returns
    -------
    ticks : np.ndarray
        Sorted array of tick positions, guaranteed to be unique and within
        the data range
        
    Notes
    -----
    The function uses KMeans clustering to identify dense regions in the data.
    If there are too few unique values for meaningful clustering, it falls back
    to simpler methods based on the data range.
    
    Algorithm steps:
    1. Check if enough unique values exist for clustering
    2. Apply KMeans to find cluster centers (dense regions)
    3. Create base ticks uniformly across range
    4. Add extra ticks around each cluster center
    5. Remove duplicates and limit to max_ticks
    
    Example
    -------
    >>> values = np.array([0.5, 0.51, 0.52, 0.7, 0.71, 0.9])
    >>> ticks = create_smart_ticks(values, min_val=0, n_clusters=3)
    >>> print(ticks)
    array([0.  , 0.5 , 0.51, 0.52, 0.7 , 0.71, 0.9 ])
    # More ticks appear around 0.5 and 0.7 where data clusters
    """
    # Convert to numpy array and ensure 1D
    values = np.asarray(values).flatten()
    
    # Ensure we have enough unique values for clustering
    unique_values = np.unique(values)
    if len(unique_values) < n_clusters:
        n_clusters = len(unique_values)

    if len(unique_values) <= min_ticks:
        # If very few unique values, just use those plus min_val
        ticks = np.sort(np.append(unique_values, min_val))
        return np.unique(ticks)
    
    # Reshape for KMeans (requires 2D input)
    X = values.reshape(-1, 1)
    
    # Apply KMeans to find dense regions
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    cluster_centers = kmeans.cluster_centers_.flatten()
    
    # Create ticks focused on cluster centers
    ticks = []
    
    # Always start from min_val (usually 0)
    ticks.append(min_val)
    
    # Add regular ticks across the whole range
    max_val = np.amax(values)
    regular_ticks = np.linspace(min_val, max_val, min_ticks)
    ticks.extend(regular_ticks)
    
    # For each cluster center, add more ticks in dense regions
    for center in cluster_centers:
        # Create ticks around center with width proportional to overall range
        width = (max_val - min_val) / (min_ticks * 2)
        cluster_ticks = np.linspace(max(min_val, center - width), 
                                     min(max_val, center + width), 
                                     ticks_per_cluster)
        ticks.extend(cluster_ticks)
    
    # Remove duplicates and sort
    ticks = np.unique(ticks)
    
    # If too many ticks, reduce them by sampling uniformly
    if len(ticks) > max_ticks:
        indices = np.round(np.linspace(0, len(ticks) - 1, max_ticks)).astype(int)
        ticks = ticks[indices]
    
    return ticks

def create_main_plot(ax, data, ax_title_fontsize, tick_fontsize, marker_size, legend_marker_size, swap_axes=True, show_grid=False,top_acc:bool = False ,error_bar:bool = False):
    """
    Create the main scatter plot with connected points showing flexibility-accuracy tradeoff.
    
    This function creates a scatter plot visualizing the relationship between model recovery
    accuracy and test accuracy across different model constraint types and training set sizes.
    Points are connected by constraint type and differentiated by color and marker according
    to their flexibility ranking and training set size.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    data : pd.DataFrame
        The data to plot, must contain columns:
        - 'constraint': constraint type
        - 'train_set_size': training set size
        - 'model_recovery_acc': model recovery accuracy
        - 'test_acc': test accuracy (or 'max_acc_model_acc' if top_acc=True)
        - 'test_acc_std': standard deviation of test accuracy (optional)
    ax_title_fontsize : int
        Font size for axis titles
    tick_fontsize : int
        Font size for tick labels
    marker_size : int
        Size of the markers in the plot
    legend_marker_size : int
        Size of the markers in the legend
    swap_axes : bool, default=True
        If True, swap X and Y axes (model_recovery_acc on X)
        If False, use original orientation (test_acc on X)
    show_grid : bool, default=False
        Whether to show grid lines
    top_acc : bool, default=False
        If True, use max_acc_model_acc instead of test_acc
    error_bar : bool, default=False
        If True, add error bars using test_acc_std
        
    Returns
    -------
    constraint_colors : dict
        Dictionary mapping constraints to colors
    train_size_markers : dict
        Dictionary mapping train set sizes to markers
    unique_constraints : list
        List of unique constraints sorted by flexibility
    unique_train_sizes : array
        Array of unique train set sizes
    constraint_latex_map : dict
        Dictionary mapping constraints to LaTeX formatted names
        
    Notes
    -----
    The function implements a flexibility ranking for constraint types, where:
    - zero_shot (least flexible, rank 0)
    - diagonal (rank 1)
    - Rectangular_30 (rank 2)
    - full_W (most flexible, rank 3)
    
    Colors follow a cool-to-warm gradient based on flexibility. Connecting lines
    use dotted style and the same color as their corresponding constraints.
    """
    # Define flexibility ranking (lower number means less flexible)
    flexibility_ranking = {
        'zero_shot': 0,  # Least flexible
        'diagonal': 1,
        #'Rectangular_10': 2,
        'Rectangular_30': 2,
        "full_W_L1": 3,
        'full_W': 4  # Most flexible
    }
    
    # Get unique values for coloring and markers
    unique_constraints = sorted(data['constraint'].unique(), key=lambda x: flexibility_ranking.get(x, 999))
    unique_train_sizes = data['train_set_size'].unique()
    # Remove Rectangular_10 from the unique constraints list
    unique_constraints = [constraint for constraint in unique_constraints if constraint != 'Rectangular_10']
    
    # Create cold-to-warm color gradient based on flexibility
    cmap = mpl.colormaps.get_cmap('coolwarm')
    constraint_colors = {constraint: cmap(flexibility_ranking[constraint] / (len(unique_constraints)-1)) for i, constraint in enumerate(unique_constraints)}
    
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    train_size_markers = {size: markers[i % len(markers)] for i, size in enumerate(unique_train_sizes)}
    
    # The axes is already created and passed to us
    # Remove top and right spines
    ax.spines['top'].set_visible(False)   
    ax.spines['right'].set_visible(False)
    
    # Sort data for connecting lines
    for constraint in unique_constraints:
        constraint_data = data[data['constraint'] == constraint].sort_values('model_recovery_acc')
        
        # Group by constraint for connecting lines
        points_x = []
        points_y = []
        
        # Plot each data point for this constraint
        for train_size in unique_train_sizes:
            subset = constraint_data[constraint_data['train_set_size'] == train_size]
            
            if not subset.empty:
                if swap_axes:
                    # Swapped axes - model_recovery_acc as X and test_acc as Y
                    x = subset['model_recovery_acc'].values
                    y = subset['test_acc' if not top_acc else 'max_acc_model_acc'].values
                    # Get error if available
                    yerr = subset['test_acc_std'].values if error_bar and 'test_acc_std' in subset.columns else None
                else:
                    # Original axes - test_acc as X and model_recovery_acc as Y
                    x = subset['test_acc' if not top_acc else 'max_acc_model_acc'].values
                    y = subset['model_recovery_acc'].values
                    # Get error if available
                    xerr = subset['test_acc_std'].values if error_bar and 'test_acc_std' in subset.columns else None
                
                # Add to points collection for line
                points_x.extend(x)
                points_y.extend(y)
                
                
                # Plot the point
                ax.scatter(
                    x, y,
                    color=constraint_colors[constraint],
                    marker=train_size_markers[train_size],
                    s=marker_size,
                    label=f"{constraint}, {train_size}"
                )
                # Plot the point with error bars if requested
                if error_bar and 'test_acc_std' in subset.columns:
                    if swap_axes:
                        assert yerr is not None, "yerr should not be None when error_bar is True"
                        ax.errorbar(
                            x, y,
                            yerr=yerr/2,
                            fmt='none',  # No connecting line
                            ecolor=constraint_colors[constraint],
                            elinewidth=1,
                            capsize=3
                        )
                    else:
                        ax.errorbar(
                            x, y,
                            xerr=xerr,
                            fmt='none',  # No connecting line
                            ecolor=constraint_colors[constraint],
                            elinewidth=1,
                            capsize=3
                        )
        
        # Connect points of the same constraint with dotted lines
        if len(points_x) > 1:
            # Sort points to connect in order of x-coordinate
            points = sorted(zip(points_x, points_y))
            sorted_x, sorted_y = zip(*points)
            
            # Plot the connecting line
            ax.plot(
                sorted_x, sorted_y,
                color=constraint_colors[constraint],
                linestyle=':', 
                linewidth=1,
                alpha=0.7,
                zorder=0  # Ensure lines are drawn below points
            )
    
    # Set axis labels based on swap_axes
    if swap_axes:
        ax.set_xlabel('model recovery accuracy', fontsize=ax_title_fontsize/2,fontweight='bold')
        ax.set_ylabel('mean predictive accuracy' if not top_acc else 'top model predictive accuracy', fontsize=ax_title_fontsize/2,fontweight='bold')
        
        x_values = data['model_recovery_acc'].values
        y_values = data['test_acc' if not top_acc else 'max_acc_model_acc'].values
    else:
        ax.set_xlabel('mean predictive accuracy' if not top_acc else 'top model predictive accuracy', fontsize=ax_title_fontsize/2,fontweight='bold')
        ax.set_ylabel('model recovery accuracy', fontsize=ax_title_fontsize/2,fontweight='bold')

        x_values = data['test_acc' if not top_acc else 'max_acc_model_acc'].values
        y_values = data['model_recovery_acc'].values
    
    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Add grid if requested
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    else:
        ax.grid(False)
    
    # Create smart ticks focused on dense regions
    x_ticks = create_smart_ticks(x_values, min_val=0, 
                                n_clusters=min(4, len(unique_constraints)), 
                                ticks_per_cluster=3, min_ticks=6, max_ticks=12)
    
    y_ticks = create_smart_ticks(y_values, min_val=0, 
                                n_clusters=min(4, len(unique_constraints)), 
                                ticks_per_cluster=3, min_ticks=6, max_ticks=12)
    
    max_x_tick = max(x_ticks)
    max_y_tick = max(y_ticks)
    min_x_tick = min(x_values)
    min_y_tick = min(y_values)
    
    
    if swap_axes:
        x_ticks = np.linspace(min_x_tick-0.02, max_x_tick, 7)
        y_ticks = np.linspace(min_y_tick-0.01, max_y_tick+0.002, 7)
        x_lim_min = min(x_ticks)
        x_lim_max = max(x_ticks)
        y_lim_min = min(y_ticks)
        y_lim_max = max(y_ticks)
        
        # Make axis lengths equal by adjusting limits
        x_range = x_lim_max - x_lim_min
        y_range = y_lim_max - y_lim_min
        ax.set_xlim(x_lim_min, x_lim_max+0.02)
        ax.set_ylim(y_lim_min, y_lim_max)
        # The aspect ratio is now handled by the figure size, so we don't need this.
        # ax.set_aspect('equal', adjustable='box')
    else:
        x_ticks = np.linspace(min_x_tick-(0.02 if error_bar else 0.02), max_x_tick+(0.02 if error_bar else 0.02), 10)
        y_ticks = np.linspace(min_y_tick-0.02, max_y_tick, 10)
        x_lim_min = min(x_ticks)
        x_lim_max = max(x_ticks)
        y_lim_min = min(y_ticks)
        y_lim_max = max(y_ticks)
        ax.set_xlim(x_lim_min, x_lim_max+0.01)
        ax.set_ylim(y_lim_min, y_lim_max+0.01)
        # The aspect ratio is now handled by the figure size, so we don't need this.
        # ax.set_aspect('equal', adjustable='box')
    
    # Set the ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Format tick labels with high precision for very close values
    # Use 4 decimal places to differentiate very close values
    
    x_tick_labels = [f"{tick:.1f}" for tick in x_ticks]
    y_tick_labels = [f"{tick:.2f}" for tick in y_ticks]
    
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)
    
    # Create LaTeX mapping for constraint names
    constraint_latex_map = {
        'zero_shot': {"flexibility": r"$\mathbf{W} = I_{p \times p}$", "regularization": r"$\mathbf{W} = I$"},
        'diagonal': {"flexibility": r"$\mathbf{W}\in \mathrm{Diag}_p(\mathbb{R})$", "regularization": r"$\|\mathbf{W} - I\|_F^2$"},
        'Rectangular_10': {"flexibility": r"$\mathbf{W} \in \mathbb{R}^{p\times 10}$", "regularization": r"$\|\mathbf{W}\mathbf{W}^T -\gamma I\|_F^2$"},
        'Rectangular_30': {"flexibility": r"$\mathbf{W} \in \mathbb{R}^{p\times 30}$", "regularization": r"$\|\mathbf{W}\mathbf{W}^T -\gamma I\|_F^2$"},
        'full_W_L1': {"flexibility": r"$\mathbf{W} \in \mathbb{R}^{p\times p}$", "regularization": r"$\|\mathbf{W}\|_1$"},
        'full_W': {"flexibility": r"$\mathbf{W} \in \mathbb{R}^{p\times p}$", "regularization": r"$\|\mathbf{W} -\gamma I\|_F^2$"},
    }
    
    # Don't create legend here - will be created separately
    return constraint_colors, train_size_markers, unique_constraints, unique_train_sizes, constraint_latex_map

def create_meta_plot(final_results_path, width_cm=20, height_cm=12, ax_title_fontsize=14, 
                    tick_fontsize=12, legend_fontsize=12, marker_size=100, 
                    legend_marker_size=10, swap_axes=True, show_grid=False,top_acc:bool = False, 
                    error_bar:bool = False, constraints_list:Optional[List[str]]=None):
    """
    Create the complete meta plot figure visualizing flexibility-accuracy tradeoff.
    
    This function is the main entry point for creating the tradeoff figure. It loads data,
    sets up the figure, and calls create_main_plot to generate the visualization showing
    the relationship between model flexibility and performance.
    
    Parameters
    ----------
    data_file : str
        Path to the CSV file containing the data
    width_cm : float, default=20
        Width of the figure in centimeters
    height_cm : float, default=12
        Height of the figure in centimeters
    ax_title_fontsize : int, default=14
        Font size for axis titles
    tick_fontsize : int, default=12
        Font size for tick labels
    legend_fontsize : int, default=12
        Font size for legend text
    marker_size : int, default=100
        Size of the markers in the plot
    legend_marker_size : int, default=10
        Size of the markers in the legend
    swap_axes : bool, default=True
        If True, swap the X and Y axes (model_recovery_acc on X)
    show_grid : bool, default=False
        Whether to show grid lines
    top_acc : bool, default=False
        If True, use top_acc instead of test_acc
    error_bar : bool, default=False
        If True, add error bars using test_acc_std
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
        
    Notes
    -----
    The created figure visualizes the tradeoff between model recovery accuracy
    and test accuracy across different constraint types (flexibility levels) and
    training set sizes. Different constraint types are represented by colors,
    while training set sizes are indicated by marker shapes.
    """
    # Load data
    data = load_data(final_results_path, constraints_list=constraints_list)
    
    # Calculate data range for aspect ratio
    x_values = np.asarray(data['model_recovery_acc'].values)
    y_values = np.asarray(data['test_acc' if not top_acc else 'max_acc_model_acc'].values)
    x_range = np.max(x_values) - np.min(x_values)
    y_range = np.max(y_values) - np.min(y_values)

    # Setup figure
    fig, gs = setup_figure(width_cm, height_cm, aspect_ratio_data=(x_range, y_range))
    
    # Remove default figure padding to give gridspec full control
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Create main plot axis - row 1 (index 1), columns 1-8 (indices 1:7)
    plot_ax = fig.add_subplot(gs[1, 1:8])
    
    # Call create_main_plot with the axis
    constraint_colors, train_size_markers, unique_constraints, unique_train_sizes, constraint_latex_map = create_main_plot(
        plot_ax, data, ax_title_fontsize, tick_fontsize, marker_size, legend_marker_size, swap_axes, show_grid, top_acc, error_bar
    )
    
    # Create a background axes that spans all three legends for the frame


    # Create legend axes on row 3 (index 3) - these will be on top of the frame
    # Flexibility legend: row 3, column 2
    flex_ax = fig.add_subplot(gs[3, 2])
    flex_ax.axis('off')
    
    # Regularization legend: row 3, column 4
    reg_ax = fig.add_subplot(gs[3, 4])
    reg_ax.axis('off')

    # Train set size legend: row 3, column 6
    train_ax = fig.add_subplot(gs[3, 6])
    train_ax.axis('off')
    
    # --- Create Legend Elements ---
    
    # 1. Flexibility Legend Elements
    flexibility_elements = []
    for constraint in unique_constraints:
        flex_label = constraint_latex_map.get(constraint, {}).get('flexibility', constraint)
        flexibility_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=constraint_colors[constraint], 
                   markersize=legend_marker_size, label=flex_label)
        )
    
    # 2. Regularization Legend Elements (invisible markers)
    regularization_elements = []
    for constraint in unique_constraints:
        reg_label = constraint_latex_map.get(constraint, {}).get('regularization', '')
        regularization_elements.append(
            Line2D([0], [0], marker='', color='none', markerfacecolor='none', 
                   markersize=0, label=reg_label)  # Completely transparent, no marker
        )
    
    # 3. Train Set Size Legend Elements
    train_size_elements = []
    for size, marker in train_size_markers.items():
        train_size_elements.append(
            Line2D([0], [0], marker=marker, color='black', 
                   markersize=legend_marker_size*0.7, label=f"{size:,}")  # Smaller markers for train size
        )
    
    # --- Draw Legends ---
    # All legends have higher z-order than the frame so they appear on top
    
    # 1. Flexibility Legend
    leg1 = flex_ax.legend(
        handles=flexibility_elements,
        title='flexibility',
        fontsize=legend_fontsize,
        frameon=False,
        loc='center left',
        handletextpad=0.5,
        labelspacing=0.5,
        title_fontsize=legend_fontsize
    )
    leg1.set_zorder(100)  # Bring flexibility legend to front
   # plt.setp(leg1.get_texts(), fontweight='bold')
    plt.setp(leg1.get_title(), fontweight='bold')
    
    # 2. Regularization Legend (same format as flexibility, but with invisible markers)
    leg2 = reg_ax.legend(
        handles=regularization_elements,
        title='regularization',
        fontsize=legend_fontsize,
        frameon=False,
        loc='center left',
        handletextpad=0.5,
        labelspacing=0.5,
        title_fontsize=legend_fontsize
    )
    leg2.set_zorder(100)  # Bring regularization legend to front
    #plt.setp(leg2.get_texts(), fontweight='bold')
    plt.setp(leg2.get_title(), fontweight='bold')

    # 3. Train set size legend
    leg3 = train_ax.legend(
        handles=train_size_elements,
        title='train set size',
        fontsize=legend_fontsize,
        frameon=False,
        loc='center',
        handletextpad=0.5,
        labelspacing=1.2,
        title_fontsize=legend_fontsize
    )
    leg3.set_zorder(100)  # Bring train set size legend to front
    #plt.setp(leg3.get_texts(), fontweight='bold')
    plt.setp(leg3.get_title(), fontweight='bold')
    
    # No layout adjustments needed here - everything is controlled by gridspec
    return fig



@hydra.main(version_base=None, config_path='../scripts_configurations', config_name="Figure_5")
def main(cfg: DictConfig):
    """
    Main execution script to generate the flexibility-accuracy tradeoff figure.
    
    This script orchestrates the complete figure generation workflow:
    1. Loads configuration from Figure_5.yaml
    2. Reads pre-computed results from CSV
    3. Creates scatter plot with color-coded constraints
    4. Positions three legends using GridSpec
    5. Saves publication-quality PDF
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration loaded from scripts_configurations/Figure_5.yaml
        Contains all figure parameters, styling, and data paths
        
    Configuration Structure:
    -----------------------
    cfg.save_path : str
        Directory containing final_results_df.csv
    cfg.constraint_types_to_plot : List[str]
        Which constraints to include (e.g., ['zero_shot', 'full_W'])
    cfg.figure_dimensions : Dict
        width_cm and height_cm for figure size
    cfg.font_sizes : Dict
        Font sizes for various elements
    cfg.plot_styling : Dict
        Marker sizes, line styles, etc.
    cfg.output : Dict
        Output directory and filename
        
    Side Effects:
    ------------
    - Creates output directory if it doesn't exist
    - Saves PDF file to {output.directory}/{output.filename}
    - Prints save location to console
    
    Example Usage:
    -------------
    # Default configuration
    >>> python Figure_5_tradeoff.py
    
    # Override specific parameters
    >>> python Figure_5_tradeoff.py show_error_bars=true
    >>> python Figure_5_tradeoff.py output.filename="Figure_5_custom.pdf"
    
    Raises:
    ------
    FileNotFoundError
        If final_results_df.csv doesn't exist at cfg.save_path
    
    Notes:
    -----
    The resulting figure (Figure_5.pdf) shows the relationship between
    model recovery accuracy and test accuracy, visualizing how model flexibility
    impacts both predictive performance and model identifiability.
    
    See Also:
    --------
    create_meta_plot : Core plotting function
    load_data : Data loading with filtering
    """
    # Construct path to results file
    final_results_path = os.path.join(cfg.save_path, "final_results_df.csv")
    
    # Create output directory if needed
    output_dir = cfg.output.directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which constraints to plot
    # Use filter_constraints if specified, otherwise use all from config
    if cfg.get('filter_constraints') is not None:
        constraints_list = cfg.filter_constraints
        print(f"Filtering to constraints: {constraints_list}")
    else:
        constraints_list = cfg.constraint_types_to_plot
        print(f"Plotting all configured constraints: {constraints_list}")
    
    # Extract configuration parameters
    # Figure dimensions
    width_cm = cfg.figure_dimensions.width_cm
    height_cm = cfg.figure_dimensions.height_cm
    
    # Font sizes
    ax_title_fontsize = cfg.font_sizes.panel_letter
    tick_fontsize = cfg.font_sizes.tick_labels
    legend_fontsize = cfg.font_sizes.legend
    
    # Plot styling
    marker_size = cfg.plot_styling.marker_size
    legend_marker_size = cfg.plot_styling.legend_marker_size
    
    # Advanced options
    swap_axes = cfg.axis_settings.swap_axes
    show_grid = cfg.axis_settings.show_grid
    use_top_acc = cfg.get('use_top_model_accuracy', False)
    show_error_bars = cfg.get('show_error_bars', False)
    
    # Create the figure
    fig = create_meta_plot(
        final_results_path=final_results_path,
        width_cm=width_cm,
        height_cm=height_cm,
        ax_title_fontsize=ax_title_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize,
        marker_size=marker_size,
        legend_marker_size=legend_marker_size,
        swap_axes=swap_axes,
        show_grid=show_grid,
        top_acc=use_top_acc,
        error_bar=show_error_bars,
        constraints_list=constraints_list,
    )
    
    # Save the figure
    save_path = os.path.join(output_dir, cfg.output.filename)
    fig.savefig(
        save_path, 
        format=cfg.output.format, 
        dpi=cfg.output.dpi,
        bbox_inches='tight'  # Remove extra whitespace
    )
    
    print(f"Figure saved successfully to: {save_path}")
    print(f"Figure dimensions: {width_cm} cm × {height_cm} cm")
    print(f"Resolution: {cfg.output.dpi} DPI")
    print(f"Constraints shown: {', '.join(constraints_list)}")

    


if __name__ == "__main__":
    main()
