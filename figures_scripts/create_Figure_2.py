"""
================================================================================
Figure 2: Model Predictive Accuracy on THINGS Odd-One-Out Task
================================================================================

This script generates Figure 2 from the paper.

Purpose:
--------
Visualizes how state-of-the-art vision models perform on predicting human
odd-one-out choices from the THINGS behavioral dataset. The figure demonstrates
the relationship between transformation flexibility (constraint types) and
predictive accuracy.

Figure Components:
------------------
Each panel shows:
1. Horizontal bar plots: Model accuracy with 95% confidence intervals
2. Metroplot: Statistical pairwise comparisons (Bonferroni-corrected sign tests)
3. Color coding: Models grouped by training objective
   - Supervised (ImageNet classification)
   - Self-supervised (contrastive/predictive)
   - Image-Text Alignment (multimodal)
4. Architecture annotations: 'T' (Transformer) vs 'C' (CNN)
5. Reference lines:
   - Chance level: 33% (random guessing on 3-alternative forced choice)
   - Noise ceiling: 67% (upper bound from human consistency)

Constraint Types (Transformation Flexibility):
----------------------------------------------
- zero_shot: W = I (identity, no transformation)
- diagonal: W ∈ Diag(ℝ^{p×p}) (feature-wise scaling)
- Rectangular_N: W ∈ ℝ^{p×N} (low-rank projection to N dimensions)
- full_W: W ∈ ℝ^{p×p} (full linear transformation)

Statistical Analysis:
--------------------
Pairwise model comparisons using sign tests with Bonferroni correction:
- Tests whether one model predicts significantly more triplets correctly
- Only considers triplets where models disagree (excludes ties)
- Corrects for multiple comparisons across all model pairs

Usage:
------
    # Using Hydra configuration (recommended)
    python create_Figure_2.py
    
    # Override configuration parameters
    python Figure_2_THINGS_accuracy.py constraint_types="['zero_shot','full_W']"
    python Figure_2_THINGS_accuracy.py regularization_type="L2"
    python Figure_2_THINGS_accuracy.py force_recompute=true
    python Figure_2_THINGS_accuracy.py use_additional_models=True
    
    # Switch between L2 and eye_distance results (using same directory):
    python Figure_2_THINGS_accuracy.py regularization_type="L2"
    python Figure_2_THINGS_accuracy.py regularization_type="eye_distance"

Configuration:
--------------
All parameters are specified in: scripts_configurations/figures_2.yaml
See that file for detailed parameter documentation.

IMPORTANT - Understanding results_base_dir:
    The default evaluation creates results for BOTH L2 and eye_distance in the SAME directory:
    
    Directory: Results/regularization_methods_compare_all_models_eye_distance/
    Contains:
        ├── best_test_acc_L2_full_full_W.csv              # L2 results
        ├── best_test_acc_eye_distance_full_full_W.csv    # eye_distance results
        └── best_test_acc_for_each_model_full_full_W.csv  # Combined
    
    The regularization_type parameter selects WHICH file to load:
        - regularization_type="eye_distance" → loads best_test_acc_eye_distance_*.csv
        - regularization_type="L2" → loads best_test_acc_L2_*.csv
    
    You only need to change results_base_dir if you created a separate evaluation run.

Inputs:
-------
- Model predictions: Results/regularization_methods_compare_all_models_eye_distance/
- THINGS ground truth: Data/Things_data_preprocessed/
- Configuration: scripts_configurations/figures_2.yaml

Outputs:
--------
- Figure: Plots/figure_2_THINGS_accuracy_Delta.pdf
- Prepared data: Results/models_accuracy_analysis_THINGS_accuracy/

Dependencies:
-------------
- tools/prepare_dfs_for_figure_2.py: Data preparation and statistical analysis
- tools/metroplot.py: Pairwise comparison visualization 
- scripts_configurations/: Hydra configuration files
- run analysis_scripts/evaluate_THINGS_OOO_accuracy_CV.py : Model evaluation on THINGS OOO and make sure that you save the prediction for the constraint types you want to use in figure 2
================================================================================
"""

import os
import sys
from pathlib import Path
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PARENT_DIR)  # We want the script to work with relative paths as if script is one level up
sys.path.append(str(PARENT_DIR))
from itertools import combinations
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib as mpl
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from collections import Counter
from tools.metroplot import metroplot

# Import data preparation and statistical analysis functions
from tools.prepare_dfs_for_figure_2 import (
    prepare_figure_2_data,
    save_prepared_data,
    check_existing_results,
)

# =============================================================================
# Matplotlib Configuration
# =============================================================================
# Use Arial for consistency with publication standards
# Use Computer Modern for mathematical notation
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "cm",
    }
)

# =============================================================================
# Initialize Hydra for Configuration Management
# =============================================================================
# Hydra is initialized once to load base figure formatting parameters
# These parameters ensure consistent styling across all figures in the paper
with initialize(version_base=None, config_path="../scripts_configurations"):
    base_configs = compose(config_name="figure_base")

# Extract and scale font sizes for publication-quality figures
AXIS_FONTSIZE = base_configs.axis_labels_fontsize * 0.7      # Axis label font size
TICK_FONTSIZE = base_configs.tick_labels_fontsize * 0.6      # Tick label font size
FIGURE_WIDTH = base_configs.width_size_figure                 # Figure width in cm
PANEL_LETTER_FONTSIZE = base_configs.panel_letter_fontsize * 0.7  # Panel letter (A, B, C, D)
LEGEND_FONTSIZE = base_configs.legend_fontsize * 0.9          # Legend font size


# =============================================================================
# Color Palette Generation
# =============================================================================


def create_colors_list(
    n_colors, start=0, rot=0.5, dark=0.3, light=0.85, palette_type="cubehelix"
):
    """
    Create a dictionary mapping models to distinct colors using cubehelix palette.

    This function generates a list of distinct colors using seaborn's cubehelix palette,
    with customizable color properties. It's used to ensure consistent color coding
    across different model types in visualizations.

    Parameters:
        n_colors (int): Number of colors to generate
        start (float, default=0): Starting position in color wheel (0-3)
        rot (float, default=0.5): Number of rotations through the color wheel
        dark (float, default=0.3): Darkness of the darkest color (0-1)
        light (float, default=0.85): Lightness of the lightest color (0-1)
        palette_type (str, default="cubehelix"): Type of color palette to use

    Returns:
        list: List of hex color codes with n_colors distinct colors

    Notes:
        The function uses seaborn's cubehelix_palette which creates perceptually uniform
        color palettes with good sequential properties. The colors are converted to
        hex format for compatibility with various matplotlib functions.
    """

    palette = sns.cubehelix_palette(
        n_colors=n_colors, start=start, rot=rot, dark=dark, light=light
    )

    colors_list = [mcolors.rgb2hex(color) for color in palette]
    return colors_list




# --- Statistical helpers ---
def process_predictions_to_success_and_failures(x_preds, y_preds, correct):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y, c = x_preds.to(device), y_preds.to(device), correct.to(device)
    corr_x = x == c
    corr_y = y == c
    wins_x = corr_x & ~corr_y
    wins_y = corr_y & ~corr_x
    n_x = wins_x.sum().item()
    n_y = wins_y.sum().item()
    total = n_x + n_y
    ties = (corr_x == corr_y).sum().item()
    return n_x, total, ties, n_y


def sign_test(n_x, total, alternative="two-sided"):
    if total > 0:
        return binomtest(n_x, total, 0.5, alternative=alternative).pvalue
    return np.nan


def multiple_sign_test(preds_dict, correct, alpha=0.05):
    results = []
    names = list(preds_dict.keys())
    for a, b in combinations(names, 2):
        nx, tot, ties, ny = process_predictions_to_success_and_failures(
            preds_dict[a], preds_dict[b], correct
        )
        p = sign_test(nx, tot)
        results.append(
            {"a": a, "b": b, "n_a": nx, "n_b": ny, "total": tot, "ties": ties, "p": p}
        )
    valid = [r for r in results if not np.isnan(r["p"])]
    ps = np.array([r["p"] for r in valid])
    rej, p_corr, _, _ = multipletests(ps, alpha=alpha, method="bonferroni")
    for r, rc, sig in zip(valid, p_corr, rej):
        r["p_corr"] = rc
        r["sig"] = sig
        r["dir"] = 1 if r["n_a"] > r["n_b"] else -1
    return valid, results


def prepare_results_for_metro(results, formal_dict):
    df = pd.DataFrame(results)[["a", "b", "dir", "sig"]]
    df.columns = ["level1", "level2", "effect_direction", "is_sig"]
    df["level1"] = df["level1"].map(lambda m: go_formal(m, formal_dict))
    df["level2"] = df["level2"].map(lambda m: go_formal(m, formal_dict))
    return df


def go_formal(name, formal_dict):
    if name in ("Google_ViT_Large_224", "Google_ViT_Large"):
        return "ViT L/16"
    return formal_dict.get(name, name)


def calculate_binomial_ci(k, n, conf=0.95):
    res = binomtest(k, n)
    ci = res.proportion_ci(confidence_level=conf)
    return ci.low, ci.high


def draw_gradient_legend(ax, grad_dict, steps=50, box_h=0.15):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    n = len(grad_dict)
    total_h = n * box_h + (n - 1) * 0.02
    start = 0.95 - total_h / 2
    for i, (label, cols) in enumerate(grad_dict.items()):
        y = start - i * (box_h + 0.02)
        cmap = mcolors.LinearSegmentedColormap.from_list('gradient_legend', cols)
        for j in range(steps):
            x0 = 0.1 + j / steps * 0.3
            rect = patches.Rectangle(
                (x0, y), 0.3 / steps, box_h, color=cmap(j / (steps - 1)), lw=0
            )
            ax.add_patch(rect)
        ax.text(
            0.45,
            y + box_h / 2,
            label,
            va="center",
            ha="left",
            fontsize=LEGEND_FONTSIZE,
            fontweight="bold",
        )


def create_colors(n, start, rot, dark, light):
    pal = sns.cubehelix_palette(n, start=start, rot=rot, dark=dark, light=light)
    return [mcolors.rgb2hex(c) for c in pal]


# --- Drawing a single panel ---
def draw_panel(
    ax_bar,
    ax_metro,
    ax_leg,
    ax_ct,
    data,
    colors_dict,
    formal_names_dict,
    legend_grad,
    arch_dict,
    order,
    metro_df,
    alpha=0.05,
    panel_letter=None,
    x_label=None,
    axis_label_fontsize=AXIS_FONTSIZE,
    tick_label_fontsize=TICK_FONTSIZE,
    panel_letter_fontsize=PANEL_LETTER_FONTSIZE,
    legend_fontsize=LEGEND_FONTSIZE,
):
    """
    Draw a single panel of the model accuracy figure with bar plot and metroplot.

    This function creates one panel of the composite figure showing model accuracies
    on the THINGS odd-one-out task. It includes a horizontal bar plot of model
    accuracies, a metroplot showing pairwise statistical relationships between models,
    and a legend explaining the color coding of different model types.

    Parameters:
        ax_bar (matplotlib.axes.Axes): Axes for the bar plot
        ax_metro (matplotlib.axes.Axes): Axes for the metroplot
        ax_leg (matplotlib.axes.Axes): Axes for the legend
        ax_ct (matplotlib.axes.Axes): Axes for the C/T (CNN/Transformer) text annotation
        data (pd.DataFrame): DataFrame containing model test accuracies and confidence intervals
        colors_dict (dict): Dictionary mapping model names to colors
        formal_names_dict (dict): Dictionary mapping model names to formal display names
        legend_grad (dict): Dictionary defining gradient colors for different model categories
        arch_dict (dict): Dictionary mapping model names to architecture types ('C' or 'T')
        order (list): List defining the order of models in the plot
        metro_df (pd.DataFrame): DataFrame for metroplot containing model pairwise comparison results
        alpha (float, default=0.05): Significance level for statistical tests
        panel_letter (str, optional): Letter label for the panel (e.g., 'A', 'B')
        x_label (str, optional): Custom x-axis label

    Notes:
        - The panel shows a horizontal bar plot of model accuracies with error bars
        - Chance level (1/3) and noise ceiling (0.67) are indicated by vertical lines
        - Models are annotated as 'C' (CNN) or 'T' (Transformer) based on architecture
        - The metroplot shows pairwise statistical relationships between models
        - The color legend shows model categories (Supervised, Self-supervised, Multimodal)
    """
    # Unpack
    df = data
    yerr = np.vstack([df["test_acc_low"], df["test_acc_high"]])
    # panel letter & axes formatting
    ax_bar.axvline(1 / 3, color="k", linestyle=":", linewidth=0.5, alpha=0.7)
    ax_bar.axvline(0.67, color="gray", linestyle="-", linewidth=0.5, alpha=0.7)
    bars = ax_bar.barh(
        range(len(df)),
        df["test_acc"],
        xerr=yerr,
        capsize=2,
        color=[colors_dict[m] for m in df["model_name"]],
        height=0.7,
        error_kw={"linewidth": 0.2, "capthick": 0.2},
    )
    # ylabels
    labs = [go_formal(m, formal_names_dict) for m in df["model_name"]]
    ax_bar.set_yticks(range(len(labs)))
    ax_bar.set_yticklabels(labs, fontsize=TICK_FONTSIZE, weight="bold")
    for lbl, color in zip(
        ax_bar.get_yticklabels(), [colors_dict[m] for m in df["model_name"]]
    ):
        lbl.set_color(color)
    ax_bar.set_xticks([1 / 3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.67])
    ax_bar.set_xticklabels(
        ["0.33", "0.40", "0.45", "0.50", "0.55", "0.60", "0.67"],
        fontsize=TICK_FONTSIZE,
        weight="bold",
    )
    for spine in ("right", "top"):
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["left"].set_linewidth(0.3)
    ax_bar.spines["bottom"].set_linewidth(0.3)
    ax_bar.tick_params(width=0.3, length=2)
    ax_bar.set_xlim(0.325, 0.7)
    if x_label:
        ax_bar.set_xlabel(x_label, fontsize=AXIS_FONTSIZE, weight="bold")
    else:
        ax_bar.set_xlabel("Accuracy", fontsize=AXIS_FONTSIZE, weight="bold")
    # C/T legend
    if panel_letter == "A":
        ax_ct.text(
            0.5,
            0.7,
            "T - Transformer",
            ha="center",
            va="center",
            fontsize=legend_fontsize,
            weight="bold",
        )
        ax_ct.text(
            0.5,
            0.5,
            "C - CNN",
            ha="right",
            va="center",
            fontsize=legend_fontsize,
            weight="bold",
        )
    ax_ct.set_xticks([])
    ax_ct.set_yticks([])
    ax_ct.patch.set_alpha(0)
    # Add chance level line (1/3) - black and dotted, thinner
    ax_bar.axvline(x=1 / 3, color="black", linestyle=":", alpha=0.7, linewidth=0.5)
    ax_bar.annotate(
        "chance level",
        xy=(1 / 3, len(formal_names_dict)),
        xytext=(1 / 3 + 0.03, len(formal_names_dict) + 1),
        fontsize=TICK_FONTSIZE,
        ha="right",
        va="bottom",
    )

    # Add noise ceiling line - gray and solid, thinner
    ax_bar.axvline(x=0.67, color="gray", linestyle="-", alpha=0.7, linewidth=0.5)
    ax_bar.annotate(
        "noise ceiling",
        xy=(0.67, len(formal_names_dict)),
        xytext=(0.67 - 0.03, len(formal_names_dict) + 1),
        fontsize=TICK_FONTSIZE,
        ha="left",
        va="bottom",
    )

    for sp in ax_ct.spines.values():
        sp.set_visible(False)
    # gradient legend
    if panel_letter == "A":
        draw_gradient_legend(ax_leg, legend_grad)
    else:
        ax_leg.axis("off")
    # metro - only if data is available
    if metro_df is not None and len(metro_df) > 0:
        pal = {go_formal(m, formal_names_dict): colors_dict[m] for m in order}
        metroplot(
            metro_df,
            level_to_location={
                go_formal(m, formal_names_dict): i for i, m in enumerate(order)
            },
            metroplot_element_order=[go_formal(m, formal_names_dict) for m in order],
            ax=ax_metro,
            level_axis="y",
            dominating_effect_direction=1,
            level_pallete=pal,
            level_axis_lim=ax_bar.get_ylim(),
            marker=".",
            linewidth=0.3,
            markeredgewidth=0.3,
            markersize=2,
        )
    else:
        # If no metro data, display a message or leave blank
        ax_metro.text(
            0.5,
            0.5,
            "Statistical\ncomparisons\nnot available",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax_metro.transAxes,
            alpha=0.5,
        )
    ax_metro.axis("off")
    if panel_letter:
        ax_ct.text(
            -3,
            0.4,
            panel_letter,
            fontsize=PANEL_LETTER_FONTSIZE,
            fontweight="bold",
            horizontalalignment="right",
            verticalalignment="top",
        )

    # Add C/T annotations on the bars
    for idx, (model_name, bar) in enumerate(zip(df["model_name"], bars)):
        # Get architecture type from dictionary
        architecture = arch_dict.get(model_name, "unknown")
        annotation = "T" if architecture == "Transformer" else "C"

        # Get bar width (accuracy value)
        width = df["test_acc"].iloc[idx]

        # Add text annotation in the middle of the bar
        # For horizontal bars, calculate middle position between bar start and end
        bar_middle = 0.33 + 0.01  # 0.33 is the chance level (bar start)

        ax_bar.text(
            bar_middle,  # X position (middle of the bar)
            idx,  # Y position (same as bar)
            annotation,
            va="center",
            ha="center",
            fontsize=TICK_FONTSIZE * 0.5,  # Small font size
            color="black",  # Black color
            weight="bold",  # Make it bold for better visibility
        )
    ax_bar.set_ylim(-0.5, len(df))


# --- Main ---
@hydra.main(
    version_base=None, config_path="../scripts_configurations", config_name="figures_2"
)
def main(cfg: DictConfig):
    orig = get_original_cwd()

    # Extract configuration
    constraint_types = cfg.constraint_types
    regularization_type = cfg.regularization_type
    features_type = cfg.features_type
    figure_output_name = cfg.figure_output_name
    results_base_dir_config = cfg.results_base_dir
    confidence_level = cfg.get("confidence_level", 0.95)
    force_recompute = cfg.get("force_recompute", False)
    models_list = (
        cfg.models if not cfg.use_additional_models else cfg.models + cfg.new_models
    )
    models_objective_dict = {
        key: value for key, value in cfg.models_objective.items() if key in models_list
    }
    formal_models_names_dict = {
        key: value for key, value in cfg.formal_names_dict.items() if key in models_list
    }
    models_oredered = list(models_objective_dict.keys())

    # Validate constraint types limit (figure layout supports max 4 panels)
    if len(constraint_types) > 4:
        raise ValueError(
            f"constraint_types can have maximum 4 items, got {len(constraint_types)}. "
            f"Please reduce to 4 or fewer constraint types."
        )

    # Run data preparation pipeline
    print("=" * 60)
    print("PREPARING DATA FOR FIGURE 2")
    print("=" * 60)

    # Set up paths for data preparation
    results_base_dir = os.path.join(orig, f"{results_base_dir_config}/{features_type}")
    output_dir = os.path.join(
        orig, f"Results/models_accuracy_analysis_{figure_output_name}/{features_type}"
    )

    # Check existing results
    existing = check_existing_results(output_dir, constraint_types)

    # Determine what needs to be computed
    to_compute = []
    for constraint_type in constraint_types:
        accuracy_exists = existing[constraint_type]["accuracy"] and existing[constraint_type]["metro"]
        if force_recompute:
            to_compute.append(constraint_type)
            if accuracy_exists:
                print(f"Force recomputing data for {constraint_type} (overwriting existing)")
            else:
                print(f"Computing data for {constraint_type} (new)")
        elif not accuracy_exists:
            to_compute.append(constraint_type)
            print(f"Computing data for {constraint_type} (missing)")
        else:
            print(f"Skipping {constraint_type} (already exists, use force_recompute=true to override)")

    if to_compute:
        print(f"Computing data for: {to_compute}")

        # Prepare the data
        prepared_data, successful_types, failed_types = prepare_figure_2_data(
            results_base_dir=results_base_dir,
            constraint_types=to_compute,
            regularization_types=[regularization_type],
            models_order=models_oredered,
            formal_names_dict=cfg.formal_names_dict,
            feature_type=features_type,
            original_cwd=orig,
            confidence_level=confidence_level,
        )

        # Report results
        if successful_types:
            print(f"Successfully processed: {successful_types}")
        if failed_types:
            print(f"Failed to process: {failed_types}")
            print(
                "  Possible reasons: CSV files don't exist, missing evaluation results, or file access issues"
            )

        # Save the prepared data
        save_prepared_data(prepared_data, output_dir, features_type)
        print(f"Data preparation completed! Results saved to {output_dir}")
    else:
        print("All required data already exists!")

    # Update the list of constraint types to only include successfully processed ones
    # Check which constraint types actually have data files
    available_constraint_types = []
    for constraint_type in constraint_types:
        accuracy_path = os.path.join(output_dir, f"{constraint_type}.csv")
        if os.path.exists(accuracy_path):
            available_constraint_types.append(constraint_type)
        else:
            print(
                f"Warning: No data available for {constraint_type}, excluding from figure"
            )

    if not available_constraint_types:
        raise FileNotFoundError(
            f"No valid constraint types found with data files in {output_dir}"
        )

    constraint_types = available_constraint_types
    print(f"Creating figure with available constraint types: {constraint_types}")

    print("\n" + "=" * 60)
    print("CREATING FIGURE 2")
    print("=" * 60)
    # Define panel letters and labels based on constraint types
    letters = ["A", "B", "C", "D"][: len(constraint_types)]

    # Get labels from configuration
    labels = [
        cfg.panel_labels.get(t, f"predictive accuracy ({t})") for t in constraint_types
    ]
    
    objectives_count = Counter(models_objective_dict.values())
    # Create mapping from original values to desired keys
    value_mapping = {
        "Supervised": "supervised",
        "Self-supervised": "self_supervised",
        "Image & Text Alignment": "multimodal",
    }

    # Create the final dictionary with mapped keys
    objectives_count_dict = {}
    for original_value, count in objectives_count.items():
        if original_value in value_mapping:
            new_key = value_mapping[original_value]
            objectives_count_dict[new_key] = count

    # prepare color lists
    models_colors_dict = {}
    N_models = len(models_oredered)
    # Draw number of models colors for each objective type
    supervised_colors_list = create_colors_list(N_models, start=0, rot=0.5)
    self_supervised_colors_list = create_colors_list(N_models, start=1, rot=0.5)
    multimodal_colors_list = create_colors_list(N_models, start=2, rot=0.5)

    N_supervised = objectives_count_dict["supervised"]
    N_self_supervised = objectives_count_dict["self_supervised"]
    # Draw colors for each model
    for i, model in enumerate(models_oredered):
        if i < N_supervised:
            models_colors_dict[model] = supervised_colors_list[i]
        elif i < N_supervised + N_self_supervised:
            models_colors_dict[model] = self_supervised_colors_list[i - N_supervised]
        else:
            models_colors_dict[model] = multimodal_colors_list[
                i - N_supervised - N_self_supervised
            ]

    # create the dict for the gradint legend
    for key in cfg.legend_gradient_dict.keys():
        if key == "Supervised":
            cfg.legend_gradient_dict[key] = supervised_colors_list
        elif key == "Self-supervised":
            cfg.legend_gradient_dict[key] = self_supervised_colors_list
        else:
            cfg.legend_gradient_dict[key] = multimodal_colors_list

    architecture_dict = cfg.architecture_dict
    legend_gradient_dict = cfg.legend_gradient_dict

    # Load prepared data
    base = output_dir
    all_data = {}
    model_order = list(models_colors_dict.keys())

    # load data for each constraint type
    for t in constraint_types:
        dfp = os.path.join(base, f"{t}.csv")
        df_metro_path = os.path.join(base, f"metro_{t}.csv")

        if os.path.exists(dfp):
            accuracy_df = pd.read_csv(dfp)

            # Try to load metro data, use None if not available
            metro_df = None
            if os.path.exists(df_metro_path):
                try:
                    metro_df = pd.read_csv(df_metro_path)
                    print(f"Loaded metro plot data for {t}")
                except Exception as e:
                    print(f"Warning: Could not load metro plot data for {t}: {e}")
            else:
                print(
                    f"Metro plot data not available for {t} - will skip statistical comparisons"
                )

            all_data[t] = (accuracy_df, metro_df)
        else:
            raise FileNotFoundError(f"Required accuracy data file not found: {dfp}")

    # Create figure with dynamic layout based on number of constraint types
    n_panels = len(constraint_types)

    width_cm = FIGURE_WIDTH
    height_cm = 9
    fig_size = (width_cm / 2.54, height_cm / 2.54)
    nrows, ncols = 2, 2

    fig = plt.figure(figsize=fig_size)
    outer = fig.add_gridspec(nrows, ncols, wspace=0, hspace=0)

    for i, (t, let, label) in enumerate(zip(constraint_types, letters, labels)):
        if n_panels <= 2:
            sub = outer[i].subgridspec(
                4,
                8,
                height_ratios=[0.01, 1.5, 12, 1.5],
                width_ratios=[1, 0.5, 3, 1.48, 2, 0.02, 1.8, 1.5],
                wspace=0,
                hspace=0,
            )
        elif i in [1, 3]:  # right panels
            sub = outer[i].subgridspec(
                4,
                7,
                height_ratios=[0.1, 2.8, 11.8, 0.5],
                width_ratios=[1, 0.3, 2.48, 1, 2, 1.8, 0.02],
                wspace=0,
                hspace=0,
            )
        else:
            sub = outer[i].subgridspec(
                4,
                7,
                height_ratios=[0.1, 2.8, 11.8, 0.5],
                width_ratios=[0.02, 0.3, 2.48, 1, 2, 1.8, 1],
                wspace=0,
                hspace=0,
            )

        ax_bar = fig.add_subplot(sub[2, 2:5])
        ax_metro = fig.add_subplot(sub[2, 5])
        ax_leg = fig.add_subplot(sub[1, 3])
        ax_ct = fig.add_subplot(sub[1, 1])

        # Draw panel - handle case where metro data might be None
        accuracy_df, metro_df = all_data[t]
        # Filter the results to include only relevant models
        accuracy_df = accuracy_df[accuracy_df["model_name"].isin(models_list)]
        metro_df = metro_df[
            metro_df["level1"].isin(formal_models_names_dict.values())
            & metro_df["level2"].isin(formal_models_names_dict.values())
        ]
        draw_panel(
            ax_bar,
            ax_metro,
            ax_leg,
            ax_ct,
            accuracy_df,
            models_colors_dict,
            formal_models_names_dict,
            legend_gradient_dict,
            architecture_dict,
            model_order,
            metro_df,
            panel_letter=let,
            x_label=label,
        )

        # If no metro data, hide the metro axis
        if metro_df is None:
            ax_metro.set_visible(False)

    fig.tight_layout(pad=0)
    outdir = os.path.join(get_original_cwd(), f"Plots/")
    os.makedirs(outdir, exist_ok=True)
    # Save WITHOUT tight layout (original layout)
    fname_no_tight = os.path.join(
        outdir, f"Figure_2_{figure_output_name}.pdf"
    )
    fig.savefig(fname_no_tight, format="pdf", dpi=600)

    # Save WITH tight layout (cropped to remove whitespace)
    fname = os.path.join(outdir, f"Figure_2_{figure_output_name}_tight_layout.pdf")
    fig.savefig(fname, format="pdf", dpi=600, bbox_inches="tight")

    print(f"Saved combined figure with tight layout to {fname}")
    print(f"Saved combined figure without a tight layout to {fname_no_tight}")


if __name__ == "__main__":
    main()
