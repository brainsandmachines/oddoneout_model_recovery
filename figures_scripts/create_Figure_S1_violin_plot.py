"""
Figure S1: Violin Plot for Regularization Method Comparison

This script generates Figure S1 of the supplementary materials, which compares
the performance of different regularization methods (L2 vs. Scalar Matrix 
Shrinkage) across various transformation flexibility constraints.

The figure shows:
- Violin plots of test accuracy distributions for each constraint type
- Statistical equivalence testing using TOST (Two One-Sided Tests)
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib as mpl
from typing import Tuple
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
# Data Loading and Preprocessing Functions
# =============================================================================

def combine_dfs(main_folder_path, base_results_name, features_type, create_new_df=False):
    """
    Combines CSV files from multiple constraint subdirectories into a single DataFrame.
    
    Parameters
    ----------
    main_folder_path : str
        Path to the main directory containing constraint subdirectories
    base_results_name : str
        Base name for result files (without constraint suffix)
    features_type : str
        Type of features to use (e.g., 'full', 'PCA_500')
    create_new_df : bool, default False
        If True, recreates the combined DataFrame even if it already exists
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with data from all constraint subdirectories
    """
    full_dir = os.path.join(main_folder_path, features_type)
    constraints_names = os.listdir(full_dir) 
    # Remove zero_shot from the constraint names because it does not have regularization methods
    constraints_names = [name for name in constraints_names if name != 'zero_shot']
    # Create an empty df with the columns: model_name, reg_con, test_acc, Regularization_type, feature_name, constraint_name
    columns = ["model_name", "reg_con", "test_acc", "Regularization_type", "feature_name", "constraint_name"]
    combined_df = pd.DataFrame(columns=columns)
    combined_save_path = os.path.join(main_folder_path, f"{base_results_name}_{features_type}.csv")
    # If it exists, load it
    if os.path.exists(combined_save_path) and not create_new_df:
        combined_df = pd.read_csv(combined_save_path)
        # Filter out MoCo_V2_RN50 results
        combined_df = combined_df[combined_df['model_name'] != 'MoCo_V2_RN50']
        # Filter out zero shot models
        combined_df = combined_df[~combined_df['model_name'].str.contains('ZS')]
        return combined_df
    elif create_new_df and os.path.exists(combined_save_path):
        os.remove(combined_save_path)
        
    # Load all CSV files in the main folder
    for constraint_name in constraints_names:
        constraint_dir = os.path.join(full_dir, constraint_name)
        df_file_name = f"{base_results_name}_{features_type}_{constraint_name}.csv"
        df_path = os.path.join(constraint_dir, df_file_name)
        df = pd.read_csv(df_path)
        # Filter out MoCo_V2_RN50 results
        df = df[df['model_name'] != 'MoCo_V2_RN50']
        # Filter out zero shot models
        df = df[~df['model_name'].str.contains('ZS')]
        # Select required columns and add feature_name and constraint_name
        selected_df = df[["model_name", "reg_con", "test_acc", "Regularization_type"]].copy()
        selected_df["feature_name"] = features_type
        selected_df["constraint_name"] = constraint_name
        # Concatenate with the combined DataFrame
        combined_df = pd.concat([combined_df, selected_df], ignore_index=True)
    
    # Save the combined df
    combined_df.to_csv(combined_save_path, index=False)
    
    return combined_df

def _sort_key(name):
    """
    Custom sorting key for constraint names to ensure proper ordering.
    
    Constraints are ordered by increasing flexibility:
    1. diagonal (0, 0) - least flexible
    2. Rectangular_k (1, k) - sorted by k value
    3. full_W (3, 0) - most flexible
    
    Parameters
    ----------
    name : str
        Constraint name (e.g., 'diagonal', 'Rectangular_50', 'full_W')
        
    Returns
    -------
    tuple
        (priority, secondary_sort_key) for sorting
    """
    if name == 'full_W':
        return (3, 0)  # lowest priority (most flexible)
    elif name == 'diagonal':
        return (0, 0)  # highest priority (least flexible)
    elif name.startswith('Rectangular_'):
        number = int(name.split('_')[1])
        return (1, number)  # sort ascending by dimensionality
    else:
        return (2, 0)  # fallback for any unexpected values


# =============================================================================
# Statistical Analysis Functions
# =============================================================================

def sort_key(name: str):
    """
    Custom constraint ordering: diagonal → Rectangular_k → misc → full_W.
    
    Parameters
    ----------
    name : str
        Constraint name
        
    Returns
    -------
    tuple
        Sorting key
    """
    if name == "full_W":
        return (3, 0)
    if name == "diagonal":
        return (0, 0)
    if name.startswith("Rectangular_"):
        return (1, int(name.split("_")[1]))
    return (2, 0)


def sort_key_df(col):
    """
    Apply sort_key to a pandas Series/column.
    
    Parameters
    ----------
    col : pd.Series
        Column containing constraint names
        
    Returns
    -------
    pd.Series
        Series of sorting keys
    """
    return col.apply(sort_key)


# =============================================================================
# Low-level Statistical Helper Functions
# =============================================================================

def _paired_arrays(df: pd.DataFrame, constraint: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return *paired* accuracy vectors (L2, eye_distance) for a given constraint.

    Rows are paired by `model_name` to guarantee one‑to‑one alignment. Any
    model missing either regularizer is dropped for that constraint.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing test accuracy results with columns:
        'constraint_name', 'model_name', 'Regularization_type', 'test_acc'
    constraint : str
        Constraint type to filter (e.g., 'diagonal', 'full_W')
        
    Returns
    -------
    tuple of np.ndarray
        (L2_accuracies, eye_distance_accuracies) paired by model
    """
    sub = df[df["constraint_name"] == constraint]
    wide = (
        sub.pivot_table(
            index="model_name", columns="Regularization_type", values="test_acc", aggfunc="mean"
        )
        .dropna(subset=["L2", "eye_distance"], how="any")
    )
    return wide["L2"].to_numpy(), wide["eye_distance"].to_numpy()


def _tost_paired(
    diffs: np.ndarray,
    eps: float,
    alpha: float,
    *,
    bootstrap: bool = False,
    n_bootstrap: int = 10_000,
    rng: np.random.Generator | None = None,
):
    """
    Paired TOST (Two One-Sided Tests) — analytical (default) or bootstrap.

    TOST is used to test equivalence rather than difference. The null hypothesis
    is that the difference is outside the equivalence margin [-eps, eps].
    
    Parameters
    ----------
    diffs : np.ndarray
        Paired differences (eye_distance - L2)
    eps : float
        Equivalence margin (absolute value)
    alpha : float
        Significance level (after Bonferroni correction)
    bootstrap : bool, default False
        If True, use bootstrap estimation for p-values
    n_bootstrap : int, default 10_000
        Number of bootstrap resamples
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    
    Returns
    -------
    dict
        Dictionary with keys:
        - n: sample size
        - mean_diff: mean difference
        - se: standard error
        - t_low: t-statistic for lower bound test
        - p_low: p-value for lower bound test
        - t_up: t-statistic for upper bound test
        - p_up: p-value for upper bound test
        - p_tost: max(p_low, p_up) - overall TOST p-value
        - equivalent: True if p_tost < alpha (methods are equivalent)
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(diffs)
    mean_diff = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    df = n - 1

    if bootstrap:
        # Empirical p‑values from resampled mean differences ----------------
        boot = rng.choice(diffs, size=(n_bootstrap, n), replace=True).mean(axis=1)
        p_low = np.mean(boot <= -eps)
        p_up = np.mean(boot >= eps)
        t_low = t_up = np.nan  # t‑stats not used here
    else:
        # Classical t‑based one‑sided tests --------------------------------
        t_low = (mean_diff + eps) / se            # H0: μ ≤ −eps  vs H1: μ > −eps
        p_low = stats.t.sf(t_low, df=df)
        t_up  = (mean_diff - eps) / se            # H0: μ ≥  eps  vs H1: μ <  eps
        p_up  = stats.t.cdf(t_up, df=df)

    p_tost = max(p_low, p_up)

    return {
        "n": n,
        "mean_diff": mean_diff,
        "se": se,
        "t_low": t_low,
        "p_low": p_low,
        "t_up": t_up,
        "p_up": p_up,
        "p_tost": p_tost,
        "equivalent": p_tost < alpha,
    }


# =============================================================================
# Public API - TOST Analysis
# =============================================================================

def perform_tost_on_means(
    combined_df: pd.DataFrame,
    output_path: str,
    *,
    alpha: float = 0.05,
    epsilon: float = 0.01,
    bootstrap: bool = False,
    n_bootstrap: int = 10_000,
) -> pd.DataFrame:
    """Run paired TOST across all constraints and write a rich summary.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Columns required: ``model_name``, ``constraint_name``,
        ``Regularization_type`` (values "L2", "eye_distance"), ``test_acc``.
    output_path : str
        Directory in which to save the CSV summary.
    alpha : float, default 0.05
        Family‑wise error rate; Bonferroni correction applied across constraints.
    epsilon : float, default 0.01
        Equivalence margin (same units as ``test_acc``).
    bootstrap : bool, default False
        If True, p‑values are estimated via bootstrap with ``n_bootstrap`` samples.
    n_bootstrap : int, default 10_000
        Number of bootstrap resamples.

    Returns
    -------
    pd.DataFrame
        One row per constraint with means, TOST stats, and decision.
    """

    # ---------------- 0. Input sanity checks -----------------------------
    df = combined_df.copy()
    reg_types = sorted(df["Regularization_type"].unique())
    if reg_types != ["L2", "eye_distance"]:
        raise ValueError(
            f"Expected exactly two regularizers ('L2', 'eye_distance'), got {reg_types}"
        )

    constraints = [c for c in sorted(df["constraint_name"].unique(), key=sort_key) if c != "zero_shot"]
    K = len(constraints)
    alpha_per_test = alpha / K

    print("\n==================== PAIRED TOST SUMMARY ====================")
    print(f"Equivalence margin ε = {epsilon:.4g}")
    print(f"Bootstrap mode  = {bootstrap}  (B = {n_bootstrap if bootstrap else '—'})")
    print(f"Family‑wise α   = {alpha:.3g} ⇒ per‑constraint α = {alpha_per_test:.4g}")
    print(f"Constraints     = {K}")
    print(f"Unique models   = {df['model_name'].nunique()}")
    print("============================================================\n")

    # ---------------- 1. Means for reporting ----------------------------
    means = df.groupby(["constraint_name", "Regularization_type"], sort=False)["test_acc"].mean().unstack()
    means = means.rename(columns={"L2": "Mean L2", "eye_distance": "Mean Scalar Matrix Shrinkage"})

    summary_rows = []
    rng = np.random.default_rng()

    # ---------------- 2. Loop over constraints --------------------------
    for c in constraints:
        l2_arr, shrink_arr = _paired_arrays(df, c)
        if len(l2_arr) < 2:
            print(f"⚠️  {c}: <2 paired observations — skipping")
            stats_dict = {k: np.nan for k in [
                "n", "mean_diff", "se", "t_low", "p_low", "t_up", "p_up", "p_tost", "equivalent"
            ]}
        else:
            diffs = shrink_arr - l2_arr
            stats_dict = _tost_paired(
                diffs, epsilon, alpha_per_test,
                bootstrap=bootstrap, n_bootstrap=n_bootstrap, rng=rng,
            )
            eq_flag = "✅" if stats_dict["equivalent"] else "❌"
            print(
                f"{eq_flag} {c:15s} | Δ̄ = {stats_dict['mean_diff']:+.4g} ± {stats_dict['se']:.2g} "
                f"| p_tost = {stats_dict['p_tost']:.3g}"
            )

        row = {
            "constraint_name": c,
            **means.loc[c].to_dict(),
            "Mean Difference": means.loc[c, "Mean Scalar Matrix Shrinkage"] - means.loc[c, "Mean L2"],
            **stats_dict,
            "Alpha per test": alpha_per_test,
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("constraint_name", key=lambda s: s.map(sort_key)).reset_index(drop=True)

    # ---------------- 3. End‑of‑run diagnostics -------------------------
    n_equiv = summary["equivalent"].sum()
    print("\n================ OVERALL RESULT (ε = {:.4g}) ================".format(epsilon))
    print(f"Equivalent constraints: {n_equiv}/{K}  ({(n_equiv/K):.1%})")
    if summary["mean_diff"].notna().any():
        print(f"Median |Δ̄|      : {summary['mean_diff'].abs().median():.4g}")
        print(f"Max |Δ̄|         : {summary['mean_diff'].abs().max():.4g}")
        print(f"Smallest p_tost : {summary['p_tost'].min():.3g}")
        print(f"Largest  p_tost : {summary['p_tost'].max():.3g}")
    print("============================================================\n")

    # ---------------- 4. Save CSV ---------------------------------------
    fname = f"tost_summary_eps{epsilon}{'_boot' if bootstrap else ''}.csv"
    out_fp = os.path.join(output_path, fname)
    os.makedirs(output_path, exist_ok=True)
    summary.to_csv(out_fp, index=False)
    print(f"📄  Summary table written to {out_fp}\n")

    return summary



class DFAnalysis:
    """
    Class for analyzing and visualizing regularization method comparisons.
    
    This class handles:
    - Data filtering and preprocessing
    - Violin plot generation
    - Statistical equivalence testing (TOST)
    
    Attributes
    ----------
    df : pd.DataFrame
        Filtered and sorted DataFrame containing test accuracy results
    save_path : str
        Directory path for saving output figures
    """
    
    def __init__(self, combined_df: pd.DataFrame, save_path: str):
        """
        Initialize the analysis with the data and the destination directory.
        
        Parameters
        ----------
        combined_df : pd.DataFrame
            Combined DataFrame with columns:
            - model_name: Name of neural network model
            - reg_con: Regularization constant used
            - test_acc: Test accuracy achieved
            - Regularization_type: Type of regularization (L2 or eye_distance)
            - feature_name: Feature type (e.g., 'full')
            - constraint_name: Constraint type (e.g., 'diagonal', 'full_W')
        save_path : str
            Directory path where plots will be saved
        """
        #Remove zero-shot , because it doesn't related to the regularization methods
        self.df = combined_df[combined_df['constraint_name'] != 'zero_shot']
        #Remove MoCo_V2_RN50 , because we removed him from the study 
        self.df = self.df[self.df['model_name'] != 'MoCo_V2_RN50']
        self.df = self.df.sort_values(
                                    by='constraint_name',
                                    key=sort_key_df
                                )
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            


    def violin_plot(self, cfg: DictConfig):
        """
        Generates a violin plot to show the distribution of test accuracy 
        by constraint and regularization type.
        
        The plot shows:
        - X-axis: Constraint types (ordered by flexibility)
        - Y-axis: Test accuracy
        - Split violins: L2 (left) vs. Scalar Matrix Shrinkage (right)
        - LaTeX formatted labels for publication quality
        
        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration containing:
            - figure_size: width_cm and height_cm
            - violin_plot: visualization settings
            - regularization_labels: LaTeX labels for legend
            - constraint_labels: LaTeX labels for x-axis
            - axis_labels: xlabel and ylabel text
            - output_files: violin_plot filename and dpi
        
        Returns
        -------
        None
            Figure is saved to file
        """
        import seaborn as sns  # for violin plots, only for visualization convenience
        
        # Convert cm to inches for matplotlib
        width_inches = cfg.figure_size.width_cm / 2.54
        height_inches = cfg.figure_size.height_cm / 2.54
        
        # Create a violin plot.
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        sns.violinplot(data=self.df, x='constraint_name', y='test_acc', hue='Regularization_type', 
                       split=cfg.violin_plot.split, inner=cfg.violin_plot.inner, ax=ax,legend=True)
        
        # Set axis labels with configured font properties
        ax.set_xlabel(
            cfg.axis_labels.xlabel,
            fontsize=cfg.violin_plot.xlabel_fontsize,
            fontweight=cfg.violin_plot.xlabel_fontweight
        )
        ax.set_ylabel(
            cfg.axis_labels.ylabel,
            fontsize=cfg.violin_plot.ylabel_fontsize,
            fontweight=cfg.violin_plot.ylabel_fontweight
        )
        
        # Get the legend handles and labels
        handles, original_labels = ax.get_legend_handles_labels()
        
        # Create mapping to ensure correct order
        label_mapping = dict(zip(original_labels, handles))
        ordered_handles = [label_mapping[label] for label in reversed(sorted(self.df["Regularization_type"].unique()))]
        ordered_labels = [cfg.regularization_labels[label] for label in reversed(sorted(self.df["Regularization_type"].unique()))]
        
        # Create a new legend with the LaTeX formatted labels
        ax.legend(
            handles=ordered_handles, 
            labels=ordered_labels, 
            title="regularization",
            fontsize=cfg.violin_plot.legend_fontsize
        )
        
        # Get current tick positions and labels
        positions = ax.get_xticks()
        labels = ax.get_xticklabels()
        
        # Replace the labels with LaTeX formatted versions from the dictionary
        new_labels = []
        for label in labels:
            text = label.get_text()
            if text in cfg.constraint_labels:
                new_labels.append(cfg.constraint_labels[text])
            else:
                new_labels.append(text)
        
        # Set the new tick labels
        ax.set_xticklabels(
            new_labels,
            fontsize=cfg.violin_plot.xticklabels_fontsize
        )
        plt.xticks(
            rotation=cfg.violin_plot.xtick_rotation,
            ha=cfg.violin_plot.xtick_ha,
            fontsize=cfg.violin_plot.xticklabels_fontsize
        )
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.save_path, cfg.output_files.violin_plot)
        fig.savefig(output_path, dpi=cfg.output_files.dpi)
        plt.close(fig)
        print(f"📊 Violin plot saved to: {output_path}")
        


    def run_analysis(self, cfg: DictConfig):
        """
        Run all analyses sequentially.
        
        This method:
        1. Generates violin plot visualization
        2. Performs TOST analysis with and without bootstrap
        
        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration containing all analysis parameters
        
        Returns
        -------
        None
            Results are saved to files
        """
        # Generate violin plot
        self.violin_plot(cfg)
        
        # Perform TOST analysis with different epsilon values
        for epsilon in cfg.tost_params.epsilon_values:
            # Bootstrap TOST
            summary_table = perform_tost_on_means(
                self.df, 
                cfg.tost_output_path, 
                epsilon=epsilon,
                bootstrap=cfg.tost_params.bootstrap.enabled,
                n_bootstrap=cfg.tost_params.bootstrap.n_bootstrap,
                alpha=cfg.tost_params.alpha
            )
            
            # Analytical TOST (without bootstrap)
            summary_table_no_bootstrap = perform_tost_on_means(
                self.df, 
                cfg.tost_output_path, 
                epsilon=epsilon,
                bootstrap=False,
                alpha=cfg.tost_params.alpha
            )
            

# =============================================================================
# Main Execution
# =============================================================================

@hydra.main(version_base=None, 
            config_path='../scripts_configurations', 
            config_name="figure_S1_violin_plot")
def main(cfg: DictConfig):
    """
    Main function to generate Figure S1.
    
    This function:
    1. Sets random seed for reproducibility
    2. Loads and combines CSV data from constraint subdirectories
    3. Initializes DFAnalysis class
    4. Runs complete analysis pipeline (plots + TOST tests)
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration loaded from figure_S1_violin_plot.yaml
    
    Configuration Override Examples
    --------------------------------
    # Change epsilon value:
    python Figure_S1_violin_plot.py tost_params.epsilon_values=[0.01,0.05,0.1]
    
    # Disable bootstrap:
    python Figure_S1_violin_plot.py tost_params.bootstrap.enabled=False
    
    # Change output path:
    python Figure_S1_violin_plot.py save_path="Plots/alternative/"
    
    # Recreate combined DataFrame:
    python Figure_S1_violin_plot.py create_new_df=True
    """
    # Set random seed for reproducibility
    np.random.seed(cfg.random_seed)
    
    # Print configuration
    print("\n" + "="*70)
    print("Figure S1: Violin Plot - Regularization Method Comparison")
    print("="*70)
    print(f"Features path: {cfg.features_path}")
    print(f"Features type: {cfg.features_type}")
    print(f"Output path: {cfg.save_path}")
    print(f"TOST epsilon values: {cfg.tost_params.epsilon_values}")
    print(f"Bootstrap enabled: {cfg.tost_params.bootstrap.enabled}")
    print("="*70 + "\n")
    
    # Load and combine data from all constraint subdirectories
    combined_dfs = combine_dfs(
        main_folder_path=cfg.features_path,
        base_results_name=cfg.base_results_name,
        features_type=cfg.features_type,
        create_new_df=cfg.create_new_df
    )
    
    # Initialize analyzer and run analysis
    analyzer = DFAnalysis(combined_dfs, cfg.save_path)
    analyzer.run_analysis(cfg)
    
    # Print summary of outputs
    figure_path = os.path.join(cfg.save_path, cfg.output_files.violin_plot)
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print("="*70)
    print(f"📊 Figure saved to: {figure_path}")
    print(f"📄 TOST results saved to: {cfg.tost_output_path}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

