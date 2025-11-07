"""
Figure 4: Multidimensional Scaling (MDS) Visualization of Model Representations

This script generates Figure 4 of the main paper, which shows a 2D multidimensional
scaling (MDS) plot visualizing the similarity relationships between different neural
network model representations before and after transformation.

The figure visualizes:
- Similarity relationships between neural network architectures
- How representations change after transformation (raw → post)
- Clustering patterns by architecture type and training paradigm
- Transformation directions indicated by green arrows

Key Features:
- Supports both 20-model (paper version) and 30-model (extended) configurations
- Uses metric MDS to convert dissimilarities to 2D coordinates
- Shows original representations (red dots) and transformed versions (connected by arrows)
- Includes carefully positioned labels to minimize overlap #might need manual adjustments



"""

import os
import sys
from pathlib import Path
PARENT_DIR = Path(__file__).resolve().parent.parent
os.chdir(
    PARENT_DIR
)  # We want the script to work with relative paths as if script is one level up
sys.path.append(str(PARENT_DIR))
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from rsatoolbox.util.vis_utils import Weighted_MDS
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import hydra
from omegaconf import DictConfig


# =============================================================================
# Configuration Constants
# =============================================================================

OUT_DIR       = "Plots/"
FIG_BASENAME  = "Figure_4_MDS"

FIG_SIZE_CM   = (10, 10)   # width × height in centimetres
SHOW_TICKS    = False      # toggle axis ticks
LINE_THICK    = 0.3
MARKER_BASE   = 8


# =============================================================================
# Helper Functions
# =============================================================================

def _legend_open_arrow(legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize):
    """
    Custom handler function for drawing open arrow heads in the legend.
    
    This function creates a FancyArrowPatch with an open (hollow) arrow head
    styled to match the original handle, for use in legend entries representing
    transformations.
    
    Parameters:
        legend: matplotlib.legend.Legend
            The legend instance
        orig_handle: matplotlib.lines.Line2D
            The original handle being replaced
        xdescent: float
            Left padding in font coordinates
        ydescent: float
            Bottom padding in font coordinates
        width: float
            Width of the legend key in font coordinates
        height: float
            Height of the legend key in font coordinates
        fontsize: float
            Font size in points
            
    Returns:
        matplotlib.patches.FancyArrowPatch
            A configured arrow patch with open head for the legend
            
    Notes:
        This function is used as a custom handler with matplotlib's legend_handler_map
        to represent transformation arrows in the legend.
    """
    return mpatches.FancyArrowPatch(
        (0,        height / 2),          # tail
        (width,    height / 2),          # head
        arrowstyle='->',                 # open head
        mutation_scale=height*2,           # head size ≈ key height
        linewidth=orig_handle.get_linewidth(),
        edgecolor=orig_handle.get_edgecolor(),
        facecolor='none')                # keeps the head hollow

# ── 2. proxy artist (never drawn in the axes) ──
proxy_arrow = mpatches.FancyArrowPatch(
    (0, 0), (1, 0),                     # geometry unimportant
    arrowstyle='->',
    linewidth=LINE_THICK * 1.3,
    edgecolor='#2ca02c',
    facecolor='none')                   # open head



from matplotlib.transforms import Bbox

def label_behind_perp(raw_xy, post_xy, span,
                      back_frac=0.02,   # shift opposite arrow
                      perp_rot=90):   # rotate text 90° from arrow
    """
    Position a label behind a point with text perpendicular to the arrow.
    
    This function calculates the optimal position and rotation angle for placing
    a text label near a point in a way that minimizes overlap with arrows.
    It positions the label slightly behind the raw point (opposite to the arrow
    direction) and rotates the text perpendicular to the arrow.
    
    Parameters:
        raw_xy: numpy.ndarray
            Array of shape (2,) containing the (x,y) coordinates of the raw model
        post_xy: numpy.ndarray or None
            Array of shape (2,) containing the (x,y) coordinates of the post model,
            or None if there is no corresponding post model
        span: float
            The maximum span (max of x-range and y-range) of the entire scatter plot,
            used to scale the offset distance
        back_frac: float, default=0.02
            Fraction of span to shift label opposite to the arrow direction
        perp_rot: float, default=90
            Rotation angle in degrees relative to the arrow direction
            
    Returns:
        tuple
            ((x, y), angle) tuple containing:
            - Label position coordinates as tuple
            - Text rotation angle in degrees
            
    Notes:
        - For models without post twins, labels are placed below the point
        - The function handles the case where raw and post positions are identical
        - The text rotation is calculated to be perpendicular to the transformation direction
    """
    
    # no post twin → just drop label a bit below the point, upright
    if post_xy is None or np.allclose(raw_xy, post_xy):
        return (raw_xy[0], raw_xy[1] - back_frac * span), 90.0

    # arrow unit vector
    v = post_xy - raw_xy
    v /= np.linalg.norm(v)

    # position a bit *behind* raw point along −v
    pos = raw_xy - back_frac * span * v

    # arrow orientation
    arrow_deg = np.degrees(np.arctan2(v[1], v[0]))

    # text rotated perpendicular to arrow
    text_deg  = arrow_deg + perp_rot
    return pos, text_deg

@hydra.main(config_path="../scripts_configurations", config_name="RSA_MDS_figures", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to generate Figure 4 MDS plot visualization.
    
    This function creates a multidimensional scaling (MDS) plot showing the relationships
    between different neural network model representations. It visualizes both original 
    model representations and their transformed versions, with arrows indicating the 
    transformation direction.
    
    The plot illustrates:
    1. The similarity relationships between different neural network architectures
    2. How model representations change after transformation during model recovery
    3. The clustering of models by architecture type and training paradigm
    
    Configuration Options (via Hydra):
    -----------------------------------
    The script can generate two versions of the figure:
    
    1. Paper version (20 models) - DEFAULT:
       python Figure_4_MDS.py
       
    2. Extended version (30 models):
       python Figure_4_MDS.py include_additional_models=True
    
    The function performs the following steps:
    1. Loads pre-computed similarity data from a netCDF file
    2. Applies metric MDS to convert similarities to 2D coordinates
    3. Creates a figure with properly positioned and styled points and arrows
    4. Adds labels for each model with manual adjustments to prevent overlaps
    5. Creates a legend explaining the visualization elements
    6. Saves the plot as a high-resolution PDF file
    
    Parameters:
        cfg : DictConfig
            Hydra configuration containing:
            - include_additional_models: Boolean flag for model count
            - features_dim: Feature dimensionality ('full' or 'PCA_500')
            - transformed_features_constraints: Transformation type
            - models: Base 20 models list
            - new_models: Additional 10 models list
            - formal_names_dict: Mapping of model names to display names
    
    Outputs:
        Saves a PDF file to: Plots/{features_dim}_dim_{constraints}/Figure_4_MDS.pdf
    
    Notes:
        - Original representations are shown in red
        - Transformed representations are connected by green arrows
        - Model name labels are positioned to minimize overlaps
        - Special handling is included for the VICE model
        - The paper version uses 20 models for clarity
        - Minor aesthetic edits were applied to the published version
        - MUST run build_rsa_sim_matrix.py first to generate similarity data
    """
    
    # =============================================================================
    # Configuration and Model Selection
    # =============================================================================
    
    # Build model list and formal names dict from configuration
    models_list = (
        cfg.models if not cfg.include_additional_models 
        else cfg.models + cfg.new_models
    )
    
    formal_names_dict = {
        key: value for key, value in cfg.formal_names_dict.items() 
        if key in models_list
    }
    # Add VICE to the dictionary
    formal_names_dict['VICE'] = 'VICE'
    
    # Determine version name for reporting
    version_name = "EXTENDED (30 models)" if cfg.include_additional_models else "PAPER (20 models)"
    
    # =============================================================================
    # Construct Similarity Matrix Path
    # =============================================================================
    
    # Build filename following the convention from build_rsa_sim_matrix.py
    features_dim = cfg.features_dim
    constraints = cfg.transformed_features_constraints
    
    # Construct base filename
    sim_filename = f"sim_{features_dim}_dim_{constraints}"
    
    # Add suffix if using additional models
    if cfg.include_additional_models:
        sim_filename += "_additional_models"
    
    sim_filename += ".nc"
    
    # Full path to similarity matrix
    similarity_path = os.path.join(cfg.rsa_results_dir, sim_filename)
    
    print(f"\n{'='*70}")
    print(f"Generating Figure 4 MDS - {version_name}")
    print(f"{'='*70}")
    print(f"Loading data from: {similarity_path}")
    
    # Verify file exists
    if not os.path.exists(similarity_path):
        error_msg = f"""
        ERROR: Similarity matrix file not found!
        
        Expected: {similarity_path}
        
        You must first generate the similarity matrix by running:
        python analysis_scripts/build_rsa_sim_matrix.py \\
            features_dim={features_dim} \\
            transformed_features_constraints={constraints} \\
            include_new_models={str(cfg.include_additional_models)}
        
        This will create the required .nc file with RSA similarity data.
        """
        raise FileNotFoundError(error_msg)
    
    sim_array = xr.load_dataarray(similarity_path)
    
    # =============================================================================
    # Data Preprocessing
    # =============================================================================
    sim = np.array(sim_array.values)
    labels = np.array(sim_array.coords['model1'].values, dtype=str)
    
    seed = 52
    base = np.char.replace(np.char.replace(labels, '_raw', ''), '_post', '')
    is_post = np.char.endswith(labels, '_post')
    is_vice = np.char.startswith(base, 'VICE')
    is_raw  = ~is_post

    # keep only one VICE (raw)
    keep = ~(is_vice & is_post)
    sim, labels, base = sim[np.ix_(keep, keep)], labels[keep], base[keep]
    is_post, is_vice, is_raw = is_post[keep], is_vice[keep], is_raw[keep]
   

    # ── dissimilarity & metric MDS ───────────────────────────────────────────
    dissim = 1.0 - sim
    coords = Weighted_MDS(n_components=2, dissimilarity='precomputed',
                        metric=True,random_state = seed).fit_transform(dissim)
    assert coords is not None,"something went wrong with the MDS process"
    post_coord = {b: coords[i] for i, (b, p) in enumerate(zip(base, is_post)) if p}

    # ── figure & gridspec (legend row + spacer + scatter row) ────────────────
    fig_inches = tuple(x / 2.54 for x in FIG_SIZE_CM)
    fig = plt.figure(figsize=fig_inches)
    gs  = gridspec.GridSpec(nrows=2, ncols=1,
                            height_ratios=[0.5, 10],
                            left=0, right=1, top=1, bottom=0,
                            hspace=0)

    # legend axis (row 0)
    ax_legend = fig.add_subplot(gs[0])
    ax_legend.axis("off")

    # scatter axis (row 2)
    ax = fig.add_subplot(gs[1])

    # limit square box around data
    span = coords.ptp()
    midx, midy = coords.mean(0)
    pad = 0.05 * span

    x_min,x_max = coords[:,0].min(),coords[:,0].max()
    y_min,y_max = coords[:,1].min(),coords[:,1].max()
    ax.set_xlim(x_min-0.05,x_max+0.05)
    ax.set_ylim(y_min-0.05,y_max+0.01)
    
    
    ax.margins(0)
    # show / hide ticks
    if not SHOW_TICKS:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_frame_on(False)
    else:
        ax.tick_params(width=LINE_THICK)

    # ── colours & markers ────────────────────────────────────────────────────
    colours = np.full_like(is_raw, '#d62728', dtype='U7')  # red raw
    colours[is_vice] = '#1f77b4'                           # blue VICE
    markers = np.where(is_vice, '*', 'o')
    texts = []
    vice_counter = 0
    # ── plot raw points, arrows, labels ──────────────────────────────────────
    for i in np.where(is_raw)[0]:
        x, y  = coords[i]
        col   = colours[i]
        mk    = markers[i]
        bname = base[i]
        pretty = formal_names_dict.get(bname, bname)

        # inside the for-loop over raw points
        facecol = col                      # default: use colour table
        # (no special case for '*'; let VICE be blue)
        ax.scatter(x, y,
                s=MARKER_BASE * (5 if mk == '*' else 0.8),
                marker=mk,
                facecolor=facecol,
                edgecolor='k', linewidth=LINE_THICK, zorder=10)


        # arrow raw → post
        if bname in post_coord:
            ax.annotate('', xy=post_coord[bname], xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color='#2ca02c',
                                        lw=LINE_THICK*1.5),
                        zorder=3)
    
        if bname != "VICE":
            pos, ang = label_behind_perp(np.array([x, y]),
                                post_coord.get(bname),
                                span, back_frac=0.03)
            if bname in ["Google_ViT_Large"]:
                pos[1] = pos[1] + 0.01
                pos[0] = pos[0] 
            elif bname in ["Deit3_Huge","DINOv2_ViT_Large"]:
                pos[1] = pos[1] + 0.02
                pos[0] = pos[0] - 0.01
            elif bname in ["VicReg_RN50","DenseNet201","Beit_V2_Large"]:
                pos[1] = pos[1] - 0.015
                pos[0] = pos[0] - 0.02
            elif bname in ["BLIP_2","SWAV_RN50"]:
                pos[1] = pos[1] - 0.025
            elif bname in ["AlexNet"]:
                pos[0] = pos[0] - 0.035
            elif bname in ["EVA02_CLIP_Enormous","ConvNext_Large","EfficientNet_B7"]:
                pos[1] = pos[1] + 0.01
            elif bname in ["SigLIP2_Base_P16_224"]:
                pos[1] = pos[1] + 0.015
                pos[0] = pos[0] - 0.025
            elif bname in ["Image_Bind_Huge"]:
                pos[1] = pos[1] + 0.015
                pos[0] = pos[0] + 0.015
            elif bname in ["VGG19"]:
                pos[1] = pos[1] - 0.015
            elif bname in ["CORnet_S"]:
                pos[0] = pos[0] - 0.035
                pos[1] = pos[1] - 0.025
                
            text = ax.text(pos[0], pos[1], pretty,
                ha='center', va='center',
                rotation=0, rotation_mode='default',weight='bold',
                fontsize=3, color='black', zorder=1)
            texts.append(text)
        
        elif vice_counter == 0:
            text = ax.text(x, y+0.03, pretty,
                ha='center', va='center',
                rotation=0, rotation_mode='default',weight='bold',
                fontsize=4, color='black', zorder=2)
            texts.append(text)
            vice_counter += 1

    handles = [Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='#d62728', markeredgecolor='k',
                    markeredgewidth=LINE_THICK, markersize=4,
                    label='Original Representations'),
            proxy_arrow,
            ]
    
    ax_legend.legend(handles=handles, loc='center', ncol=3,
                        labels=['Original Representations','Transformed Representations'],
                    fontsize=6, frameon=True, framealpha=.9,
                                handler_map={mpatches.FancyArrowPatch:
                    HandlerPatch(patch_func=_legend_open_arrow)})
    ax_legend.set_zorder(1)
    ax.margins(0)
    ax.set_aspect('equal', adjustable='box')
    ax.relim()                 # recompute the data limits based on your artists
    ax.autoscale_view(tight=True)  # apply them immediately, tightly
    # =============================================================================
    # Save Figure
    # =============================================================================
    
    # Construct output directory name
    output_subdir = f"{features_dim}_dim_{constraints}"
    if cfg.include_additional_models:
        output_subdir += "_additional_models"
    
    # Create output directory if it doesn't exist
    saving_dirs_path = os.path.join(cfg.output_dir, output_subdir)
    os.makedirs(saving_dirs_path, exist_ok=True)
    
    # Save as high-resolution PDF
    pdf_path = os.path.join(saving_dirs_path, f"{FIG_BASENAME}.pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight', pad_inches=0)
    
    # =============================================================================
    # Summary Report
    # =============================================================================
    
    print(f"\n{'='*70}")
    print(f"Figure 4 MDS - Generation Complete")
    print(f"{'='*70}")
    print(f"Version: {version_name}")
    print(f"Output file: {pdf_path}")
    print(f"Number of models displayed: {len([b for b in base if b != 'VICE']) + 1}")  # +1 for VICE
    print(f"Figure size: {FIG_SIZE_CM[0]:.1f} × {FIG_SIZE_CM[1]:.1f} cm")
    print(f"Resolution: 600 DPI")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
