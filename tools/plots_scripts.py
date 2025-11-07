import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pickle
import os
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path
import colorsys

def generate_distinct_colors(n: int) -> List[str]:
    """
    Generate n visually distinct colors.
    Uses a predefined set of distinct colors first, then generates additional ones if needed.
    """
    # Predefined distinct colors (carefully chosen for maximum distinction)
    base_colors = [
        '#FF0000',  # Red
        '#0000FF',  # Blue
        '#00FF00',  # Green
        '#FF00FF',  # Magenta
        '#00FFFF',  # Cyan
        '#FF8C00',  # Dark Orange
        '#800080',  # Purple
        '#008000',  # Dark Green
        '#000080',  # Navy
        '#800000',  # Maroon
        '#FF69B4',  # Hot Pink
        '#4B0082',  # Indigo
        '#556B2F',  # Dark Olive Green
        '#8B4513',  # Saddle Brown
        '#483D8B',  # Dark Slate Blue
    ]

    if n <= len(base_colors):
        return base_colors[:n]

    # If we need more colors, generate them with maximum distinction
    def hsv_to_hex(h, s, v):
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'

    additional_colors = []
    for i in range(n - len(base_colors)):
        hue = i / (n - len(base_colors))
        sat = 0.9
        val = 0.9
        additional_colors.append(hsv_to_hex(hue, sat, val))

    return base_colors + additional_colors

def save_plot(fig: go.Figure, filename: str, directory: str = "Results/plots"):
    """Save plotly figure to file"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    fig.write_html(f"{directory}/{filename}.html")
    fig.write_image(f"{directory}/{filename}.png")

def load_reg_results(reg_method: str) -> Dict:
    """Load regularization results from pickle file"""
    results_path = f"03_Results/regularization_methods_compare/models_results_{reg_method}.pkl"
    with open(results_path, "rb") as f:
        return pickle.load(f)

def plot_single_regularization(results: Dict, reg_method: str):
    """Plot accuracies for a single regularization method."""
    # Create subplots
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=(
                            f'{reg_method} Training Accuracy', 
                            f'{reg_method} Validation Accuracy'
                        ))

    # Generate colors
    n_models = len(results)
    colors = generate_distinct_colors(n_models)

    # Plot each model
    for model_idx, (model_name, lambda_results) in enumerate(results.items()):
        lambdas = list(lambda_results.keys())
        train_accs = [results['train_acc'] for results in lambda_results.values()]
        val_accs = [results['val_acc'] for results in lambda_results.values()]
        
        # Add training accuracy
        fig.add_trace(
            go.Scatter(x=lambdas, y=train_accs, 
                      name=f'{model_name}',
                      legendgroup=f'{model_name}',
                      line=dict(color=colors[model_idx]),
                      showlegend=True),
            row=1, col=1
        )
        # Add validation accuracy
        fig.add_trace(
            go.Scatter(x=lambdas, y=val_accs, 
                      name=f'{model_name}',
                      legendgroup=f'{model_name}',
                      line=dict(color=colors[model_idx]),
                      showlegend=False),
            row=1, col=2
        )

    # Add chance level line
    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=[min(lambdas), max(lambdas)],
                y=[1/3, 1/3],
                name="Chance level",
                legendgroup="Chance level",
                line=dict(color='#666666', dash='dash'),
                showlegend=(col == 1)
            ),
            row=1, col=col
        )

    # Update layout
    fig.update_layout(
        width=1500,
        height=400,
        showlegend=True,
        template="simple_white",
        hovermode='x unified',
        margin=dict(t=100, b=50, l=50, r=50),
        grid=dict(
            rows=1,
            columns=2,
            pattern='independent',
            xgap=0.2
        ),
        legend=dict(
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5,
            orientation="h",
            font=dict(size=12)
        )
    )

    # Update axes
    for col in [1, 2]:
        # X axis
        fig.update_xaxes(
            title_text="Lambda",
            type="log",
            ticktext=['0.001', '0.003', '0.01', '0.03', '0.1', '0.3', '1', '3', '10', '30', '100', '300', '1k', '3k', '10k'],
            tickvals=lambdas,
            title_font=dict(size=14),
            tickfont=dict(size=12),
            row=1,
            col=col
        )
        
        # Y axis with custom spacing
        y_ticks = (
            [0.25, 0.3] +  # Start values
            list(np.arange(0.35, 0.7, 0.05)) +  # 0.05 steps from 0.35 to 0.7
            list(np.arange(0.7, 1.1, 0.1))  # 0.1 steps from 0.7 to 1.0
        )
        
        fig.update_yaxes(
            title_text="Accuracy",
            range=[0.25, 1.0],
            tickmode='array',
            tickvals=y_ticks,
            ticktext=[f'{x:.2f}' for x in y_ticks],
            title_font=dict(size=14),
            tickfont=dict(size=12),
            row=1,
            col=col
        )

    return fig

def plot_regularization_comparison(results1: Dict, results2: Dict, reg_methods: List[str]):
    """Plot comparison between two regularization methods."""
    # Verify that both dictionaries have the same models
    if set(results1.keys()) != set(results2.keys()):
        raise ValueError("Both regularization methods must have the same models")

    # Create subplots
    fig = make_subplots(rows=3, cols=2, 
                       subplot_titles=(
                           f'{reg_methods[0]} Training Accuracy', 
                           f'{reg_methods[0]} Validation Accuracy',
                           f'{reg_methods[1]} Training Accuracy', 
                           f'{reg_methods[1]} Validation Accuracy',
                           f'Training Accuracy Difference\n({reg_methods[1]} - {reg_methods[0]})',
                           f'Validation Accuracy Difference\n({reg_methods[1]} - {reg_methods[0]})'
                       ))

    # Generate colors
    n_models = len(results1)
    colors = generate_distinct_colors(n_models)

    # Plot each model
    for model_idx, model_name in enumerate(results1.keys()):
        # First regularization method
        lambdas1 = list(results1[model_name].keys())
        train_accs1 = [results['train_acc'] for results in results1[model_name].values()]
        val_accs1 = [results['val_acc'] for results in results1[model_name].values()]
        
        # Second regularization method
        lambdas2 = list(results2[model_name].keys())
        train_accs2 = [results['train_acc'] for results in results2[model_name].values()]
        val_accs2 = [results['val_acc'] for results in results2[model_name].values()]

        # Plot first method
        fig.add_trace(
            go.Scatter(x=lambdas1, y=train_accs1, 
                      name=f'{model_name}',
                      legendgroup=f'{model_name}',
                      line=dict(color=colors[model_idx]),
                      showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=lambdas1, y=val_accs1, 
                      name=f'{model_name}',
                      legendgroup=f'{model_name}',
                      line=dict(color=colors[model_idx]),
                      showlegend=False),
            row=1, col=2
        )

        # Plot second method
        fig.add_trace(
            go.Scatter(x=lambdas2, y=train_accs2, 
                      name=f'{model_name}',
                      legendgroup=f'{model_name}',
                      line=dict(color=colors[model_idx]),
                      showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=lambdas2, y=val_accs2, 
                      name=f'{model_name}',
                      legendgroup=f'{model_name}',
                      line=dict(color=colors[model_idx]),
                      showlegend=False),
            row=2, col=2
        )

        # Plot differences
        train_diff = [t2 - t1 for t1, t2 in zip(train_accs1, train_accs2)]
        val_diff = [v2 - v1 for v1, v2 in zip(val_accs1, val_accs2)]

        fig.add_trace(
            go.Scatter(x=lambdas1, y=train_diff,
                      name=f'{model_name}',
                      legendgroup=f'{model_name}',
                      line=dict(color=colors[model_idx]),
                      showlegend=False),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=lambdas1, y=val_diff,
                      name=f'{model_name}',
                      legendgroup=f'{model_name}',
                      line=dict(color=colors[model_idx]),
                      showlegend=False),
            row=3, col=2
        )

    # Add chance level lines for all plots except difference plots
    for row in [1, 2]:
        for col in [1, 2]:
            fig.add_trace(
                go.Scatter(
                    x=[min(lambdas1), max(lambdas1)],
                    y=[1/3, 1/3],
                    name="Chance level",
                    legendgroup="Chance level",
                    line=dict(color='#666666', dash='dash'),
                    showlegend=(row == 1 and col == 1)
                ),
                row=row, col=col
            )

    # Update layout
    fig.update_layout(
        width=1500,
        height=1200,
        showlegend=True,
        template="simple_white",
        hovermode='x unified',
        margin=dict(t=100, b=50, l=50, r=50),
        grid=dict(
            rows=3,
            columns=2,
            pattern='independent',
            xgap=0.2,
            ygap=0.3
        ),
        legend=dict(
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5,
            orientation="h",
            font=dict(size=12)
        )
    )

    # Update axes for each subplot
    for row in range(1, 4):
        for col in [1, 2]:
            # X axis
            fig.update_xaxes(
                title_text="Lambda",
                type="log",
                ticktext=['0.001', '0.003', '0.01', '0.03', '0.1', '0.3', '1', '3', '10', '30', '100', '300', '1k', '3k', '10k'],
                tickvals=lambdas1,
                title_font=dict(size=14),
                tickfont=dict(size=12),
                row=row,
                col=col
            )
            
            # Y axis with custom spacing (different for difference plots)
            if row < 3:
                y_ticks = (
                    [0.25, 0.3] +  # Start values
                    list(np.arange(0.35, 0.7, 0.05)) +  # 0.05 steps from 0.35 to 0.7
                    list(np.arange(0.7, 1.1, 0.1))  # 0.1 steps from 0.7 to 1.0
                )
                y_range = [0.25, 1.0]
            else:
                # For difference plots
                y_ticks = np.arange(-0.5, 0.55, 0.05)
                y_range = [-0.5, 0.5]
            
            fig.update_yaxes(
                title_text="Accuracy" if row < 3 else "Accuracy Difference",
                range=y_range,
                tickmode='array',
                tickvals=y_ticks,
                ticktext=[f'{x:.2f}' for x in y_ticks],
                title_font=dict(size=14),
                tickfont=dict(size=12),
                row=row,
                col=col
            )

    return fig

def plot_accuracies_hyperparameters(model_results: Union[Dict, List[Dict]], reg_methods: Optional[List[str]] = None):
    """Router function to appropriate plotting function based on input."""
    if isinstance(model_results, dict):
        return plot_single_regularization(model_results, reg_methods[0])
    elif isinstance(model_results, list):
        assert len(model_results) == 2, "Expected a list of two dictionaries for comparison"
        return plot_regularization_comparison(model_results[0], model_results[1], reg_methods)

# Example usage with fake data
# if __name__ == "__main__":
#     # Load results
#     L2_results = load_reg_results("L2")
#     eye_distance_results = load_reg_results("eye_distance")
    
#     # Create output directory
#     plots_dir = "Results/plots/regularization_comparison"
#     Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
#     # Create individual method plots
#     fig_L2 = plot_accuracies_hyperparameters(L2_results, reg_methods=['L2'])
#     fig_eye_distance = plot_accuracies_hyperparameters(eye_distance_results, reg_methods=['eye_distance'])
    
#     # Create comparison plot
#     fig_comparison = plot_accuracies_hyperparameters(
#         [L2_results, eye_distance_results],
#         reg_methods=['L2', 'eye_distance']
#     )
    
#     # Save all plots
#     save_plot(fig_L2, "L2_accuracies", directory=plots_dir)
#     save_plot(fig_eye_distance, "eye_distance_accuracies", directory=plots_dir)
#     save_plot(fig_comparison, "regularization_methods_comparison", directory=plots_dir)
    
#     # Show plots
#     fig_L2.show()
#     fig_eye_distance.show()
#     fig_comparison.show()

