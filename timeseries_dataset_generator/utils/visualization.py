"""
Visualization utilities for time series datasets.

This module provides helper functions for visualizing time series data,
including plotting individual series, comparing multiple series, and
highlighting anomalies and structural breaks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict, Union
import ast
import re


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_single_series(
    df: pd.DataFrame,
    series_id: Optional[int] = None,
    title: Optional[str] = None,
    highlight_anomalies: bool = True,
    highlight_breaks: bool = True,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a single time series with optional anomaly and structural break highlighting.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id : int, optional
        ID of the series to plot. If None, plots the first series
    title : str, optional
        Custom title for the plot
    highlight_anomalies : bool, default=True
        Whether to highlight anomalies
    highlight_breaks : bool, default=True
        Whether to highlight structural breaks
    figsize : tuple, default=(14, 6)
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    if series_id is None:
        series_id = df['series_id'].iloc[0]
    
    series_data = df[df['series_id'] == series_id].copy()
    
    if series_data.empty:
        raise ValueError(f"Series ID {series_id} not found in DataFrame")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the main series
    ax.plot(series_data['time'], series_data['data'], 
            label='Time Series', linewidth=1.5, color='steelblue')
    
    if highlight_anomalies or highlight_breaks:
        mark_anom_and_breaks(series_data, highlight_anomalies, highlight_breaks,ax)
         
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    if title is None:
        label = series_data['label'].iloc[0] if 'label' in series_data.columns else 'Unknown'
        title = f'Time Series (ID: {series_id}, Label: {label})'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

def mark_anom_and_breaks(
    series_data,
    highlight_anomalies: bool = True,
    highlight_breaks: bool = True,
    ax=None):

    # Highlight anomalies
    if highlight_anomalies:
        # Point anomalies
        if series_data["primary_label"].iloc[0] == 1 and series_data["sub_label"].iloc[0] == 0:
            x = series_data["time"].to_numpy()
            anom = series_data["anomaly_indices"].dropna().iloc[0]

            if isinstance(anom, str):
                anom = ast.literal_eval(anom)

            anom = np.asarray(anom, dtype=int)

            # keep only valid indices (avoid crashes)
            anom = anom[(anom >= 0) & (anom < len(x))]

            # plot anomaly points as red dots
            ax.scatter(series_data["time"].iloc[anom],series_data["data"].iloc[anom],
            color="red", s=25, zorder=5, label="Point Anomaly")

        
        # Collective anomalies
        if series_data["primary_label"].iloc[0] == 1 and series_data["sub_label"].iloc[0] == 1:
            x = series_data["time"].to_numpy()
            anom = series_data["anomaly_indices"].dropna().iloc[0]
    
            if isinstance(anom, str):
                anom = ast.literal_eval(anom)

            starts, ends = anom

            labeled = False
            for s, e in zip(starts, ends):
                ax.axvspan(x[s], x[e], alpha=0.15, color="orange",
                           label="Collective Anomaly" if not labeled else None)
                labeled = True
        
        # Contextual anomalies
        if series_data["primary_label"].iloc[0] == 1 and series_data["sub_label"].iloc[0] == 2:
            x = series_data["time"].to_numpy()
            anom = series_data["anomaly_indices"].dropna().iloc[0]
    
            if isinstance(anom, str):
                anom = ast.literal_eval(anom)

            starts, ends = anom

            labeled = False
            for s, e in zip(starts, ends):
                ax.axvspan(x[s], x[e], alpha=0.15, color="orange",
                           label="Contextual Anomaly" if not labeled else None)
                labeled = True
    
    # Highlight structural breaks
    if highlight_breaks:
        if series_data["primary_label"].iloc[0] ==6:
            x = series_data["time"].to_numpy()
            n = len(x)
            bs = series_data["break_indices"].iloc[0]
            if isinstance(bs, str):
                breaks = list(map(int, re.findall(r"int\d+\((-?\d+)\)", bs)))
                if not breaks:
                    breaks = list(map(int, re.findall(r"-?\d+", bs)))
            else:
                breaks = [int(b) for b in bs]

            breaks = sorted(set(b for b in breaks if 0 <= b < n))
            labeled = False
            if series_data["sub_label"].iloc[0] == 0:
                label = "Mean Shift"
            elif series_data["sub_label"].iloc[0] == 1:
                label = "Variance Shift"
            elif series_data["sub_label"].iloc[0] == 2:
                label = "Trend Shift"
            for k, b in enumerate(breaks):
                alpha = min(0.06 + 0.06 * k, 0.35)  # darker as k increases
                ax.axvspan(x[b], x[-1], color="orange", alpha=alpha,
                        label=f"{label}" if not labeled else None)
                labeled = True

            # optional: mark exact start locations
            for k, b in enumerate(breaks):
                ax.axvline(x[b], color="gray", linestyle="--", linewidth=1,
                        label="Break start" if k == 0 else None)

def plot_multiple_series(
    df: pd.DataFrame,
    series_ids: Optional[List[int]] = None,
    n_series: int = 4,
    figsize: Tuple[int, int] = (14, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple time series in a grid layout.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_ids : list of int, optional
        List of series IDs to plot. If None, plots first n_series
    n_series : int, default=4
        Number of series to plot if series_ids is None
    figsize : tuple, default=(14, 10)
        Figure size (width, height)
    title : str, optional
        Overall title for the figure
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    if series_ids is None:
        unique_ids = df['series_id'].unique()
        series_ids = unique_ids[:min(n_series, len(unique_ids))]
    
    n_plots = len(series_ids)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, series_id in enumerate(series_ids):
        series_data = df[df['series_id'] == series_id]
        
        if not series_data.empty:
            ax = axes[idx]
            ax.plot(series_data['time'], series_data['data'], 
                   linewidth=1.2, color='steelblue')
            
            label = series_data['label'].iloc[0] if 'label' in series_data.columns else 'Unknown'
            ax.set_title(f'Series {series_id}: {label}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_series_comparison(
    df: pd.DataFrame,
    series_ids: List[int],
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    title: str = 'Time Series Comparison',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare multiple time series on the same plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_ids : list of int
        List of series IDs to compare
    labels : list of str, optional
        Custom labels for each series
    figsize : tuple, default=(14, 6)
        Figure size (width, height)
    title : str, default='Time Series Comparison'
        Title for the plot
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(series_ids)))
    
    for idx, series_id in enumerate(series_ids):
        series_data = df[df['series_id'] == series_id]
        
        if not series_data.empty:
            if labels is not None and idx < len(labels):
                label = labels[idx]
            else:
                label = f'Series {series_id}'
            
            ax.plot(series_data['time'], series_data['data'], 
                   label=label, linewidth=1.5, color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_distribution(
    df: pd.DataFrame,
    series_id: Optional[int] = None,
    bins: int = 50,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the distribution of a time series with histogram and KDE.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id : int, optional
        ID of the series to plot. If None, uses first series
    bins : int, default=50
        Number of bins for histogram
    figsize : tuple, default=(12, 5)
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    if series_id is None:
        series_id = df['series_id'].iloc[0]
    
    series_data = df[df['series_id'] == series_id]['data']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram with KDE
    axes[0].hist(series_data, bins=bins, alpha=0.6, color='steelblue', 
                 density=True, edgecolor='black', label='Histogram')
    
    # KDE plot
    series_data.plot.kde(ax=axes[0], linewidth=2, color='red', label='KDE')
    
    axes[0].set_xlabel('Value', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Distribution (Histogram + KDE)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(series_data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='steelblue'),
                    whiskerprops=dict(color='steelblue'),
                    capprops=dict(color='steelblue'),
                    medianprops=dict(color='red', linewidth=2))
    
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Box Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    label = df[df['series_id'] == series_id]['label'].iloc[0] if 'label' in df.columns else 'Unknown'
    fig.suptitle(f'Series {series_id} Distribution ({label})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_acf_pacf(
    df: pd.DataFrame,
    series_id: Optional[int] = None,
    lags: int = 40,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ACF and PACF for a time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id : int, optional
        ID of the series to plot. If None, uses first series
    lags : int, default=40
        Number of lags to show
    figsize : tuple, default=(14, 5)
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        raise ImportError("statsmodels is required for ACF/PACF plots")
    
    if series_id is None:
        series_id = df['series_id'].iloc[0]
    
    series_data = df[df['series_id'] == series_id]['data']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ACF
    plot_acf(series_data, lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # PACF
    plot_pacf(series_data, lags=lags, ax=axes[1], alpha=0.05)
    axes[1].set_title('Partial Autocorrelation Function (PACF)', 
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    label = df[df['series_id'] == series_id]['label'].iloc[0] if 'label' in df.columns else 'Unknown'
    fig.suptitle(f'Series {series_id} - ACF & PACF ({label})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_rolling_statistics(
    df: pd.DataFrame,
    series_id: Optional[int] = None,
    window: int = 20,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot rolling mean and standard deviation.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id : int, optional
        ID of the series to plot. If None, uses first series
    window : int, default=20
        Window size for rolling statistics
    figsize : tuple, default=(14, 8)
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    if series_id is None:
        series_id = df['series_id'].iloc[0]
    
    series_data = df[df['series_id'] == series_id].copy()
    
    # Calculate rolling statistics
    rolling_mean = series_data['data'].rolling(window=window).mean()
    rolling_std = series_data['data'].rolling(window=window).std()
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot original series with rolling mean
    axes[0].plot(series_data['time'], series_data['data'], 
                label='Original', linewidth=1, alpha=0.7, color='steelblue')
    axes[0].plot(series_data['time'], rolling_mean, 
                label=f'Rolling Mean (window={window})', 
                linewidth=2, color='red')
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_title('Time Series with Rolling Mean', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Plot rolling standard deviation
    axes[1].plot(series_data['time'], rolling_std, 
                linewidth=2, color='green')
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Std Dev', fontsize=12)
    axes[1].set_title(f'Rolling Standard Deviation (window={window})', 
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    label = series_data['label'].iloc[0] if 'label' in series_data.columns else 'Unknown'
    fig.suptitle(f'Series {series_id} - Rolling Statistics ({label})', 
                 fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_category_overview(
    df: pd.DataFrame,
    max_series: int = 9,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create an overview plot showing examples from different categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    max_series : int, default=9
        Maximum number of series to show
    figsize : tuple, default=(16, 12)
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Get unique labels
    if 'label' not in df.columns:
        print("Warning: No 'label' column found. Using series_id instead.")
        labels = df['series_id'].unique()[:max_series]
        use_labels = False
    else:
        labels = df['label'].unique()[:max_series]
        use_labels = True
    
    n_plots = min(len(labels), max_series)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, label in enumerate(labels):
        if idx >= max_series:
            break
        
        if use_labels:
            series_data = df[df['label'] == label].groupby('series_id').first()
            series_id = series_data.index[0]
            plot_data = df[df['series_id'] == series_id]
        else:
            plot_data = df[df['series_id'] == label]
        
        if not plot_data.empty:
            ax = axes[idx]
            ax.plot(plot_data['time'], plot_data['data'], 
                   linewidth=1.2, color='steelblue')
            
            ax.set_title(f'{label}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Time Series Categories Overview', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def create_dashboard(
    df: pd.DataFrame,
    series_id: Optional[int] = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive dashboard for a single time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    series_id : int, optional
        ID of the series to plot. If None, uses first series
    figsize : tuple, default=(16, 12)
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    if series_id is None:
        series_id = df['series_id'].iloc[0]
    
    series_data = df[df['series_id'] == series_id].copy()
    
    if series_data.empty:
        raise ValueError(f"Series ID {series_id} not found")
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Time series plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(series_data['time'], series_data['data'], 
            linewidth=1.5, color='steelblue')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('Time Series', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(series_data['data'], bins=30, alpha=0.6, color='steelblue', 
            edgecolor='black', density=True)
    series_data['data'].plot.kde(ax=ax2, linewidth=2, color='red')
    ax2.set_xlabel('Value', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.boxplot(series_data['data'], vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='steelblue'),
               whiskerprops=dict(color='steelblue'),
               capprops=dict(color='steelblue'),
               medianprops=dict(color='red', linewidth=2))
    ax3.set_ylabel('Value', fontsize=11)
    ax3.set_title('Box Plot', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. ACF
    try:
        from statsmodels.graphics.tsaplots import plot_acf
        ax4 = fig.add_subplot(gs[2, 0])
        plot_acf(series_data['data'], lags=min(40, len(series_data) // 2 - 1), 
                ax=ax4, alpha=0.05)
        ax4.set_title('ACF', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    except ImportError:
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.text(0.5, 0.5, 'statsmodels required for ACF', 
                ha='center', va='center')
        ax4.set_title('ACF', fontsize=12, fontweight='bold')
    
    # 5. Rolling statistics
    ax5 = fig.add_subplot(gs[2, 1])
    window = min(20, len(series_data) // 10)
    rolling_std = series_data['data'].rolling(window=window).std()
    ax5.plot(series_data['time'], rolling_std, linewidth=2, color='green')
    ax5.set_xlabel('Time', fontsize=11)
    ax5.set_ylabel('Std Dev', fontsize=11)
    ax5.set_title(f'Rolling Std (window={window})', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Overall title
    label = series_data['label'].iloc[0] if 'label' in series_data.columns else 'Unknown'
    fig.suptitle(f'Time Series Dashboard - Series {series_id} ({label})', 
                fontsize=16, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

