import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TimeSeriesVisualizer:
    def __init__(self):
        """
        Initialize the TimeSeriesVisualizer class.
        This class provides methods to visualize time series data with scatter plots and line plots.
        """
        pass

    def plot_series_scatter(self, series_list, save=False, filename_prefix="scatter_plot"):
        """
        Plot time series data with scatter markers for specific features.

        Parameters:
            series_list (list): A list of dataframe, where each dataframe contains the time series data
                                and optional features like 'mean_shift', 'variance_shift', 'trend_shift', and 'anomaly'.
            save (bool): Whether to save the plot to a file. Default is False.
            filename_prefix (str): Prefix for the saved file name. Default is "scatter_plot".
        """
        for i, series in enumerate(series_list):
            plt.figure(figsize=(15, 5))
            plt.plot(series['data'], label='Base Series', color='black')
            if 'mean_shift' in series and np.any(series['mean_shift'] != 0):
                mean_shift_indices = np.where(series['mean_shift'] == 1)[0]
                plt.scatter(mean_shift_indices, series['data'][mean_shift_indices], label='Mean Shift', color='green', marker='o')
            if 'variance_shift' in series and np.any(series['variance_shift'] != 0):
                variance_shift_indices = np.where(series['variance_shift'] == 1)[0]
                plt.scatter(variance_shift_indices, series['data'][variance_shift_indices], label='Variance Shift', color='red', marker='o')
            if 'trend_shift' in series and np.any(series['trend_shift'] != 0):
                trend_shift_indices = np.where(series['trend_shift'] == 1)[0]
                plt.scatter(trend_shift_indices, series['data'][trend_shift_indices], label='Trend Shift', color='blue', marker='o')
            if 'anomaly' in series and np.any(series['anomaly'] != 0):
                anomaly_indices = np.where(series['anomaly'] == 1)[0]
                plt.scatter(anomaly_indices, series['data'][anomaly_indices], label='Anomaly', color='orange', marker='o')
            plt.title(f'Series {i + 1} with Scatter Markers')
            plt.legend()
            if save:
                filename = f"{filename_prefix}_series_{i + 1}.png"
                plt.savefig(filename, bbox_inches='tight')
            plt.show()

    def plot_series_lines(self, series_list, save=False, filename_prefix="line_plot"):
        """
        Plot time series data with line plots for specific features.

        Parameters:
            series_list (list): A list of dataframe, where each dataframe contains the time series data
                                and optional features like 'mean_shift', 'variance_shift', 'trend_shift', and 'anomaly'.
            save (bool): Whether to save the plot to a file. Default is False.
            filename_prefix (str): Prefix for the saved file name. Default is "line_plot".
        """
        for i, series in enumerate(series_list):
            plt.figure(figsize=(15, 5))
            plt.plot(series['data'], label='Base Series', color='black')
            if 'mean_shift' in series and np.any(series['mean_shift'] != 0):
                plt.plot(series['mean_shift'], label='Mean Shift', color='green')
            if 'variance_shift' in series and np.any(series['variance_shift'] != 0):
                plt.plot(series['variance_shift'], label='Variance Shift', color='red')
            if 'trend_shift' in series and np.any(series['trend_shift'] != 0):
                plt.plot(series['trend_shift'], label='Trend Shift', color='blue')
            if 'anomaly' in series and np.any(series['anomaly'] != 0):
                plt.plot(series['anomaly'], label='Anomaly', color='orange')
            plt.title(f'Series {i + 1} with Line Plots')
            plt.legend()
            if save:
                filename = f"{filename_prefix}_series_{i + 1}.png"
                plt.savefig(filename, bbox_inches='tight')
            plt.show()