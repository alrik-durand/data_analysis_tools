""" Helper functions useful to easily plot stuff

"""
import numpy as np
import matplotlib.pyplot as plt
import data_analysis_tools.data as dat
import pandas as pd


def plot_data(ax, df, rebin_ratio=1, colors=None, cmap=plt.cm.viridis, window=None, x='x', y='y', **test_dic):
    """ Helper function to plot PL traces of a dataframe

    @param Axe ax: The axe object from patplotlib.pyplot
    @param DataFrame df: The dataframe containing the 'x' and 'y' columns
    @param int rebin_ratio (optional): A integer to rebin the data by a certain value
    @param colors (optional): An array (same length as df) or the name of a column to compute the color based on max
    @param cmap (optional): A color map
    @param (float, float) window (optional): a window (x_min, x_max) to plot only part of the data
    @param the name of the column to use as x values
    @param the name of the column to use as y values
    @param diect test_dic (optional): A dictionary of (key, value) to plot only some rows where df[key]=value

    @return array of lines created by plot() function
    """
    lines = []
    for i, row in df.iterrows():
        show = True
        for key in test_dic:
            if row[key] != test_dic[key]:
                show = False
        if show:
            y_decimated = dat.rebin(row[y], int(rebin_ratio), do_average=True)
            x_decimated = dat.decimate(row[x], int(rebin_ratio))

            color = None
            if colors is not None and type(colors) == str:
                n = int(row[colors]/max(df[colors])*256)
                color = cmap(n)
            if colors is not None and \
                    (type(colors) == pd.core.series.Series or type(colors) == list or type(colors) == np.ndarray):
                n = int(colors[i]/max(colors)*256)
                color = cmap(n)
            label = row['label'] if 'label' in row.index else None
            if window is not None:
                x_data, y_data = dat.get_window(x_decimated, y_decimated, *window)
            else:
                x_data, y_data = x_decimated, y_decimated
            line = ax.plot(x_data, y_data, label=label, color=color)
            lines.append(line)
    return lines

