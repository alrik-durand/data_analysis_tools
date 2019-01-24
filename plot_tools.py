""" Helper functions useful to easily plot stuff

"""
import numpy as np
import matplotlib.pyplot as plt
import data_analysis_tools.data as dat
import pandas as pd


def plot_simple_data(x, y, ax=None, **kw):
    """
    Function to plot simple graph

    :param x: (array-like or scalar) data to plot on the x_axis
    :param y: (array-like or scalar) data to plot on the y_axis
    :param ax: (optional) the axe object for matplotlib.pyplot
    :param keywords: (optional) standard keyword arguments to change the appearance of the figure

    :return : array of the plotted data
    """

    figsize      = kw.pop('figsize', (10, 6))
    x_range      = kw.pop('x_range', None)
    y_range      = kw.pop('y_range', None)
    color        = kw.pop('color', 'r')
    linestyle    = kw.pop('linestyle', '-')
    marker       = kw.pop('marker', None)
    marker_s     = kw.pop('marker_size', 6)
    x_label      = kw.pop('x_label', '')
    y_label      = kw.pop('y_label', '')
    show_legend  = kw.pop('show_legend', False)
    legend_label = kw.pop('legend_label', 'plot')
    title        = kw.pop('title', '')
    label_fs     = kw.pop('label_fontsize', 14)
    legend_fs    = kw.pop('legend_fontsize', 14)
    title_fs     = kw.pop('title_fontsize', 15)
    tick_fs      = kw.pop('tick_fontsize', 12)

    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)

    line = ax.plot(x, y,
            color=color, linestyle=linestyle, marker=marker, markersize=marker_s,
            label = legend_label)

    if x_range != None:
        ax.set_xlim(x_range)
    if y_range != None:
        ax.set_ylim(y_range)

    ax.set_xlabel(x_label, fontsize=label_fs)
    ax.set_ylabel(y_label, fontsize=label_fs)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    ax.set_title(title, fontsize=title_fs)
    if show_legend :
        ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=2., fontsize = legend_fs)

    return line


def plot_dataframe(df, x_key='x', y_key='y', ax=None, rebin_integer=1, **kw):
    """
    Function to plot a Pandas dataframe

    :param df: (Pandas Dataframe) the dataframe to plot
    :param x_key: (str) the label of the dataframe column to plot on the x_axis
    :param y_key: (str) the label of the dataframe column to plor on the y_axis
    :param ax: the axe object from matplotlib.pyplot
    :param rebin_integer: (int) the integer to rebin the data; use dat.rebin function
    :param kw: the keywords to pass to the plot_simple_data function to choose the appearance of the plot figure
    :return : array of the plotted data
    """
    cmap               = kw.pop('cmap', plt.cm.viridis)
    color_scale        = kw.pop('color_scale', None)
    legend_label       = kw.pop('legend_label', '')
    legend_fs          = kw.pop('legend_fontsize', 14)
    rebin_with_average = kw.pop('rebin_with_average', False)


    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)

    if len(legend_label) == 0:
        show_legend = False
    else :
        show_legend = True

    lines = []
    for i, row in df.iterrows():
        if len(color_scale) == 0:
            kw['color'] = cmap(i / len(df))
        else:
            kw['color'] = cmap(color_scale[i] / max(color_scale))


        if show_legend :
            kw['legend_label'] = legend_label[i]
            kw['show_legend']  = True

        y_rebin = dat.rebin(row[y_key], int(rebin_integer), do_average=rebin_with_average)
        x_rebin = dat.decimate(row[x_key], int(rebin_integer))
        line = plot_simple_data(x_rebin, y_rebin, ax=ax, **kw)
        lines.append(line)

    handles, labels = ax.get_legend_handles_labels()
    _dummy, labels, handles = zip(*sorted(zip(color_scale, labels, handles), key=lambda t: t[0]))

    compact_labels = []
    compact_handles = []
    for i in np.arange(len(labels)):
        if i == 0:
            compact_labels.append(labels[i])
            compact_handles.append(handles[i])
        else:
            if labels[i] != labels[i - 1]:
                compact_labels.append(labels[i])
                compact_handles.append(handles[i])

    ax.legend(compact_handles, compact_labels, loc="center left", bbox_to_anchor=(1.1, 0, 0.5, 1), fontsize=legend_fs)

    return lines



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

