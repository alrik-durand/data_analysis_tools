""" Helper functions useful to easily plot stuff

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import data_analysis_tools.data as dat
import pandas as pd


def set_rc_params(mpl):
    """ Helper method to set the right sizes in matplotlib """
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.titlesize'] = 15
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 13
    mpl.rcParams['xtick.labelsize'] = 13
    mpl.rcParams['ytick.labelsize'] = 13
    mpl.rcParams['figure.facecolor'] = (0, 0, 0, 0)


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
    rebin_integer= kw.pop('rebin_integer', 1)
    rebin_with_average = kw.pop('rebin_with_average', True)

    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)

    y_rebin = dat.rebin(y, int(rebin_integer), do_average=rebin_with_average)
    x_rebin = dat.decimate(x, int(rebin_integer))

    line = ax.plot(x_rebin, y_rebin,
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
    rebin_with_average = kw.pop('rebin_with_average', True)
    normalize_colors   = kw.pop('normalize_colors', True)


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
            if normalize_colors:
                kw['color'] = cmap(color_scale[i] / max(color_scale))
            else:
                kw['color'] = cmap(color_scale[i])


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


def plot_data(ax, df, rebin_ratio=1, colors=None, cmap=None, window=None, x='x', y='y', plot_kw={},
              remove_label_doubles=True, label=None, offset_increment=0, **test_dic):
    """ Helper function to plot PL traces of a dataframe

    @param Axe ax: The axe object from patplotlib.pyplot
    @param DataFrame df: The dataframe containing the 'x' and 'y' columns
    @param int rebin_ratio: A integer to rebin the data by a certain value
    @param colors: An array (same length as df) or the name of a column to compute the color based on max
    @param cmap: A color map
    @param (float, float) window: a window (x_min, x_max) to plot only part of the data
    @param the name of the column to use as x values
    @param the name of the column to use as y values
    @param plot_kw: A dictionary passed to the plot function
    @param bool remove_label_doubles: True to prevent multiple occurrences of the same label
    @param dict test_dic: A dictionary of (key, value) to plot only some rows where df[key]=value

    @return array of lines created by plot() function
    """
    lines = []
    label_set = set()
    j = 0
    for i, row in df.iterrows():
        show = True
        for key in test_dic:
            if type(test_dic[key]) is tuple:
                if not test_dic[key][0] < row[key] < test_dic[key][1]:
                    show = False
            elif type(test_dic[key]) is not list:
                if row[key] != test_dic[key]:
                    show = False
            else:
                if row[key] not in test_dic[key]:
                    show = False
        if row[x] is None or row[y] is None:
            show = False
        if show:

            y_decimated = dat.rebin(row[y], int(rebin_ratio), do_average=True)
            x_decimated = dat.decimate(row[x], int(rebin_ratio))

            color = None
            cmap = cmap if cmap else plt.cm.viridis
            if colors is not None and type(colors) == str:
                if type(row[colors])==str:
                    color = row[colors]
                else:
                    n = int(row[colors]/max(df[colors])*256)
                    color = cmap(n)
            if colors is not None and \
                    (type(colors) == pd.core.series.Series or type(colors) == list or type(colors) == np.ndarray):
                n = int(colors[i]/max(colors)*256)
                color = cmap(n)
            if row.get('color'):
                color = cmap(row.get('color'))
            if label is not None:
                row_dict = row.to_dict()
                label_row = label.format(**row_dict)
            elif 'label' in row.index:
                label_row = row['label']
            else:
                label_row = None
            if remove_label_doubles and label_row is not None and label_row in label_set:
                label_row = None
            label_set.update([label_row])
            if window is not None:
                x_data, y_data = dat.get_window(x_decimated, y_decimated, *window)
            else:
                x_data, y_data = x_decimated, y_decimated
            plot_kw_row = plot_kw.copy()
            if 'plot_kw' in row.keys() and isinstance(row['plot_kw'], dict):
                plot_kw_row.update(row['plot_kw'])
            if 'color' in plot_kw:
                line = ax.plot(x_data, y_data + j * offset_increment, label=label_row, **plot_kw_row)
            else:
                line = ax.plot(x_data, y_data + j * offset_increment, label=label_row, color=color, **plot_kw_row)
            j += 1
            lines.append(line)
    return lines


def plot_grid(data, lines_key, columns, x_label=None, y_label=None, height_per_line=4, width_per_column=5,
              rebin_ratio=1, x_label_all=False, y_label_all=False, line_ascending=True, cmap=None, ncol=1,
              x_lim=None, y_lim=None, plot_kw={}):
    """

    @param DataFrame df: The dataframe containing the 'x' and 'y' columns
    @param str lines_key: The name of the column on witch to iterate to create the lines
    @param list(dict) columns: An array of dictionaries describing the content that need to be plotted (see example)
    @param str x_label: The label for every x axis
    @param y_label: The label for every y axis
    @param height_per_line: The height per line (in matplotlib unit)
    @param width_per_column: The width per column (in matplotlib unit)
    @param rebin_ratio: A integer to rebin the data by a certain value
    @param x_label_all: Show the x label for every line instead of just the last
    @param y_label_all: Show the y label for every columns instead of just the first
    @param line_ascending: Set to true for ascending order
    @param cmap: the cmap
    @param ncol: The number of columns for the legend
    @param (float, float) x_lim: The x limit of the plot
    @param (float, float) y_lim: The x limit of the plot
    @param plot_kw: Parameters transferred to matplotlib's plot function

    Example :
        y_keys = [{'x': 'x_0', 'y': 'y_0', 'text':'$m_s = 0$\n$P_{{read}}=$ {:.0f} µW ',
               'text_kw': {'x': 0.5, 'y': 0.8} },
              {'x': 'x_1', 'y': 'y_1_norm', 'text':'$m_s = \pm 1$\n$P_{{read}}=$ {:.0f} µW',
               'text_kw': {'x': 0.5, 'y': 0.8}}
             ]

         fig, axes = pltools.plot_grid(data, 'power_read_u', y_keys, x_label='Time [us]', y_label = "PL")

    """
    lines_values = np.sort(np.array(list(set(data[lines_key]))))
    if not line_ascending:
        lines_values = np.flip(lines_values, axis=0)

    h = len(lines_values)
    w = len(columns)

    fig, axes = plt.subplots(h, w, figsize=(width_per_column * w, height_per_line * h))

    for i, line in enumerate(axes):
        line_value = lines_values[i]
        df = data[data[lines_key] == line_value]

        for j, ax in enumerate(line):
            column = columns[j]
            if x_label_all or i == h - 1:
                ax.set_xlabel(x_label)
            if y_label_all or j == 0:
                ax.set_ylabel(y_label)
            if x_lim is not None:
                    ax.set_xlim(x_lim)
            if y_lim is not None:
                ax.set_ylim(y_lim)

            cmap_line = column.get('cmap') if column.get('cmap') else cmap

            plot_data(ax, df, x=column['x'], y=column['y'], rebin_ratio=rebin_ratio, cmap=cmap_line, plot_kw=plot_kw)

            text = None
            if column.get('text'):
                text = column.get('text').format(line_value)
            if text:
                text_kw = column.get('text_kw') if column.get('text_kw') else {}
                ax.text(s=text, transform=ax.transAxes, horizontalalignment='center', fontsize=13, **text_kw)

            if j == w - 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.03, 0, 0.5, 1), fontsize=13, ncol=ncol)

    return fig, axes


def new_figure(xlabel=None, ylabel=None, figsize=None):
    """  Helper to create quickly a new plot"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def gradient_fill(x, y, fill_color=None, ax=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im


def gradient_fill_exp(x, y, fill_color=None, ax=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = 1-np.exp(-np.linspace(0, 6, 100)[:,None])

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im
