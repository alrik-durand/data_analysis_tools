import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import lmfit

# Fake data generation
#
# x = np.linspace(0, 1, 1000)
# y = np.exp(-x/.3)+(np.random.rand(1000)-.5)*.2
#
# window_size = 8
# fig, ax = plt.subplots(1, figsize=(window_size, window_size/np.sqrt(2)))
# # plt.subplots_adjust(left=0.25)
#
# ax.set_title('Shadocks population when their are pumping')
# ax.set_ylabel('Shadoks population')
# ax.set_xlabel('Time (in shadock day)')
#
# ax.plot(x, y)
#
# func = lambda x, a, tau, shift: a*np.exp(-(x-shift)/tau)
#
# model = lmfit.Model(func)
# model.set_param_hint('a', value=0.5, min=0, max=2, vary=True)
# model.set_param_hint('tau', value=0.5, min=0, max=1, vary=True)
# model.set_param_hint('shift', value=0.5, min=0, max=1, vary=True)


class Shadocks:
    """ Class to analyse a lmfit model and get nice sliders to play with it

    @param (lmfit model) model: The model with param_hint defined with min max and value
    @param (array) x: The x data the model tries to fit
    @param (array) y: The y data the model tries to fit
    @param (plt figure) fig: The figure where the data is plotted (only needed if it is not the current figure)
    @param (plt axes) ax: The ax where the data is plotted (only needed if it is not the current axe)
    @param (bool) do_fit: Whether to try to fit first and use the fit result as default value
    @param (int) resolution: The number of point in drawn fitted curve
    @param (dict) plot_kw: A dict of parameters to give the plot function for the fitted line

    @return dict {'fig': fig, 'sliders': sliders} object that should be kept to prevent garbage collection
                    from cleaning useful variables
    """
    _do_not_update = False

    def __init__(self, model, x=None, y=None, fig=None, ax=None, do_fit=False, resolution=500, plot_kw={}, update_func=None):

        self._model = model
        self._fig = fig if fig is not None else plt.gcf()
        self._ax = ax if ax is not None else plt.gca()
        self._x_range = x[0], x[-1] if x is not None else self._ax.get_xbound()
        self._x_axis = np.linspace(*self._x_range, resolution)
        self._x = x
        self._y = y
        self._update_func = update_func

        if do_fit:
            if x is None or y is None:
                print('Can not fit without x and y !')
            else:
                result = self._fit()
                self._line = self._ax.plot(self._x_axis, self._model.func(self._x_axis, **result.best_values), **plot_kw)[0]
        else:
            self._line = self._ax.plot(self._x_axis, self._model.func(self._x_axis, **self._get_non_opt()), **plot_kw)[0]

        length = len(model.param_hints)
        # Create the window with toolbar disabled
        back_up = mpl.rcParams['toolbar']
        mpl.rcParams['toolbar'] = 'None'
        self._fig_dynamic = plt.figure(num='Shadock fitting', figsize=(8, .8 * length))
        mpl.rcParams['toolbar'] = back_up
        # Create slider
        self._sliders = {}

        for i, param in enumerate(self._model.param_hints):
            data = model.param_hints[param]
            ax_color = (231 / 255, 231 / 255, 231 / 255, 1)
            fill_color = (60 / 255, 144 / 255, 230 / 255, 1)
            box = [10 / 100, 10 / 100 + 80 / 100 * (i / length), 80 / 100, 10 / 100]
            axis = plt.axes(box, facecolor=ax_color)
            slider = Slider(axis, param, data['min'], data['max'], valinit=data['value'], color=fill_color)
            slider.label.set_size(15)
            self._sliders[param] = slider
            slider.on_changed(self._update)

        # Fit from here
        fit_ax_color = (231 / 255, 231 / 255, 231 / 255, 1)
        fit_ax = plt.axes([0.1, .9, 0.25, 0.1])
        self._fit_button = Button(fit_ax, 'Fit from here', color=fit_ax_color, hovercolor='0.8')
        self._fit_button.on_clicked(self._refit)

    # Helper method :
    def _get_non_opt(self):
        """ Helper method to get a dict from the model initial values """
        res = {}
        for p in self._model.param_hints:
            res[p] = self._model.param_hints[p]['value']
        return res

    def _update(self, _):
        """ Function called when a slider is changed to update the curve """
        current = {}
        for param in self._model.param_hints:
            current[param] = self._sliders[param].val
        self._line.set_ydata(self._model.func(self._x_axis, **current))
        if self._update_func is not None:
            self._update_func(current)
        self._fig.canvas.draw_idle()

    def _fit(self):
        result = self._model.fit(y, x=x)
        for key in result.best_values:
            self._model.param_hints[key]['value'] = result.best_values[key]
        return result

    def _refit(self, _):
        for key in self._sliders:
            self._model.param_hints[key]['value'] = self._sliders[key].val
        result = self._fit()
        for key in self._sliders:
            self._sliders[key].valinit = result.best_values[key]
            # sliders[key].val = result.best_values[key]
            self._sliders[key].reset()
        self._fig_dynamic.canvas.draw()

        self._update(0)

# a = Shadocks(model, x, y, fig)
# plt.show()
