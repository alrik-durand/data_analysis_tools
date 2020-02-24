import numpy as np
import lmfit



def get_g2_model(bunching=False):
    """ Define a model to fit a g2

    @param (bool) bunching : Whether to put punching in the model or not """

    def function(params, t, data):
        x = np.abs(t - params['t0'])
        model = 0
        model += params['amplitude_antibunching'] * np.exp(-x/params['tau_antibunching'])
        model += params['amplitude_bunching'] * np.exp(-x/params['tau_bunching'])
        model += params['offset']
        return model - data

    params = lmfit.Parameters()
    params.add('t0', vary=True)
    params.add('tau_antibunching', value=10e-09, vary=True)
    params.add('tau_bunching', value=30e-9)
    params.add('amplitude_antibunching', value=-1, vary=True)
    if bunching:
        params.add('amplitude_bunching', value=1, vary=True)
    else:
        params.add('amplitude_bunching', value=0, vary=False)
    params.add('offset', value=1, vary=False)
    return function, params


def get_lifetime_model(number_decay=1, rise_time=False, order_lifetime=False):
    """ Define a model to fit lifetimes

    @param (int) number_decay: Number of decays, default is 1 for mono exponential
    @param (bool) rise_time: Whether to add a rise time to the model or not
    @param (bool) order_lifetime : If true, force the lifetime to be in ascending order

    @return function, params, report_function

    sum over i (1 to number_decay) of amplitude_decay_i * exp(-t/lifetime_decay_i)

    """

    def function(params, t, data):
        x = t - params['t0']
        growth = np.heaviside(x, 1)
        model = params['offset']
        for i in range(number_decay):
            n = i+1 # start at 1
            model += growth * params['amplitude_decay_{}'.format(n)]*np.exp(-x / params['lifetime_decay_{}'.format(n)])
        return model - data

    params = lmfit.Parameters()
    params.add('t0', value=0e-9)
    params.add('offset', value=0, vary=True)
    for i in range(number_decay):
        n = i + 1  # start at 1
        params.add('amplitude_decay_{}'.format(n), value=1)
        params.add('lifetime_decay_{}'.format(n), value=10e-9)
        if order_lifetime and i > 0:
            expr = 'lifetime_decay_{}-lifetime_decay_{}'.format(i, i+1)
            params.add('lifetime_delta_{}'.format(i), value=0, max=0, expr=expr)

    def report_function(data, row_label=None):
        """ Return markdown text to print the fit result more easily

        Example :
            from IPython.display import Markdown as md
            function, params, report_function = models.get_lifetime_model()
            dat.fit_data(data, function, params)
            md(report_function(data, '{poi}'))

        """
        output_total = ''
        for i, row in data.iterrows():
            label = row_label.format(**row.to_dict()) if row_label is not None else i
            output = '**{}**<br>'.format(label)
            if not row.amplitude_decay_1 > 0 or row.amplitude_decay_1_err > row.amplitude_decay_1:
                output += '*Failed* <br>'
            else:
                for j in range(number_decay):
                    output_line = '\u03C4 = **{:.2f}** $\pm$ {:.1f} ns <br>'
                    output_line = output_line.format(row['lifetime_decay_{}'.format(j + 1)] * 1e9,
                                                     row['lifetime_decay_{}_err'.format(j + 1)] * 1e9
                                                     )
                    output += output_line
            output_total += output + '<br>'
        return output_total


    return function, params, report_function


def get_dipole_model():
    """ Define a model to fit dipoles

    @return function, params
    """

    def function(params, theta, data):
        model = params['bg'] + params['amplitude'] * np.sin(theta-params['phi'])**2
        return model - data

    params = lmfit.Parameters()
    params.add('bg', value=1e3, vary=True)
    params.add('amplitude', value=20e3, min=0, vary=True)
    params.add('phi', value=np.pi/4, vary=True)

    return function, params
