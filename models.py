import numpy as np
import lmfit



def get_g2_model():

    def g2_function(params, t, data):
        x = np.abs(t - params['t0'])
        model = 0
        model += params['a'] * np.exp(-x/params['tau_antibunching'])
        model += params['b'] * np.exp(-x/params['tau_bunching'])
        model += params['c']
        return model - data

    params = lmfit.Parameters()
    params.add('t0', vary=True)
    params.add('tau_antibunching', value=10e-09, vary=True)
    params.add('tau_bunching', value=30e-9)
    params.add('a', value=-1, vary=True)
    params.add('b', value=1, vary=True)
    params.add('c', value=1, vary=False)
    return g2_function, params



