import numpy as np


def impulse_response_aom(dt=1e-9, window=1e-7, rise_time=20e-9):
    """ Impulse response of the AOM assuming logistic function """
    delay = 0
    k = 1/(rise_time/8)  # the slope of the tangent is k/4 for the logistic function
    nb_bin = window/dt
    f = lambda x: 1/(1+np.exp(-k*(x-delay)))
    time = (np.arange(int(nb_bin))-nb_bin/2)*dt
    response = f(time)*(1-f(time))
    response /= np.sum(response)
    return response


def convert_sequence_to_laser(sequence, dt=1e-9, aom=False):
    """ This function transform a sequence object to a 1D object of laser intensity with respect to time

    The sequence looks like : sequence = [
                                {'length': 1.5e-6, 'laser': 0},
                                {'length': 6e-6, 'laser': power},
                                {'length': 3e-6, 'laser': 0} ]

    The function return a 1D array of laser_power versus time with a time step of dt

    """
    total_time = np.sum(seq['length'] for seq in sequence)
    laser_power_theory = np.zeros(int(total_time / dt))
    i = 0
    for seq in sequence:
        nb_bin = int(seq['length'] / dt)
        laser_power_theory[i:i+nb_bin] = np.ones(nb_bin)*seq['laser']
        i += nb_bin
    if aom:
        laser_power_applied = np.convolve(laser_power_theory, impulse_response_aom(dt), mode='same')
    else:
        laser_power_applied = laser_power_theory
    return laser_power_applied


def convert_sequence_to_mw(sequence, dt=1e-9):
    """ This function compute the mw pulses from the sequence

    #TODO: do this properly ?

    The sequence looks like : sequence = [
                                {'length': 1.5e-6, 'laser': 0},
                                {'length': 1.5e-6, 'laser': 0, 'mw':np.pi},
                                {'length': 6e-6, 'laser': power},
                                {'length': 3e-6, 'laser': 0} ]

    The return is an dictionary { 'bin_number_1' : np.pi, ...
    """
    mw = {}
    i = 0
    for seq in sequence:
        i += int(seq['length'] / dt)
        if 'mw' in seq and seq['mw'] != 0:
            mw[i] = seq['mw']
    return mw