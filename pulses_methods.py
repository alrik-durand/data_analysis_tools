""" Some of these methods are copied from Qudi pulsed module

This is Qudi free so it can be used stand alone

---

This file contain helper methods to extract and analyse pulses

"""

import numpy as np
from scipy import ndimage


def analyse_mean_norm(laser_data, bin_width, signal_start=0.0, signal_end=200e-9, norm_start=300e-9,
                      norm_end=500e-9):
    """

    @param laser_data:
    @param signal_start:
    @param signal_end:
    @param norm_start:
    @param norm_end:
    @return:
    """
    # Get number of lasers
    num_of_lasers = laser_data.shape[0]
    # Get counter bin width

    if not isinstance(bin_width, float):
        return np.zeros(num_of_lasers), np.zeros(num_of_lasers)

    # Convert the times in seconds to bins (i.e. array indices)
    signal_start_bin = int(round(signal_start / bin_width))
    signal_end_bin = int(round(signal_end / bin_width))
    norm_start_bin = int(round(norm_start / bin_width))
    norm_end_bin = int(round(norm_end / bin_width))

    # initialize data arrays for signal and measurement error
    signal_data = np.empty(num_of_lasers, dtype=float)
    error_data = np.empty(num_of_lasers, dtype=float)

    # loop over all laser pulses and analyze them
    for ii, laser_arr in enumerate(laser_data):
        # calculate the sum and mean of the data in the normalization window
        tmp_data = laser_arr[norm_start_bin:norm_end_bin]
        reference_sum = np.sum(tmp_data)
        reference_mean = (reference_sum / len(tmp_data)) if len(tmp_data) != 0 else 0.0

        # calculate the sum and mean of the data in the signal window
        tmp_data = laser_arr[signal_start_bin:signal_end_bin]
        signal_sum = np.sum(tmp_data)
        signal_mean = (signal_sum / len(tmp_data)) if len(tmp_data) != 0 else 0.0

        # Calculate normalized signal while avoiding division by zero
        if reference_mean > 0 and signal_mean >= 0:
            signal_data[ii] = signal_mean / reference_mean
        else:
            signal_data[ii] = 0.0

        # Calculate measurement error while avoiding division by zero
        if reference_sum > 0 and signal_sum > 0:
            # calculate with respect to gaussian error 'evolution'
            error_data[ii] = signal_data[ii] * np.sqrt(1 / signal_sum + 1 / reference_sum)
        else:
            error_data[ii] = 0.0

    return signal_data, error_data


def analyse_mean(laser_data, bin_width, signal_start=0.0, signal_end=200e-9, norm_start=0, norm_end=0):
    """

    @param laser_data:
    @param signal_start:
    @param signal_end:
    @return:
    """
    # Get number of lasers
    num_of_lasers = laser_data.shape[0]

    if not isinstance(bin_width, float):
        return np.zeros(num_of_lasers), np.zeros(num_of_lasers)

    # Convert the times in seconds to bins (i.e. array indices)
    signal_start_bin = int(round(signal_start / bin_width))
    signal_end_bin = int(round(signal_end / bin_width))

    # initialize data arrays for signal and measurement error
    signal_data = np.empty(num_of_lasers, dtype=float)
    error_data = np.empty(num_of_lasers, dtype=float)

    # loop over all laser pulses and analyze them
    for ii, laser_arr in enumerate(laser_data):
        # calculate the mean of the data in the signal window
        signal = laser_arr[signal_start_bin:signal_end_bin].mean()
        signal_sum = laser_arr[signal_start_bin:signal_end_bin].sum()
        signal_error = np.sqrt(signal_sum) / (signal_end_bin - signal_start_bin)

        # Avoid numpy C type variables overflow and NaN values
        if signal < 0 or signal != signal:
            signal_data[ii] = 0.0
            error_data[ii] = 0.0
        else:
            signal_data[ii] = signal
            error_data[ii] = signal_error

    return signal_data, error_data


def ungated_conv_deriv(count_data, conv_std_dev=20.0, number_of_lasers=1):
    """ Detects the laser pulses in the ungated timetrace data and extracts
        them.

    @param numpy.ndarray count_data: 1D array the raw timetrace data from an ungated fast counter
    @param dict measurement_settings: The measurement settings of the currently running measurement.
    @param float conv_std_dev: The standard deviation of the gaussian used for smoothing
    @param int number_of_lasers: The number of laser to look for

    @return 2D numpy.ndarray:   2D array, the extracted laser pulses of the timetrace.
                                dimensions: 0: laser number, 1: time bin

    Procedure:
        Edge Detection:
        ---------------

        The count_data array with the laser pulses is smoothed with a
        gaussian filter (convolution), which used a defined standard
        deviation of 10 entries (bins). Then the derivation of the convolved
        time trace is taken to obtain the maxima and minima, which
        corresponds to the rising and falling edge of the pulses.

        The convolution with a gaussian removes nasty peaks due to count
        fluctuation within a laser pulse and at the same time ensures a
        clear distinction of the maxima and minima in the derived convolved
        trace.

        The maxima and minima are not found sequentially, pulse by pulse,
        but are rather globally obtained. I.e. the convolved and derived
        array is searched iteratively for a maximum and a minimum, and after
        finding those the array entries within the 4 times
        self.conv_std_dev (2*self.conv_std_dev to the left and
        2*self.conv_std_dev) are set to zero.

        The crucial part is the knowledge of the number of laser pulses and
        the choice of the appropriate std_dev for the gauss filter.

        To ensure a good performance of the edge detection, you have to
        ensure a steep rising and falling edge of the laser pulse! Be also
        careful in choosing a large conv_std_dev value and using a small
        laser pulse (rule of thumb: conv_std_dev < laser_length/10).
    """
    # Create return dictionary
    return_dict = {'laser_counts_arr': np.empty(0, dtype='int64'),
                   'laser_indices_rising': np.empty(0, dtype='int64'),
                   'laser_indices_falling': np.empty(0, dtype='int64')}

    if not isinstance(number_of_lasers, int):
        return return_dict

    # apply gaussian filter to remove noise and compute the gradient of the timetrace sum
    try:
        conv = ndimage.filters.gaussian_filter1d(count_data.astype(float), conv_std_dev)
    except:
        conv = np.zeros(count_data.size)
    try:
        conv_deriv = np.gradient(conv)
    except:
        conv_deriv = np.zeros(conv.size)

    # if gaussian smoothing or derivative failed, the returned array only contains zeros.
    # Check for that and return also only zeros to indicate a failed pulse extraction.
    if len(conv_deriv.nonzero()[0]) == 0:
        return_dict['laser_counts_arr'] = np.zeros((number_of_lasers, 10), dtype='int64')
        return return_dict

    # use a reference for array, because the exact position of the peaks or dips
    # (i.e. maxima or minima, which are the inflection points in the pulse) are distorted by
    # a large conv_std_dev value.
    try:
        conv = ndimage.filters.gaussian_filter1d(count_data.astype(float), 10)
    except:
        conv = np.zeros(count_data.size)
    try:
        conv_deriv_ref = np.gradient(conv)
    except:
        conv_deriv_ref = np.zeros(conv.size)

    # initialize arrays to contain indices for all rising and falling
    # flanks, respectively
    rising_ind = np.empty(number_of_lasers, dtype='int64')
    falling_ind = np.empty(number_of_lasers, dtype='int64')

    # Find as many rising and falling flanks as there are laser pulses in
    # the trace:
    for i in range(number_of_lasers):
        # save the index of the absolute maximum of the derived time trace
        # as rising edge position
        rising_ind[i] = np.argmax(conv_deriv)

        # refine the rising edge detection, by using a small and fixed
        # conv_std_dev parameter to find the inflection point more precise
        start_ind = int(rising_ind[i] - conv_std_dev)
        if start_ind < 0:
            start_ind = 0

        stop_ind = int(rising_ind[i] + conv_std_dev)
        if stop_ind > len(conv_deriv):
            stop_ind = len(conv_deriv)

        if start_ind == stop_ind:
            stop_ind = start_ind + 1

        rising_ind[i] = start_ind + np.argmax(conv_deriv_ref[start_ind:stop_ind])

        # set this position and the surrounding of the saved edge to 0 to
        # avoid a second detection
        if rising_ind[i] < 2 * conv_std_dev:
            del_ind_start = 0
        else:
            del_ind_start = rising_ind[i] - int(2 * conv_std_dev)
        if (conv_deriv.size - rising_ind[i]) < 2 * conv_std_dev:
            del_ind_stop = conv_deriv.size - 1
        else:
            del_ind_stop = rising_ind[i] + int(2 * conv_std_dev)
            conv_deriv[del_ind_start:del_ind_stop] = 0

        # save the index of the absolute minimum of the derived time trace
        # as falling edge position
        falling_ind[i] = np.argmin(conv_deriv)

        # refine the falling edge detection, by using a small and fixed
        # conv_std_dev parameter to find the inflection point more precise
        start_ind = int(falling_ind[i] - conv_std_dev)
        if start_ind < 0:
            start_ind = 0

        stop_ind = int(falling_ind[i] + conv_std_dev)
        if stop_ind > len(conv_deriv):
            stop_ind = len(conv_deriv)

        if start_ind == stop_ind:
            stop_ind = start_ind + 1

        falling_ind[i] = start_ind + np.argmin(conv_deriv_ref[start_ind:stop_ind])

        # set this position and the sourrounding of the saved flank to 0 to
        #  avoid a second detection
        if falling_ind[i] < 2 * conv_std_dev:
            del_ind_start = 0
        else:
            del_ind_start = falling_ind[i] - int(2 * conv_std_dev)
        if (conv_deriv.size - falling_ind[i]) < 2 * conv_std_dev:
            del_ind_stop = conv_deriv.size - 1
        else:
            del_ind_stop = falling_ind[i] + int(2 * conv_std_dev)
        conv_deriv[del_ind_start:del_ind_stop] = 0

    # sort all indices of rising and falling flanks
    rising_ind.sort()
    falling_ind.sort()

    # find the maximum laser length to use as size for the laser array
    laser_length = np.max(falling_ind - rising_ind)

    # initialize the empty output array
    laser_arr = np.zeros((number_of_lasers, laser_length), dtype='int64')
    # slice the detected laser pulses of the timetrace and save them in the
    # output array according to the found rising edge
    for i in range(number_of_lasers):
        if rising_ind[i] + laser_length > count_data.size:
            lenarr = count_data[rising_ind[i]:].size
            laser_arr[i, 0:lenarr] = count_data[rising_ind[i]:]
        else:
            laser_arr[i] = count_data[rising_ind[i]:rising_ind[i] + laser_length]

    return_dict['laser_counts_arr'] = laser_arr.astype('int64')
    return_dict['laser_indices_rising'] = rising_ind
    return_dict['laser_indices_falling'] = falling_ind
    return return_dict


def ungated_threshold(count_data, count_threshold=10, min_laser_length=200e-9,
                      threshold_tolerance=20e-9, number_of_lasers=1, counter_bin_width=1e-9):
    """

    @param count_data:

    @return:
    """
    """
    Detects the laser pulses in the ungated timetrace data and extracts them.

    @param numpy.ndarray count_data: 1D array the raw timetrace data from an ungated fast counter
    @param measurement_settings: 
    @param fast_counter_settings: 
    @param count_threshold: 
    @param min_laser_length: 
    @param threshold_tolerance: 
    @param number_of_lasers:
    @param counter_bin_width:

    @return 2D numpy.ndarray:   2D array, the extracted laser pulses of the timetrace.
                                dimensions: 0: laser number, 1: time bin

    Procedure:
        Threshold detection:
        ---------------

        All count data from the time trace is compared to a threshold value.
        Values above the threshold are considered to belong to a laser pulse.
        If the length of a pulse would be below the minimum length the pulse is discarded.
        If a number of bins which are below the threshold is smaller than the number of bins making the
        threshold_tolerance then they are still considered to belong to a laser pulse.
    """
    return_dict = dict()

    if not isinstance(number_of_lasers, int):
        return_dict['laser_indices_rising'] = np.zeros(1, dtype='int64')
        return_dict['laser_indices_falling'] = np.zeros(1, dtype='int64')
        return_dict['laser_counts_arr'] = np.zeros((1, 3000), dtype='int64')
        return return_dict
    else:
        return_dict['laser_indices_rising'] = np.zeros(number_of_lasers, dtype='int64')
        return_dict['laser_indices_falling'] = np.zeros(number_of_lasers, dtype='int64')
        return_dict['laser_counts_arr'] = np.zeros((number_of_lasers, 3000), dtype='int64')

    # Convert length in seconds into length in time bins
    threshold_tolerance = round(threshold_tolerance / counter_bin_width)
    min_laser_length = round(min_laser_length / counter_bin_width)

    # get all bin indices with counts > threshold value
    bigger_indices = np.where(count_data >= count_threshold)[0]

    # get all indices with consecutive numbering (bin chains not interrupted by values < threshold
    index_list = np.split(bigger_indices,
                          np.where(np.diff(bigger_indices) >= threshold_tolerance)[0] + 1)
    for i, index_group in enumerate(index_list):
        if index_group.size > 0:
            start, end = index_list[i][0], index_list[i][-1]
            index_list[i] = np.arange(start, end + 1)
    consecutive_indices_unfiltered = index_list

    # sort out all groups shorter than minimum laser length
    consecutive_indices = [item for item in consecutive_indices_unfiltered if
                           len(item) > min_laser_length]

    # Check if the number of lasers matches the number of remaining index groups
    if number_of_lasers != len(consecutive_indices):
        return return_dict

    # determine max length of laser pulse and initialize laser array
    max_laser_length = max([index_array.size for index_array in consecutive_indices])
    return_dict['laser_counts_arr'] = np.zeros((number_of_lasers, max_laser_length), dtype='int64')

    # fill laser array with slices of raw data array. Also populate the rising/falling index arrays
    for i, index_group in enumerate(consecutive_indices):
        return_dict['laser_indices_rising'][i] = index_group[0]
        return_dict['laser_indices_falling'][i] = index_group[-1]
        return_dict['laser_counts_arr'][i, :index_group.size] = count_data[index_group]

    return return_dict
