import os

import numpy as np
import pandas as pd
import lmfit
import scipy
import ntpath

def get_all_data_files(search_str='', folder=None, file_format='', print_info=False):
    """Search in a folder and its subfolders all the files containing a given string in their name or filepath.

    @param string search_str (optional): the string to search in the file name and filepath
    @param string folder (optional): the folder to search into
    @param string file_format (optional): by default return all the file format, or else specify the format like 'dat' or '.dat'
    @param string print_info (optional): print the number of found files if true

    @return list : the list of all files found, with full filepath

    """

    if folder == None:
        search_dir = os.getcwd()
    else:
        search_dir = folder

    valid_files = []

    if len(file_format) == 0:
        for (dirpath, dirnames, files) in os.walk(search_dir):
            for name in files:
                if (search_str in name):
                    valid_files.append(os.path.join(dirpath, name))
            for dirname in dirnames:
                if search_str in dirname:
                    for name in os.listdir(os.path.join(dirpath, dirname)):
                        valid_files.append(os.path.join(dirpath, dirname, name))
    else:
        m = -len(file_format)
        for (dirpath, dirnames, files) in os.walk(search_dir):
            for name in files:
                if (search_str in name) & (name[m:] == file_format):
                    valid_files.append(os.path.join(dirpath, name))
            for dirname in dirnames:
                if search_str in dirname:
                    for name in os.listdir(os.path.join(dirpath, dirname)):
                        if (name[m:] == file_format):
                            valid_files.append(os.path.join(dirpath, dirname, name))
    if print_info:
        print(len(valid_files), 'file(s) found in ', search_dir)

    return valid_files


def get_all_data_folders(search_str='', folder=None, file_format='', print_info=False):
    """Search in a folder and its subfolders all the files containing a given string in their name or filepath.

    @param string search_str (optional): the string to search in the file name and filepath
    @param string folder (optional): the folder to search into
    @param string file_format (optional): by default return all the file format, or else specify the format like 'dat' or '.dat'
    @param string print_info (optional): print the number of found files if true

    @return list : the list of all folders in which at least a data file has been found.

    """

    if folder == None:
        search_dir = os.getcwd()
    else:
        search_dir = folder

    valid_folders = []

    if len(file_format) == 0:
        for (dirpath, dirnames, files) in os.walk(search_dir):
            for name in files:
                if (search_str in name) & (dirpath not in valid_folders):
                    valid_folders.append(dirpath)
            for dirname in dirnames:
                if search_str in dirname:
                    valid_folders.append(os.path.join(dirpath, dirname))
    else:
        m = -len(file_format)
        for (dirpath, dirnames, files) in os.walk(search_dir):
            for name in files:
                if (search_str in name) & (name[m:] == file_format) & (dirpath not in valid_folders):
                    valid_folders.append(dirpath)
            for dirname in dirnames:
                if (search_str in dirname):
                    for name in os.listdir(os.path.join(dirpath, dirname)):
                        if (name[m:] == file_format) & (os.path.join(dirpath, dirname) not in valid_folders):
                            valid_folders.append(os.path.join(dirpath, dirname))

    if print_info:
        print(len(valid_folders), 'folder(s) found in ', search_dir)

    return valid_folders


def read_data_file(filename):
    """ Read a Qudi data file and return the data parsed

    @param string filename: the file to read

    @return tuple dict(parameters), list(columns), numpy_2d_array(data)
    """
    try:
        file = open(filename, "r")
    except FileNotFoundError:
        raise FileNotFoundError('The file specified does not exist.')
    line = file.readline()
    last_line = ''
    parameters = {}
    while line[0] == '#':  # read line by line as long as the line start by a #
        line = line[1:]
        pair = line.split(':')
        if len(pair) == 2:  # if line is a key value pair
            key, value = pair
            if key != 'Parameters' and key != 'Data':  # Exclude theses lines
                try:
                    value = float(value)
                except ValueError:
                    value = str.strip(value)
                parameters[key] = value
        last_line = line
        line = file.readline()

    columns = last_line.split('\t')
    if columns[-1] == '\n':  # remove this small artefect if present
        columns = columns[:-1]
    if columns[0][-1] == '\n':  # remove this small artefect if present
        columns[0] = columns[0][:-1]
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim == 2:
        data = data.transpose()
    else:
        raise ValueError('The number of dimension of this file is neither 1 or 2.')
    return parameters, columns, data


def get_series_from_file(filename, additional_dictionary={}):
    """ Read a Qudi data file and return the data parsed as a pandas series

    @param string filename: the file to read
    @param dictionary additional_dictionary: keys and values to add manually to the series


    @return pandas.Series: Panda series containting the parameters and data columns and their values

    """
    parameters, columns, data = read_data_file(filename)
    dictionary = {}
    if len(columns) != len(data):
        columns = np.arange(len(data))
    for i, column in enumerate(
            columns):  # write data first as Python keep the insertion order of dictionaries since 3.7
        dictionary[column] = data[i]
    dictionary['_raw_data'] = data
    for key in parameters:
        dictionary[key] = parameters[key]
    for key in additional_dictionary:
        dictionary[key] = additional_dictionary[key]

    df = pd.Series(dictionary)
    return df


def get_dataframe_from_file(filename, additional_dictionary={}):
    """ Read a Qudi data file and return the data parsed as a pandas dataframe

    @param string filename: the file to read
    @param dictionary additional_dictionary: keys and values to add manually to the dataframe


    @return pandas.Series: Panda dataframe containting one row, the parameters and data columns and its values

    """
    df = get_series_from_file(filename, additional_dictionary=additional_dictionary).to_frame().transpose()
    return df


def get_dataframe_from_folders(folders, file_format='.dat', search_str='', additional_dictionary={},
                               additional_dictionaries=[], regexp=''):
    """ Read all the Qudi file in a folder or list of folders and return the data parsed as a pandas dataframe

    @param string or list(string) folders: folder or folders in wich to read all files
    @param string file_format: string to specify the file format wanted, eg '.dat', 'dat'
    @param string search_str: select the files that contains this string in their name
    @param dictionary additional_dictionary: keys and values to add manually to each dataframe
    @param list(dictionary) additional_dictionaries: keys and values to add manually to the each dataframe from the
                                                      respective folder
    @param re regexp: A regular expression that each path must match to imported

    If a key is overwritten, the order of importance is : additional_dictionaries > additional_dictionary > data file

    @return pandas.Series: Panda dataframe containting one row, the parameters and data columns and its values
    except if '' is specified for the file_format param, then return False

    """
    if len(file_format) == 0:
        print('Specify the format of the files and try again !')
        return False

    m = -len(file_format)
    frames = []
    if type(folders) == str:
        folders = [folders]
    for i, folder in enumerate(folders):
        if len(additional_dictionaries) != 0 and len(additional_dictionaries) != len(folders):
            raise ValueError('The additional_dictionaries list must have the same length as the folders list')

        if len(additional_dictionaries) != 0:
            dictionary = {**additional_dictionary.copy(), **additional_dictionaries[i]}
        else:
            dictionary = additional_dictionary
        for filename in os.listdir(folder):
            if (filename[m:] == file_format) & (search_str in filename):
                dictionary.update({'filepath': folder})
                dictionary.update({'filename': filename})
                frames.append(
                    get_dataframe_from_file('{}/{}'.format(folder, filename), additional_dictionary=dictionary))
    df = pd.concat(frames, sort=False).reset_index(drop=True)
    create_timestamp_from_filename(df)
    return df


def get_dataframe_from_files(files, additional_dictionary={}):
    """ Read all the Qudi files in a givne list and return the data parsed as a pandas dataframe

    @param string or list(string) files: files path
    @param dictionary additional_dictionary: keys and values to add manually to each rows

    If a key is overwritten, the order of importance is : additional_dictionary > data file

    @return pandas.Dataframe: Panda Dataframe containing all the data

    """
    frames = []
    for file in files:
        dictionary = additional_dictionary.copy()
        dictionary.update({'filepath': ntpath.dirname(file)})
        dictionary.update({'filename': ntpath.basename(file)})
        frames.append(
            get_dataframe_from_file(file, additional_dictionary=dictionary))
    df = pd.concat(frames, sort=False).reset_index(drop=True)
    create_timestamp_from_filename(df)
    return df


def copy_column_dataframe(df, src, dst):
    """ Function that copy a column to another if the destination is NaN and the source is not

    When using Qudi with scripts, the names of the columns may change from one file to another.

    Example :
        dat.copy_column_dataframe(df,'bin width (s)', 'binwidth')

    """
    if not hasattr(df, dst):
        df[dst]=None
    for i, row in df.iterrows():
        if (row[dst] is None or (type(row[dst])==float and pd.isnull(row[dst])) ) \
        and (row[src] is not None and not (type(row[src])==float and pd.isnull(row[src])) ):
            df.at[i, dst] = row[src]


def rebin(data, rebin_ratio, do_average=False):
    """ Rebin a 1D array the good old way.

    @param 1d numpy array data : The data to rebin
    @param int rebin_ratio: The number of old bin per new bin

    @return 1d numpy array : The array rebinned

    The last values may be dropped if the sizes do not match."""
    rebin_ratio = int(rebin_ratio)
    length = (len(data) // rebin_ratio) * rebin_ratio
    data = data[0:length]
    data = data.reshape(length//rebin_ratio, rebin_ratio)
    if do_average :
        data_rebinned = data.mean(axis=1)
    else :
        data_rebinned = data.sum(axis=1)
    return data_rebinned


def decimate(data, decimation_ratio):
    """ Decimate a 1D array . This means some value are dropped, not averaged

    @param 1d numpy array data : The data to decimated
    @param int decimation_ratio: The number of old value per new value

    @return 1d numpy array : The array decimated

    """
    decimation_ratio = int(decimation_ratio)
    length = (len(data) // decimation_ratio) * decimation_ratio
    data_decimated = data[:length:decimation_ratio]
    return data_decimated


def rebin_xy(x, y,  ratio=1, do_average=True):
    """ Helper method to decimate x and rebin y, with do_average True as default """
    return decimate(x, ratio), rebin(y, ratio, do_average)


def get_window(x, y, a, b):
    """ Very useful method to get just a window [a, b] of a signal (x,y) """
    mask_1 = a < x
    mask_2 = x < b
    mask = np.logical_and(mask_1, mask_2)
    x = x[mask]
    y = y[mask]
    return x, y


def match_columns(df_list, columns=[]):
    """ Function to extract from multiple dataframes the rows which have a match in all other dataframes

    @param: list(DataFrame) df_list: A list of DataFrames
    @param: list(DataFrame) df_list: A list colunmns names to match

    @return A dataframe with is the concatenation of all the matched rows

    To understand with the function does, let's take an example :
    You have a series of experiment with magnetic field and another without magnetic field. You cut them in two
    DataFrames, and then you want to get only the rows corresponding the experiments you have done in both conditions
    (not the ones you have only made with one or the other).
    Then you can do :
        data = match_columns([df_1, df_2], ['laser_power', 'mw_frequeny'])
    """
    # Checks for mistakes
    for column in columns:
        for df in df_list:
            if column not in df.keys():
                raise KeyError('Column {} is not defined is all of the dataframe'.format(column))
    # Create masks
    masks = []
    for i, df in enumerate(df_list):
        masks.append(np.zeros(len(df), dtype=bool))
        df_list[i] = df.reset_index()
    # Look for matches
    for i, row in df_list[0].iterrows():
        indexes = []
        for df in df_list[1:]:
            for column in columns:
                df = df[df[column] == row[column]]
            if len(df) > 0:
                indexes.append(list(df.index))

        if len(indexes) == len(df_list[1:]):
            masks[0][i] = True
            for j, index in enumerate(indexes):
                masks[j + 1][indexes] = True
    masked = [df[masks[i]] for i, df in enumerate(df_list)]
    return pd.concat(masked, sort=False).reset_index(drop=True)


def fit_data(data, function, params, x='x', y='y', verbose=False, do_not_fit=False, iterative=False,
             params_initial_values=[]):
    """ Use lmfit to automatically fit all the curves of a dataframe based on a param object

    @param Dataframe data: The dataframe containing the data
    @param Function function: The function used to fit. See example below
    @param lmfit.Parameters params: The lmfit parameters object defining the variables
    @param str x: the name of the x axis
    @param str y: the name of the y axis
    @param bool verbose: Set to true to have the fit report of lmfit of every fit printed
    @param bool do_not_fit: Set to true to evaluate the initial parameters (for debug)
    @param iterative do_not_fit: Set to true so that initial model for each row is the result of the previous fit
    @param params_initial_values: an array of (param_name, column_key) to use as initial parameter

    Note: The function here is minimized, so it has to do the subtraction explicitly.

    Example to fit un antibunching :

        def g2_function(params, t, data):
            x = np.abs(t - params['t0'])
            model = params['amplitude'] * np.exp(-x/params['tau_antibunching'])
            model += params['offset']
            return model - data

        params = lmfit.Parameters()
        params.add('t0', value=280e-9, vary=True)
        params.add('tau_antibunching', value=10e-9, vary=True)
        params.add('amplitude', value=-1, vary=True)
        params.add('offset', value=1, vary=False)

        fit_data(data, g2_function, params)

    """
    data['{}_fitted'.format(y)] = None
    data['fit_success'] = False
    data['fit_result'] = None
    data['fit_result'] = None
    for param in list(params):
        data[param] = None
        data['{}_err'.format(param)] = None
    last_result = None
    for i, row in data.iterrows():
        if do_not_fit:
            data.at[i, '{}_fitted'.format(y)] = function(params, row[x], 0)
        else:
            try:
                params_to_use = last_result if iterative and last_result is not None else params
                for param_name, key in params_initial_values:
                    params[param_name].value = row[key]
                result = lmfit.minimize(function, params_to_use, args=(row[x], row[y]))
                if verbose:
                    print(lmfit.fit_report(result))
                data.at[i, '{}_fitted'.format(y)] = row[y] + result.residual
                for param in list(params_to_use):
                    data.at[i, param] = result.params[param].value
                    data.at[i, '{}_err'.format(param)] = result.params[param].stderr
                data.at[i, 'fit_success'] = True
                data.at[i, 'fit_result'] = result
                last_result = result.params.copy()
            except ValueError:
                if verbose:
                    print('Fit "{}" failed.'.format(i))

    for param in list(params):
        data[param] = data[param].astype(float)
        data['{}_err'.format(param)] = data['{}_err'.format(param)].astype(float)

def clean_traces(data, x_additional_prefix='n', y_additional_prefix='k'):
    """ Clean a Qudi Pulsemeasurement raw trace and construct x and y axis

    The Qudi Pulsemeasurement logic saves metadata with unfriendly names
    """
    copy_column_dataframe(data, 'Signal(counts)', 'trace')
    copy_column_dataframe(data, 'Measurement sweeps', 'sweeps')
    copy_column_dataframe(data, 'bin width (s)', 'binwidth')
    copy_column_dataframe(data, 'record length (s)', 'windows_length')

    data['x'] = None
    data['x_n'] = None
    data['y'] = data['trace'] / (data['sweeps'] * data['binwidth'])
    data['y_k'] = data['y']/1e3
    for i, row in data.iterrows():
        data.at[i, 'x'] = np.arange(len(row['trace'])) * row['binwidth']
        data.at[i, 'x_n'] = data.at[i, 'x'] * 1e9


def m_to_eV(data, x='x', result_key='{}_eV'):
    """ Convert a column in meter to eV

    @param Dataframe data: The dataframe containing the data
    @param string x: The input column
    @param string result_key: The name of the output colmun

    Convert a column (by default 'x') to eV and save it as result_key (by default 'XXX_eV')

    You can specify the result by giving an key directly or using '{}' in the name so that it is replaced with the input
    key

    """
    c = 299792458
    eV = 1.602176634e-19
    h = 6.62607015e-34
    try:
        result_key = result_key.format(x)
    except:
        result_key = result_key
    data[result_key] = None
    for i, row in data.iterrows():
        data.at[i, result_key] = (h * c / row[x]) / eV


def eV_to_m(data, x='x', result_key='{}_m'):
    """ Convert a column in eV to meter

    @param Dataframe data: The dataframe containing the data
    @param string x: The input column
    @param string result_key: The name of the output colmun

    Convert a column (by default 'x') from eV to meter and save it as result_key (by default 'XXX_eV')

    You can specify the result by giving an key directly or using '{}' in the name so that it is replaced with the input
    key

    """
    c = 299792458
    eV = 1.602176634e-19
    h = 6.62607015e-34
    axis = data[x]
    try:
        result_key = result_key.format(x)
    except:
        result_key = result_key
    data[result_key] = None
    for i, row in data.iterrows():
        data.at[i, result_key] = h * c / (row[x] * eV)


def center_around_max(data, x='x', y='y', result_key='{}_centered'):
    """ Shift x,y data so that they are centered around the maximum


    @param Dataframe data: The dataframe containing the data
    @param string x: The x column
    @param string x: The y column
    @param string result_key: The name of the output colmun

    By default, the output column is 'XXX_centered'
    """
    try:
        result_key = result_key.format(x)
    except:
        result_key = result_key
    data[result_key] = None

    for i, row in data.iterrows():
        x_max = row[x][np.argmax(row[y])]
        data.at[i, result_key] = row[x] - x_max


def difference_curves(x1, y1, x2, y2, interpolation='linear'):
    """ Compute the subtraction of two curves that do not share the same x axis values (2-1)

    @param 1d numpy x1: The x axis of the first curve
    @param 1d numpy y1: The y axis of the first curve
    @param 1d numpy x2: The x axis of the second curve
    @param 1d numpy y2: The y axis of the second curve
    @param string interpolation: The type of interpolation. Fed to scipy.interpolate.interp1d


    The result only exist in the common interval between the two.
    To compute the result, the curve with denser axis is taken as reference and the less dense is itnerpolated.

    """
    x_min = max(x1.min(), x2.min())
    x_max = min(x1.max(), x2.max())
    x1, y1 = get_window(x1, y1, x_min, x_max)
    x2, y2 = get_window(x2, y2, x_min, x_max)
    if len(x1) == 0 or len(x2) == 0:
        print('Error: The two curves do not share any common interval of definition.')
        return
    first_is_denser = len(x1) > len(x2)
    x_dense, y_dense, x_dilute, y_dilute = (x1, y1, x2, y2) if first_is_denser else (x2, y2, x1, y1)
    f_dilute_interpolated = scipy.interpolate.interp1d(x_dilute, y_dilute, kind=interpolation, fill_value="extrapolate")
    result = y_dense - f_dilute_interpolated(x_dense)
    if first_is_denser:
        result = -result
    return x_dense, result


def create_window(data, a, b, x='x', y='y', window_x='window_x', window_y='window_y', shift_x=False):
    """ Helper function to create two x,y columns around a given window

    @param (float) a: Start of the window
    @param (float) b: End of the window
    @param (string) y: The source y column
    @param (string) x: The source x column
    @param (string) window_x: The destination x column
    @param (string) window_y: The destination y column
    @param (bool) shift_x: Whether to set the start of the new window at 0


    """
    data[window_x] = None
    data[window_y] = None
    for i, row in data.iterrows():
        data.at[i, window_x], data.at[i, window_y] = get_window(row[x], row[y], a, b)
        if shift_x:
            data.at[i, window_x] -= data.at[i, window_x].min()


def compute_shared_window(curves, interpolation='linear'):
    """
    @param curves: a list of curves (2d array (2, Ni) of different sizes)
    @return: a list of curves on the shared window with all the same x axis
    """
    mini_x = np.array([curve[0].min() for curve in curves])
    maxi_x = np.array([curve[0].max() for curve in curves])
    x1 = mini_x.max()
    x2 = maxi_x.min()
    new_data = [np.array(get_window(curve[0], curve[1], x1, x2)) for curve in curves]
    sizes = [len(curve[0]) for curve in new_data]
    if np.min(sizes) == 0:
        print(sizes)
        raise ValueError('No shared x axis.')
    denser_curve = np.argmax(sizes)
    interpolations = [scipy.interpolate.interp1d(curve[0], curve[1], kind=interpolation, fill_value="extrapolate") for
                     curve in new_data]
    x = new_data[denser_curve][0]
    final_data = [interpolation(x) for interpolation in interpolations]
    return x, final_data


def create_prefix_unit(data, key, prefix):
    """ Helper method to create or update a new columns based on the original times a unit prefix (m, k...) """
    # The sign is reversed so that a multiplication is used instead of a division
    # This help prevent round error
    prefix_db = { 'y': 1e24, 'z':1e21, 'a': 1e18, 'f': 1e15, 'p':1e12, 'n':1e9, 'u':1e6, 'm':1e3,
                 'k':1e-3, 'M':1e-6, 'G':1e-9, 'T':1e-12, 'P':1e-15, 'E':1e-18, 'Z':1e-21, 'Y':1e-24}
    if prefix not in prefix_db:
        print('Error: {} is not a valid prefix'.format(prefix))
        return
    data['{}_{}'.format(key, prefix)] = data[key]*prefix_db[prefix]

def create_timestamp_from_filename(data, filename='filename', timestamp='timestamp'):
    """ Create a timestamp column from the filename """
    data[timestamp] = pd.to_datetime(data[filename].str.slice(0, 16), format='%Y%m%d-%H%M-%S')

