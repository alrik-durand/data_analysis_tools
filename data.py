import os

import numpy as np
import pandas as pd


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
                    value = None
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
            row[dst] = row[src]


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
