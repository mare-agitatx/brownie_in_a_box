import numpy as np
import scipy.stats as st
import json
import sys
import matplotlib.pyplot as plt


def load_data_json(filepath):
    '''
    Loads from file in json format the content as data in python format.
    Parameters:
        filepath: string, the path to the file to load.
    Returns:
        data: the loaded data from the input file.
    '''
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)

    return data


def parse_data_dict(data_dictionary):
    '''
    Parses some data from the dictionary given to be later analyzed.
    Parameters:
        data_dictionary: dictionary containing the data from file.
    Returns:
        parameters_dict: dictionary with the parameters of the simulation.
        bacteria_list: list of dictionaries, the entries represent bacteria.
    Raises:
        ValueError: if the given data_dictionary is empty.
    '''
    if data_dictionary == {}:
        raise ValueError('The given input dictionary is empty.')

    parameters_dict = data_dictionary['parameters']
    parameters_dict['max_time_reached'] = data_dictionary['max_time_reached']
    parameters_dict['result_flag'] = data_dictionary['result_flag']
    bacteria_list = data_dictionary['bacteria_list']

    return parameters_dict, bacteria_list


def space_value_at_some_time(bacterium_dict, time_of_interest):
    '''
    Finds the bacterium's position at some time and distinguishes the cases
    where said bacterium is alive or dead.
    Parameters:
        bacterium_dict: dictionary, represents the bacterium and all its
                        information.
        time_of_interest: the time where the position and status must be
                          evaluated.
    Returns:
        x, y, z: floats, coordinates of the bacterium at said time.
        event: string, status of the bacterium at said time.
    '''
    bacterium_states = bacterium_dict['states']
    first_state = bacterium_states[0]
    if first_state['time_of_observation'] > time_of_interest:
        return None, None, None, None

    for state_dict in bacterium_states:
        time = state_dict['time_of_observation']
        if time > time_of_interest:
            break
        x = state_dict['x']
        y = state_dict['y']
        z = state_dict['z']
        event = state_dict['event']

    return x, y, z, event


def space_distribution_at_some_time(bacteria_list, time_of_interest):
    '''
    Calculates two spatial distributions for the given bacteria input,
    distinguishing between dead bacteria and live bacteria.
    Parameters:
        bacteria_list: list of dictionaries, the entries represent bacteria.
        time_of_interest: the time at which the distributions are evaluated.
    Returns:
        output_dict: dictionary, has four entries that are lists with two
                     lists for x and y of live bacteria and two other lists
                     for coordinates of dead bacteria.
    '''
    live_xyz, dead_xyz = [], []
    for bacterium_dict in bacteria_list:
        x, y, z, event = space_value_at_some_time(bacterium_dict,
                                                 time_of_interest)
        if event is None:
            continue
        elif event == 'death':
            dead_xyz.append((x, y, z))
        else:
            live_xyz.append((x, y, z))

    return live_xyz, dead_xyz


def radial_distribution(list_xyz, x_0, y_0, z_0):
    '''
    Function to generate radial values from some origin, given a list of
    coordinates and the origin's coordinates.
    Parameters:
        list_xyz: list of tuples, representing coordinates of points.
        x_0, y_0, z_0: floats, coordinates of the origin.
    Returns:
        radii2: list of floats, the radii squared values calculated.
    '''
    radii2 = []
    for x, y, z in list_xyz:
        r2 = ((x - x_0)**2 + (y - y_0)**2 + (z - z_0)**2)
        radii2.append(r2)
    return radii2


def angular_distribution(list_xyz, x_0, y_0, z_0):
    '''
    Function to generate theta and phi values from some origin,
    given a list of coordinates and the origin's coordinates.
    Theta is the polar angle, phi the azimutal angle in a
    spherical reference system.
    Parameters:
        list_xyz: list of tuples, representing coordinates of points.
        x_0, y_0, z_0: floats, coordinates of the origin.
    Returns:
        thetas: list of floats, the polar values calculated.
        phis: list of floats, the azimutal values calculated.
    '''
    thetas, phis = [], []
    for x, y, z in list_xyz:
        is_the_origin = np.isclose(x, x_0)
        is_the_origin = is_the_origin and np.isclose(y, y_0)
        is_the_origin = is_the_origin and np.isclose(z, z_0)
        if is_the_origin is True:
            thetas.append(0.0)
            phis.append(0.0)
            continue

        r2 = (x - x_0)**2 + (y - y_0)**2 + (z - z_0)**2
        theta = np.arccos((z - z_0)/(np.sqrt(r2))).astype(float)
        thetas.append(theta)
        R2 = (x - x_0)**2 + (y - y_0)**2
        phi = np.arccos((x - x_0)/(np.sqrt(R2))).astype(float)
        phi *= np.sign(y).astype(float)
        phis.append(phi)
    return thetas, phis
