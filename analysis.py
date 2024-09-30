import numpy as np
import json
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
    Loads some data from the dictionary given to be later analyzed.
    Parameters:
        data_dictionary: dictionary containing the data from file.
    Returns:
        parameters_dict: dictionary with the parameters of the simulation.
        bacterium_states_out:
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
    Finds the bacterium position at some time and distinguishes the cases
    where said bacterium is alive or dead.
    Parameters:
        bacterium_dict: dictionary, represents the bacterium and all its
                        information.
        time_of_interest: the time where the position and status must be
                          evaluated.
    Returns:
        x, y: floats, coordinates of the bacterium at said time.
        flag: string, status of the bacterium at said time.
    '''
    bacterium_states = bacterium_dict['states']
    first_state = bacterium_states[0]
    if first_state['time_of_observation'] > time_of_interest:
        return None, None, None

    for state_dict in bacterium_states:
        time = state_dict['time_of_observation']
        if time > time_of_interest:
            break
        x = state_dict['x']
        y = state_dict['y']
        event = state_dict['event']

    if event == 'death':
        flag = 'dead'
    else:
        flag = 'alive'
    return x, y, flag


def space_distribution_at_some_time(bacteria_list, time_of_interest):
    '''
    Calculates two spatial distributions for the given bacteria input,
    separating between dead bacteria and live bacteria.
    Parameters:
        bacteria_list: list of dictionaries, the entries represent bacteria.
        time_of_interest: the time at which the distributions are evaluated.
    Returns:
        output_dict: dictionary, has four entries that are lists with two
                     lists for x and y of live bacteria and two other lists
                     for coordinates of dead bacteria.
    Raises:
        ValueError: if an unexpected flag is encountered while assigning
                    the lists.
    '''
    live_x, live_y = [], []
    dead_x, dead_y = [], []
    for bacterium_dict in bacteria_list:
        x, y, flag = space_value_at_some_time(bacterium_dict, time_of_interest)
        if flag == 'alive':
            live_x.append(x)
            live_y.append(y)
        elif flag == 'dead':
            dead_x.append(x)
            dead_y.append(y)
        elif flag is None:
            continue
        else:
            raise ValueError('Unrecognized flag.')

    output_dict = {'live_x': live_x, 'live_y': live_y,
                   'dead_x': dead_x, 'dead_y': dead_y}
    return output_dict


def multiple_scatter_plots(coords_list_dicts, times_of_interest,
                           x_min, x_max, y_min, y_max, figure_title):
    '''
    Routine to represent the scatter plots of some spatial distributions at
    some specific times, producing eight subgraphs.
    Parameters:
        coords_list_dicts: dictionary, has four entries that are lists with two
                           lists for x and y of live bacteria and two other
                           lists for coordinates of dead bacteria.
        times_of_interest: list of floats, the times to evaluate.
        x_min, x_max: floats, interval limits for the x coordinate.
        y_min, y_max: floats, interval limits for the y coordinate.
        figure_title: string, title given to the figure.
    '''
    fig, axs = plt.subplots(2, 4)
    for i, ax in enumerate(fig.axes):
        time_of_interest = times_of_interest[i]
        subtitle_string = f't = {time_of_interest:.2f}'
        ax.set_title(subtitle_string)
        live_x = coords_list_dicts[i]['live_x']
        live_y = coords_list_dicts[i]['live_y']
        dead_x = coords_list_dicts[i]['dead_x']
        dead_y = coords_list_dicts[i]['dead_y']
        ax.scatter(dead_x, dead_y, marker='*',  label='dead', color='orange')
        ax.scatter(live_x, live_y, marker='*',  label='alive', color='cyan')
        boundaries = Rectangle((x_min, y_min), x_max - x_min,
                            y_max - y_min, facecolor='None', edgecolor='red',
                            linestyle='dashed', label='boundaries')
        ax.add_patch(boundaries)
    fig.axes[0].legend()
    fig.suptitle(figure_title)
    plt.show()


def multiple_graphs_at_fixed_times_routine(bacteria_list, t_0, t_f,
                                           x_min, x_max, y_min, y_max,
                                           figure_title):
    '''
    Routine to evaluate scatter plots of the distribution at eight
    times evenly distributed.
    Parameters:
        bacteria_list: list of dictionaries, representing the bacteria.
        t_0, t_f: floats, first instant and last instant of interest.
        x_min, x_max: floats, interval limits for the x coordinate.
        y_min, y_max: floats, interval limits for the y coordinate.
        figure_title: string, title given to the figure.
    '''
    times_of_interest = np.linspace(t_0, t_f, 8)
    coords_list_dicts = []
    for time in times_of_interest:
        coords_dict = space_distribution_at_some_time(bacteria_list, time)
        coords_list_dicts.append(coords_dict)
    multiple_scatter_plots(coords_list_dicts, times_of_interest,
                           x_min, x_max, y_min, y_max, figure_title)


###############################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('No input given. Exiting...')
        print('Please input the data filename on the command line.')
        sys.exit()

    print('Loading the file...')
    filepath = sys.argv[1]
    analysis_data = load_data_json(filepath)
    print('File loaded.')

    print('Parsing the data...')
    parameters_dict, bacteria_list = parse_data_dict(analysis_data)
    x_min = parameters_dict['x_min']
    x_max = parameters_dict['x_max']
    y_min = parameters_dict['y_min']
    y_max = parameters_dict['y_max']
    t_0 = parameters_dict['t_0']
    if parameters_dict['result_flag'] != 'halted_simulation':
        t_f = parameters_dict['time_limit']
    else:
        t_f = parameters_dict['max_time_reached']

    print('Plotting the data...')
    figure_title = 'Plotting from ' + filepath
    multiple_graphs_at_fixed_times_routine(bacteria_list, t_0, t_f,
                                           x_min, x_max, y_min, y_max,
                                           figure_title)
