import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


FILEPATH = 'results/2024_09_12_11_50_00.json'
BACTERIUM = 5


def load_data_json(filepath):
    '''Blah blah.
    Parameters:
        filepath: string, the path to the file to load.
    Returns:
        data: the loaded data from the input file.
    Raises:

    '''
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    return data


def parse_data_dictionary(data_dictionary, bact_id_number):
    '''Blah blah.
    Parameters:
        data_dictionary: dictionary containing the data from file.
    Returns:

    Raises:

    '''
    if data_dictionary == {}:
        raise ValueError('The given input dictionary is empty.')

    parameters_dict = data_dictionary['parameters']
    bacteria_list = data_dictionary['bacteria']

    t, x, y = [], [], []
    bacterium = bacteria_list[bact_id_number]
    for state_dict in bacterium:
        t.append(state_dict['time_of_observation'])
        x.append(state_dict['x'])
        y.append(state_dict['y'])

    bacterium_states_out = {'times': t, 'x_coords': x, 'y_coords': y}
    return parameters_dict, bacterium_states_out


def plots(times, x_coordinates, y_coordinates, x_min, x_max, y_min, y_max):
    '''Plots the given lists as time evolutions and a trajectory.
    Parameters:
        times: list of floats, containing the times of observation
        x_coordinates, y_coordinates: lists of floats, containing
                                      the various positions
        x_min, x_max: floats, interval limits for the x coordinate
        y_min, y_max: floats, interval limits for the y coordinate
    '''

    fig1, ax1 = plt.subplots()
    ax1.set_title('Time evolution of the coordinates')
    ax1.plot(times, x_coordinates, label='x coordinate', color='orange')
    ax1.plot(times, y_coordinates, label='y coordinate', color='cyan')
    ax1.hlines([x_min, x_max], times[0], times[-1],
                label='x boundaries', color='orange', linestyle='dotted')
    ax1.hlines([y_min, y_max],  times[0], times[-1],
                label='y boundaries', color='cyan', linestyle='dotted')
    ax1.legend(loc='best')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Space')

    fig2, ax2 = plt.subplots()
    ax2.set_title('Trajectory on the xy plane')
    ax2.plot(x_coordinates, y_coordinates, label='trajectory', color='teal')
    x_0, y_0 = x_coordinates[0], y_coordinates[0]
    ax2.plot(x_0, y_0, label='origin',
             color='red', marker='o', linestyle='None')
    boundaries = Rectangle((x_min, y_min), x_max - x_min,
                            y_max - y_min, facecolor='None', edgecolor='red',
                            linestyle='dashed', label='boundaries')
    ax2.add_patch(boundaries)
    ax2.legend(loc='best')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    plt.show()


###############################################################################
if __name__ == "__main__":
    print('Loading the file...')
    analysis_data = load_data_json(FILEPATH)
    print('File loaded.')
    print('Parsing the data...')
    parameters_dict, bacterium_states = parse_data_dictionary(analysis_data,
                                                              BACTERIUM)
    x_min = parameters_dict['x_min']
    x_max = parameters_dict['x_max']
    y_min = parameters_dict['y_min']
    y_max = parameters_dict['y_max']
    times = bacterium_states['times']
    x_coords = bacterium_states['x_coords']
    y_coords = bacterium_states['y_coords']
    print('Plotting the data...')
    plots(times, x_coords, y_coords, x_min, x_max, y_min, y_max)
