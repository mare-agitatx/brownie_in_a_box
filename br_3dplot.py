import br_analysis as bra
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def cuboid_data(point, size=(1, 1, 1)):
    '''
    It generates an array of data that will later be employed to plot cuboids.
    Parameters:
        point: array-like of size 3, application point of the cuboid
               from where the sides are then spanned.
        size: array-like of size 3 if given, that influences the lengths of
              the sides of the cuboid; defaults at the tuple (1, 1, 1)
              which produces a unit-sized cube.
    Returns:
        X: np.array, data representing the cuboid.
    '''
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)

    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(point)

    return X


def plotCubes(positions, sizes=None, colors=None, **kwargs):
    '''
    Parameters:
        positions: iterable of array-like objects of size 3, application points
                   of the cuboid from where the sides are then spanned.
        sizes: iterable of array-like of size 3 if given, that influences the
               lengths of the sizes of the cuboids; defaults at None.
        colors: iterable if given, where elements should be colors interpreted
                in the style of matplotlib, which will color the faces of the
                cuboids; defaults at None.
        **kwargs: eventual keyword arguments that will be passed to
                  mpl_toolkits.mplot3d.art3d.Poly3DCollection, so that must
                  be compatible with it (check mpl_toolkits documentation
                  for more informations).
    Returns:
        output_cubes: mpl_toolkits.mplot3d.art3d.Poly3DCollection, collection
                      of 3d polygons representing the generated cuboids.
    '''
    is_colors_iterable = isinstance(colors, (list, np.ndarray))
    if is_colors_iterable is False:
        colors = ["C0"] * len(positions)

    is_sizes_iterable = isinstance(sizes, (list, np.ndarray))
    if is_sizes_iterable is False:
        sizes = [(1, 1, 1)] * len(positions)

    cuboids_arrays = []
    inputs_tuple = zip(positions, sizes)
    for point, input_size in inputs_tuple:
        new_cuboid = cuboid_data(point, size=input_size)
        cuboids_arrays.append(new_cuboid)

    output_cubes = Poly3DCollection(np.concatenate(cuboids_arrays),
                                    facecolors=np.repeat(colors, 6), **kwargs)

    return output_cubes


def make_cuboid(x_min, y_min, z_min, x_max, y_max, z_max):
    '''
    Routine to span a cuboid in a specific volume.
    Parameters:
        x_min, y_min, z_min: floats, interval minimums for the cuboid.
        x_max, y_max, z_max: floats, interval maximums for the cuboid.
    Returns:
        cuboid: mpl_toolkits.mplot3d.art3d.Poly3DCollection, collection of 3d
                polygons containing the single generated cuboid.
    '''
    positions = [(x_min, y_min, z_min)]
    sizes = [(x_max - x_min, y_max - y_min, z_max - z_min)]
    facecolors = ['None']
    cuboid = plotCubes(positions, sizes, colors=facecolors,
                       edgecolors='red', linestyle='dashed',
                       label='boundaries')

    return cuboid


def multiple_scatter_plots(coords_dict, times_of_interest,
                           x_min, y_min, z_min, x_max, y_max, z_max,
                           figure_title):
    '''
    Routine to represent the scatter plots of some spatial distributions at
    some specific times, producing eight subgraphs.
    Parameters:
        coords_list_dicts: dictionary, has four entries that are lists with two
                           lists for x and y of live bacteria and two other
                           lists for coordinates of dead bacteria.
        times_of_interest: list of floats, the times to evaluate.
        x_min, y_min, z_min: floats, interval minimums for the coordinates.
        x_max, y_max, z_max: floats, interval maximums for the coordinates.
        figure_title: string, title given to the figure.
    '''
    fig, axs = plt.subplots(2, 4, subplot_kw=dict(projection='3d'))
    for i, ax in enumerate(fig.axes):
        time_of_interest = times_of_interest[i]
        subtitle_string = f't = {time_of_interest:.2f}'
        ax.set_title(subtitle_string)
        
        live_xyz = coords_dict[str(i)]['live']
        dead_xyz = coords_dict[str(i)]['dead']
        live_x, live_y, live_z = [], [], []
        dead_x, dead_y, dead_z = [], [], []
        for some_tuple in live_xyz:
            x, y, z = some_tuple
            live_x.append(x)
            live_y.append(y)
            live_z.append(z)
        for some_tuple in dead_xyz:
            x, y, z = some_tuple
            dead_x.append(x)
            dead_y.append(y)
            dead_z.append(z)

        ax.scatter(dead_x, dead_y, dead_z,
                   marker='*', label='dead', color='orange')
        ax.scatter(live_x, live_y, live_z,
                   marker='*',  label='alive', color='cyan')
        boundaries = make_cuboid(x_min, y_min, z_min, x_max, y_max, z_max)
        ax.add_collection3d(boundaries)

    fig.axes[0].legend()
    fig.suptitle(figure_title)
    plt.show()


def multiple_graphs_at_fixed_times_routine(bacteria_list, t_0, t_f,
                                           x_min, y_min, z_min, x_max,
                                           y_max, z_max, figure_title):
    '''
    Routine to evaluate scatter plots of the distribution at eight
    times evenly distributed.
    Parameters:
        bacteria_list: list of dictionaries, representing the bacteria.
        t_0, t_f: floats, first instant and last instant of interest.
        x_min, y_min, z_min: floats, interval minimums for the coordinates.
        x_max, y_max, z_max: floats, interval maximums for the coordinates.
        figure_title: string, title given to the figure.
    '''
    times_of_interest = np.linspace(t_0, t_f, 8)
    coords_dict, index = {}, 0
    
    for time in times_of_interest:
        live_xyz, dead_xyz = bra.space_distribution_at_some_time(bacteria_list,
                                                                 time)
        coords_dict[str(index)] = {'live': live_xyz, 'dead': dead_xyz}
        index += 1
    
    multiple_scatter_plots(coords_dict, times_of_interest,
                           x_min, y_min, z_min, x_max, y_max, z_max,
                           figure_title)


################################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('No input given. Exiting...')
        print('Please input the data filename on the command line.')
        sys.exit()

    print('Loading the file...')
    filepath = sys.argv[1]
    analysis_data = bra.load_data_json(filepath)
    print('File loaded.')

    print('Parsing the data...')
    parameters_dict, bacteria_list = bra.parse_data_dict(analysis_data)
    x_min = parameters_dict['x_min']
    y_min = parameters_dict['y_min']
    z_min = parameters_dict['z_min']
    x_max = parameters_dict['x_max']
    y_max = parameters_dict['y_max']
    z_max = parameters_dict['z_max']
    t_0 = parameters_dict['t_0']
    if parameters_dict['result_flag'] != 'halted_simulation':
        t_f = parameters_dict['time_limit']
    else:
        t_f = parameters_dict['max_time_reached']

    print('Plotting the data...')
    figure_title = 'Plotting from ' + filepath
    multiple_graphs_at_fixed_times_routine(bacteria_list, t_0, t_f,
                                           x_min, y_min, z_min, x_max, y_max,
                                           z_max, figure_title)
