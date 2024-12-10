import br_analysis as bra
import numpy as np
import scipy.stats as st
import json
import sys
import matplotlib.pyplot as plt


def multiple_histogram_plots(coords_dict_lists, figure_title, x_0, y_0, z_0):
    '''
    Routine to generate the plots of the distributions as histograms and
    then some gaussians.
    Parameters:
        coords_dict_lists: dictionary, the entries containing the values for
                           the variables
        figure_title: string, title given to the figure.
        x_0, y_0 z_0: floats, coordinates of the origin.
    '''
    fig, axs = plt.subplots(2, 2)
    coords_list = list(coords_dict_lists)
    sigmas = []
    for i, ax in enumerate(fig.axes):
        # retrieving the variable data from
        # the input dictionary
        variable_name = coords_list[i]
        variable_values = coords_dict_lists[variable_name]

        # evaluating distributions from
        # empirical parameters as references
        t = np.linspace(min(variable_values), max(variable_values), 10000)
        if 'radial' not in variable_name:
            mu = np.mean(variable_values)
            sigma = np.std(variable_values, ddof=1)
            norm_pdf = st.norm.pdf(t, mu, sigma)
            ax.plot(t, norm_pdf)
            sigmas.append(sigma)
            var_title = variable_name + ' (gaussian)'
            param_str = f'mu = {mu:.2f}' + '\n' + f'sigma = {sigma:.2f}'
        else:
            dof = 3 # x, y and z, degrees of freedom
            loc = 0 # required from scipy.stats to plot
            scale = np.mean(sigmas)**2 # or chi2 will not be properly scaled
            chi2_pdf = st.chi2.pdf(t, dof, loc, scale)
            ax.plot(t, chi2_pdf)
            var_title = variable_name + ' (chi squared)'
            param_str = 'degrees of freedom = ' + str(dof) + '\n'
            param_str += f'scale factor = ({np.mean(sigmas):.2f})^2'
        
        ax.hist(variable_values, bins='auto', density=True)
        ax.text(0.95, 0.95, param_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right')
        ax.set_title(var_title)
        
    fig.suptitle(figure_title)
    plt.show()


def multiple_distributions_routine(bacteria_list, time_of_interest,
                                   figure_title, x_0, y_0, z_0):
    '''
    Routine to evaluate four distributions for x, y, z and the radii,
    by means of histograms, three gaussians and a chi squared bells.
    Parameters:
        bacteria_list: list of dictionaries, representing the bacteria.
        time_of_interest: float, the time at which the distributions are
                          evaluated.
        figure_title: string, title given to the figure.
        x_0, y_0 z_0: floats, coordinates of the origin.
    '''
    live_xyz, dead_xyz = bra.space_distribution_at_some_time(bacteria_list, t_f)

    xs, ys, zs = [], [], []
    for some_tuple in live_xyz:
        x, y, z = some_tuple
        xs.append(x)
        ys.append(y)
        zs.append(z)
    for some_tuple in dead_xyz:
        x, y, z = some_tuple
        xs.append(x)
        ys.append(y)
        zs.append(z)
    
    # the order in this dictionary must not be altered:
    # it is important that firstly are evaluated the
    # single coordinates and then the radii squared, 
    # since the function to plot the distributions
    # depends on that order to properly work
    coords_dict_lists = {}
    coords_dict_lists['x'] = xs
    coords_dict_lists['y'] = ys
    coords_dict_lists['z'] = zs
    
    xyzs = []
    for some_tuple in live_xyz:
        xyzs.append(some_tuple)
    for some_tuple in dead_xyz:
        xyzs.append(some_tuple)

    coords_dict_lists['radial^2'] = bra.radial_distribution(xyzs, x_0, y_0, z_0)
    multiple_histogram_plots(coords_dict_lists, figure_title, x_0, y_0, z_0)


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
    if parameters_dict['result_flag'] != 'halted_simulation':
        t_f = parameters_dict['time_limit']
    else:
        t_f = parameters_dict['max_time_reached']
    x_0 = parameters_dict['x_0']
    y_0 = parameters_dict['y_0']
    z_0 = parameters_dict['z_0']
    N = len(bacteria_list)
        
    figure_title = 'Histograms from ' + filepath
    multiple_distributions_routine(bacteria_list, t_f, figure_title,
                                   x_0, y_0, z_0)