import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def brownian_formula(point_at_t_minus_one, dt, gaussian_term):
    '''The most simple formula to implement Brownian motion.
    Parameters:
    Returns:
    Raises:
    '''
    if dt < 0.:
        raise ValueError('Given a negative time gap to dt.')

    point_at_t = point_at_t_minus_one + np.sqrt(dt) * gaussian_term
    return point_at_t


def gaussian_distribution(mu, sigma, seed=None):
    '''It computes a gaussian distribution's value.
    Parameters:
    Returns:
    '''
    if seed is None:
        output = np.random.normal(mu, sigma)
    else:
        random_number_generator = np.random.default_rng(seed)
        output = random_number_generator.normal(mu, sigma)

    return output


def exponential_distribution(beta, seed=None):
    '''It computes an exponential distribution's value.
    Parameters:
    Returns:
    '''
    if seed is None:
        output = np.random.exponential(beta)
    else:
        random_number_generator = np.random.default_rng(seed)
        output = random_number_generator.exponential(beta)

    return output


def init_time_state(initial_time=None, n_points=None):
    '''Something something.
    Parameters:
    Returns:
    '''
    if initial_time is None or n_points is None:
        return None
    time_state = np.empty(n_points)
    time_state[0] = initial_time
    return time_state


def time_state_updater(time_array, time_distribution, *time_parameters):
    '''It updates the time array by evaluating the distribution given
    as an input n_points - 1 times.
    Parameters:
    '''
    for i in range(1, len(time_array)):
        time_array[i] = time_array[i-1] + time_distribution(*time_parameters)


def init_space_state(initial_position=None, n_points=None):
    '''Something something.
    Parameters:
    Returns:
    '''
    if initial_position is None or n_points is None:
        return None
    space_state = np.empty(n_points)
    space_state[0] = initial_position
    return space_state


def space_state_updater(positions, interval_min, interval_max,
                        times, *gaussian_parameters):
    '''It updates the positions list by evaluating the brownian formula given
    before this function.
    Parameters:
    '''
    if times is None:
        return
    if positions[0] < interval_min or positions[0] > interval_max:
        raise ValueError('The starting point is out of bounds.')

    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        gaussian_term = gaussian_distribution(*gaussian_parameters)
        previous_point = positions[i-1]
        new_point = brownian_formula(previous_point, dt, gaussian_term)

        if new_point < interval_min:
            gap = abs(new_point - interval_min)
            reflected_position = interval_min + gap
            new_point = reflected_position

        if new_point > interval_max:
            gap = abs(interval_max - new_point)
            reflected_position = interval_max - gap
            new_point = reflected_position

        positions[i] = new_point


#def velocity_1d(space_state, time_state):
#    return (np.gradient(space_state) / np.gradient(time_state))


def initialize_states(n_points, t_0, x_0, y_0):
    times = init_time_state(t_0, n_points)
    x_coordinates = init_space_state(x_0, len(times))
    y_coordinates = init_space_state(y_0, len(times))

    return times, x_coordinates, y_coordinates


def update_states(times, x_coordinates, y_coordinates, x_min, x_max, 
                   y_min, y_max, mu, sigma, beta):
    time_state_updater(times, exponential_distribution, beta)
    space_state_updater(x_coordinates, x_min, x_max, times, mu, sigma)
    space_state_updater(y_coordinates, y_min, y_max, times, mu, sigma)


def display_plots(times, x_coordinates, y_coordinates,
                  x_min, x_max, y_min, y_max):
    fig1, ax1 = plt.subplots()
    ax1.plot(times, x_coordinates, label='x coordinate', color='orange')
    ax1.plot(times, y_coordinates, label='y coordinate', color='cyan')
    ax1.hlines([x_min, x_max], times[0], times[-1],
                label='x boundaries', color='orange', linestyle='dotted')
    ax1.hlines([y_min, y_max],  times[0], times[-1],
                label='y boundaries', color='cyan', linestyle='dotted')
    ax1.legend(loc='best')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Space')
    ax1.set_title('Time evolution of the coordinates')

    # plotting y over x, to see the trajectory on the 2d plane
    fig2, ax2 = plt.subplots()
    ax2.plot(x_coordinates, y_coordinates, label='trajectory', color='teal')
    ax2.plot(x_0, y_0, label='origin',
             color='red', marker='o', linestyle='None')
    boundaries = Rectangle((x_min, y_min), x_max - x_min,
                           y_max - y_min, facecolor='None', edgecolor='red',
                           linestyle='dashed', label='boundaries')
    ax2.add_patch(boundaries)
    ax2.legend(loc='best')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title('Trajectory on the xy plane')

    plt.show()


###############################################################################
if __name__ == "__main__":
    # parameters for state init
    n_points, t_0 = 1000, 0.0
    x_0, y_0 = 5.0, 2.0
    times, x_coordinates, y_coordinates = initialize_states(n_points,
                                                            t_0, x_0, y_0)
    # parameters for simulation and plotting
    x_min, x_max = 0.0, 20.0
    y_min, y_max = -12.0, 12.0
    mu, sigma, beta = 0.0, 1.0, 1.0
    update_states(times, x_coordinates, y_coordinates,
                  x_min, x_max, y_min, y_max, mu, sigma, beta)
    display_plots(times, x_coordinates, y_coordinates,
                  x_min, x_max, y_min, y_max)
