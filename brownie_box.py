import numpy as np
import matplotlib.pyplot as plt


def brownian_formula(point_at_t_minus_one, dt, gaussian_term):
    '''The most simple formula to implement Brownian motion.
    It accepts a float time difference dt, a former position as a float at
    point_at_t_minus_one and a gaussian evaluation which here is just a float
    and must be computed outside of the function.
    It returns the updated point position as a float.
    '''
    if dt < 0.:
        raise ValueError('Given a negative time gap to dt.')

    point_at_t = point_at_t_minus_one + np.sqrt(dt) * gaussian_term
    return point_at_t


def gaussian_distribution(mu, sigma, seed=None):
    '''It computes a gaussian distribution's value.
    It accepts the average mu and the standard deviation sigma as floats, and
    it may also accept a seed as an integer.
    If the seed is given the function will use it to generate the result of
    the distribution in a predictable manner, otherwise it will be a
    (pseudo-)random result. 
    It returns an output value as a float.
    '''
    if seed is None:
        output = np.random.normal(mu, sigma)
    else:
        random_number_generator = np.random.default_rng(seed)
        output = random_number_generator.normal(mu, sigma)

    return output


def exponential_distribution(beta, seed=None):
    '''It computes an exponential distribution's value.
    It accepts the scale parameter beta as a float, and
    it may also accept a seed as an integer.
    If the seed is given the function will use it to generate the result of
    the distribution in a predictable manner, otherwise it will be a
    (pseudo-)random result. 
    It returns an output value as a float.
    '''
    if seed is None:
        output = np.random.exponential(beta)
    else:
        random_number_generator = np.random.default_rng(seed)
        output = random_number_generator.exponential(beta)

    return output


def init_time_state(initial_time=None):
    if initial_time is None:
        return None
    return [initial_time]


def time_state_updater(time_list, n_points,
    time_distribution, *time_parameters):
    '''It updates the time list by evaluating the distribution given
    as an input n_points - 1 times.
    It accept a time_list as a list of floats, the number of instants n_points
    that the list must contain as an integer, the time_distribution as a 
    function name that is passed as input and a tuple of parameters
    time_parameters that are passed to the distribution() function.
    The distribution is passed as a name so that the user may set any
    time distribution as they see fit, since more than a proper choice for
    the time statistics is possible.
    '''
    for i in range(1, n_points):
        new_time_value = time_list[-1] + time_distribution(*time_parameters)
        time_list.append(new_time_value)


def init_space_state(initial_position=None):
    if initial_position is None:
        return None
    return [initial_position]


def space_state_updater(positions, times, *gaussian_parameters):
    '''It updates the positions list by evaluating the brownian formula given
    before this function.
    It accepts positions and times both as lists of floats, and 
    then gaussian parameters as a tuple, so that the user may decide
    if to give mu, sigma as floats and the rng seed as integer or
    just mu and sigma without giving the seed.
    '''
    if len(times) == 0:
        return

    previous_time = times[0]
    for time in times[1:]:
        dt = time - previous_time
        gaussian_term = gaussian_distribution(*gaussian_parameters)
        previous_point = positions[-1]
        new_point = brownian_formula(previous_point, dt, gaussian_term)
        positions.append(new_point)
        previous_time = time


def run_simulation():
    # initializing the parameters
    mu, sigma, beta = 0.0, 1.0, 1.0
    x_0, y_0, t_0 = 0.0, 0.0, 0.0
    n_points = 1000

    # initializing the states
    times = init_time_state(t_0)
    x_coordinates = init_space_state(x_0)
    y_coordinates = init_space_state(y_0)

    # updating the states
    time_state_updater(times, n_points, exponential_distribution, beta)
    space_state_updater(x_coordinates, times, mu, sigma)
    space_state_updater(y_coordinates, times, mu, sigma)

    # plotting x and y over t, to see their time evolution
    fig1, ax1 = plt.subplots()
    ax1.plot(times, x_coordinates, label='x coordinate', color='orange')
    ax1.plot(times, y_coordinates, label='y coordinate', color='cyan')
    ax1.legend(loc='best')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Space')
    ax1.set_title('Time evolution of the coordinates')

    # plotting y over x, to see the trajectory on the 2d plane
    fig2, ax2 = plt.subplots()
    ax2.plot(x_coordinates, y_coordinates, label='trajectory', color='teal')
    ax2.plot(x_0, y_0, label='origin',
                color='red', marker='o', linestyle='None')
    ax2.legend(loc='best')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title('Trajectory on the xy plane')

    plt.show()


###############################################################################
if __name__ == "__main__":
    run_simulation()
