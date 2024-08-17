import numpy as np
import matplotlib.pyplot as plt
import pytest


def brownian_formula(point_at_t_minus_one, dt, gaussian_term):
    '''
    The most simple formula to implement Brownian motion.
    It accepts a float time difference dt, a former position as a float at
    point_at_t_minus_one and a gaussian evaluation which here is just a float
    and must be computed outside of the function.
    It returns the updated point position as a float.
    '''
    point_at_t = point_at_t_minus_one + np.sqrt(dt) * gaussian_term
    return point_at_t


def gaussian_distribution(mu, sigma, seed=None):
    '''
    It computes a gaussian distribution's value.
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
    '''
    It computes an exponential distribution's value.
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
    if len(time_list) == 0:
        return
    for i in range(1, n_points):
        new_time_value = time_list[-1] + time_distribution(*time_parameters)
        time_list.append(new_time_value)


def init_space_state(initial_position=None):
    if initial_position is None:
        return None
    return [initial_position]


def space_state_updater(positions, times, *gaussian_parameters):
    '''
    It updates the positions list by evaluating the brownian formula given
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
    x_coordinates, y_coordinates = init_space_state(x_0), init_space_state(y_0)

    # updating the states
    time_state_updater(times, n_points, exponential_distribution, beta)
    space_state_updater(x_coordinates, times, mu, sigma)
    space_state_updater(y_coordinates, times, mu, sigma)

    # plotting x and y over t, to see their time evolution
    plt.figure(1)
    plt.plot(times, x_coordinates, label='x coordinate', color='orange')
    plt.plot(times, y_coordinates, label='y coordinate', color='cyan')
    plt.legend(loc='best')
    plt.xlabel('Time (s)')
    plt.ylabel('Space (m)')

    # plotting y over x, to see the trajectory on the 2d plane
    plt.figure(2)
    plt.plot(x_coordinates, y_coordinates, label='trajectory', color='teal')
    plt.plot(x_0, y_0, label='origin',
    color='red', marker='o', linestyle='None')
    plt.legend(loc='best')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.show()


################################################################################
if __name__ == "__main__":
    run_simulation()


################################################################################
def test_brownian_formula_1():
    dt = 2.0
    gaussian = 0.5
    previous_point = 3.0

    expected = 3.0 + 0.5 * 1.41421356
    observed = brownian_formula(previous_point, dt, gaussian)
    assert observed == pytest.approx(expected)


def test_brownian_formula_2():
    dt = 0.0
    gaussian = 0.0
    previous_point = 0.0

    expected = 0.0
    observed = brownian_formula(previous_point, dt, gaussian)
    assert observed == pytest.approx(expected)


def test_gaussian_distribution():
    rng = np.random.default_rng(42)
    mu, sigma = 0.0, 1.0

    expected = rng.normal(mu, sigma)
    observed = gaussian_distribution(0.0, 1.0, 42)
    assert observed == pytest.approx(expected)


def test_exponential_distribution():
    rng = np.random.default_rng(69)
    beta = 1.0

    expected = rng.exponential(beta)
    observed = exponential_distribution(1.0, 69)
    assert observed == pytest.approx(expected)


def test_init_time_state_1():
    empty_time_state = init_space_state()
    assert empty_time_state is None


def test_init_time_state_2():
    time_state = init_time_state(1.0)
    assert len(time_state) == 1


def test_init_time_state_3():
    time_state = init_time_state(3.0)
    assert time_state == [3.0]


def test_time_state_updater_1():
    observed_times = [0.0]
    distribution = exponential_distribution
    parameters = [1.0]
    time_state_updater(observed_times, 10, distribution, *parameters)
    previous_time = observed_times[0]

    for time in observed_times[1:]:
        assert time >= previous_time
        previous_time = time


def test_time_state_updater_2():
    observed_times = [0.0]
    n_points = 10
    time_state_updater(observed_times, n_points, exponential_distribution, 1.0)

    assert len(observed_times) == n_points


def test_init_space_state_1():
    empty_space_state = init_space_state()
    assert empty_space_state is None


def test_init_space_state_2():
    space_state = init_space_state(1.0)
    assert len(space_state) == 1


def test_init_space_state_3():
    space_state = init_space_state(3.0)
    assert space_state == [3.0]


def test_space_state_updater_1():
    empty_time_state = []
    space_list = [0.1]
    initial_space_list = space_list
    space_state_updater(space_list, empty_time_state, 0.0, 1.0)

    assert space_list == initial_space_list


def test_space_state_updater_2():
    time_state = [1.0, 2.0, 3.0, 4.0, 5.0]
    gaussian_parameters = (0.0, 1.0, 50)

    observed = [0.1]
    space_state_updater(observed, time_state, *gaussian_parameters)

    expected = [0.1]
    for time in time_state[1:]:
        gaussian = gaussian_distribution(*gaussian_parameters)
        new_point = brownian_formula(expected[-1], 1.0, gaussian)
        expected.append(new_point)

    assert observed == expected


def test_space_state_updater_3():
    time_state = [1.0, 2.0, 3.0, 4.0, 5.0]
    gaussian_parameters = (0.0, 1.0, 50)

    space_state = [0.1]
    space_state_updater(space_state, time_state, *gaussian_parameters)

    assert len(space_state) == len(time_state)
