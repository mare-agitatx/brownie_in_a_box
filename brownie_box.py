import numpy as np
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


def init_time_state():
    return [1.0, 2.0, 3.0, 4.0, 5.0]


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


def init_space_state(initial_position=None):
    if initial_position is None:
        return None
    return [initial_position]


def run_simulation():
    mu, sigma = 0.0, 1.0
    x_0, y_0 = 0.0, 0.0

    times = init_time_state()
    x_coordinates, y_coordinates = init_space_state(x_0), init_space_state(y_0)

    space_state_updater(x_coordinates, times, mu, sigma)
    space_state_updater(y_coordinates, times, mu, sigma)

    print(x_coordinates)
    print(y_coordinates)


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
    assert expected == pytest.approx(observed)


def test_brownian_formula_2():
    dt = 0.0
    gaussian = 0.0
    previous_point = 0.0

    expected = 0.0
    observed = brownian_formula(previous_point, dt, gaussian)
    assert expected == pytest.approx(observed)


def test_gaussian_distribution():
    rng = np.random.default_rng(42)
    mu, sigma = 0.0, 1.0

    expected = rng.normal(mu, sigma)
    observed = gaussian_distribution(0.0, 1.0, 42)
    assert expected == pytest.approx(observed)


def test_init_time_state_1():
    expected = [1.0, 2.0, 3.0, 4.0, 5.0]
    observed = init_time_state()

    assert expected == pytest.approx(observed)


def test_init_time_state_2():
    observed = init_time_state()
    previous_time = observed[0]

    for time in observed[1:]:
        assert time >= previous_time
        previous_time = time


def test_space_state_updater_1():
    empty_time_state = []
    space_list = [0.1]
    initial_space_list = space_list
    space_state_updater(space_list, empty_time_state, 0.0, 1.0)

    assert initial_space_list == space_list


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

    assert expected == observed


def test_init_space_state():
    empty_space_state = init_space_state()
    assert empty_space_state is None
