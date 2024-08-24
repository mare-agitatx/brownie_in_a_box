from brownie_box import *
import pytest


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


def test_brownian_formula_3():
    with pytest.raises(ValueError):
        dt = -1.0
        brownian_formula(1.0, dt, 1.0)


def test_gaussian_distribution_1():
    rng = np.random.default_rng(42)
    mu, sigma = 0.0, 1.0

    expected = rng.normal(mu, sigma)
    observed = gaussian_distribution(0.0, 1.0, 42)
    assert observed == pytest.approx(expected)


def test_gaussian_distribution_2():
    mu = 2.0
    observed = gaussian_distribution(mu, 0.0)
    assert observed == pytest.approx(mu)


def test_exponential_distribution_1():
    rng = np.random.default_rng(69)
    beta = 1.0

    expected = rng.exponential(beta)
    observed = exponential_distribution(1.0, 69)
    assert observed == pytest.approx(expected)


def test_exponential_distribution_2():
    beta = 0.0

    observed = exponential_distribution(0.0)
    assert observed == pytest.approx(0.0)


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
    space_state_updater(space_list, 0.0, 1.0, empty_time_state, 0.0, 1.0)

    assert space_list == initial_space_list


def test_space_state_updater_2():
    time_state = [1.0, 2.0, 3.0, 4.0, 5.0]
    gaussian_parameters = (0.0, 1.0, 50)

    space_state = [0.1]
    s_min, s_max = 0.0, 1.0
    space_state_updater(space_state, s_min, s_max,
                        time_state, *gaussian_parameters)

    assert len(space_state) == len(time_state)


def test_space_state_updater_3():
    time_state = [1.0, 2.0, 3.0, 4.0, 5.0]
    seed = None
    gaussian_parameters = (0.0, 1.0, seed)
    s_min, s_max = 0.0, 1.0

    observed = [0.1]
    space_state_updater(observed, s_min, s_max,
                        time_state, *gaussian_parameters)

    out_of_bounds_counter = 0
    expected = [0.1]
    for time in time_state[1:]:
        gaussian = gaussian_distribution(*gaussian_parameters)
        new_point = brownian_formula(expected[-1], 1.0, gaussian)
        expected.append(new_point)
        if new_point < s_min or new_point > s_max:
            out_of_bounds_counter += 1

    if out_of_bounds_counter == 0:
        assert observed == expected
    else:
        assert observed != expected


def test_space_state_updater_4():
    time_state = [1.0, 2.0, 3.0, 4.0, 5.0]
    seed = None
    gaussian_parameters = (0.0, 1.0, seed)
    s_min, s_max = 0.0, 10.0

    observed = [0.1]
    space_state_updater(observed, s_min, s_max,
                        time_state, *gaussian_parameters)

    assert min(observed) >= s_min
    assert max(observed) <= s_max


def test_space_state_updater_5():
    out_of_bounds_origin = 500.0
    time_state = [1.0, 2.0, 3.0, 4.0, 5.0]
    seed = None
    gaussian_parameters = (0.0, 1.0, seed)
    s_min, s_max = 0.0, 10.0

    with pytest.raises(ValueError):
        observed = [out_of_bounds_origin]
        space_state_updater(observed, s_min, s_max,
                            time_state, *gaussian_parameters)
