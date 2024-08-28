from brownie_box import *
import pytest


def test_brownian_formula_1():
    '''
    '''
    dt = 2.0
    gaussian = 0.5
    previous_point = 3.0

    expected = 3.0 + 0.5 * 1.41421356
    observed = brownian_formula(previous_point, dt, gaussian)
    assert observed == pytest.approx(expected)


def test_brownian_formula_2():
    '''
    '''
    dt = 0.0
    gaussian = 0.0
    previous_point = 0.0

    expected = 0.0
    observed = brownian_formula(previous_point, dt, gaussian)
    assert observed == pytest.approx(expected)


def test_brownian_formula_3():
    '''
    '''
    with pytest.raises(ValueError):
        dt = -1.0
        brownian_formula(1.0, dt, 1.0)


def test_gaussian_distribution_1():
    '''
    '''
    rng = np.random.default_rng(42)
    mu, sigma = 0.0, 1.0
    expected = [rng.normal(mu, sigma), rng.normal(mu, sigma)]

    # the seed must be set again or the next calls
    # to the distribution will proceed with the
    # following pseudo-random numbers in the sequence,
    # while we want to check that the same distribution
    # values are drawn from the same pseudo-random numbers
    rng = np.random.default_rng(42)
    observed = [gaussian_distribution(0.0, 1.0, rng),
                gaussian_distribution(0.0, 1.0, rng)]
    for i in range(len(expected)):
        assert observed[i] == pytest.approx(expected[i])


def test_gaussian_distribution_2():
    '''
    '''
    mu = 2.0
    observed = gaussian_distribution(mu, 0.0)
    assert observed == pytest.approx(mu)


def test_exponential_distribution_1():
    '''
    '''
    rng = np.random.default_rng(69)
    beta = 1.0
    expected = [rng.exponential(beta), rng.exponential(beta)]

    # the seed must be set again or the next calls
    # to the distribution will proceed with the
    # following pseudo-random numbers in the sequence,
    # while we want to check that the same distribution
    # values are drawn from the same pseudo-random numbers
    rng = np.random.default_rng(69)
    observed = [exponential_distribution(beta, rng),
                exponential_distribution(beta, rng)]
    assert observed == expected


def test_exponential_distribution_2():
    '''
    '''
    beta = 0.0

    observed = exponential_distribution(0.0)
    assert observed == pytest.approx(0.0)


def test_space_state_updater_1():
    '''
    '''
    point = 0.1
    initial_point = point
    dt = 0.0
    space_state_updater(point, 0.0, 1.0, dt, 0.0, 1.0)

    assert point == pytest.approx(initial_point)


def test_space_state_updater_2():
    '''
    '''
    n_repetitions = 100
    dt = 2.0
    gaussian_parameters = (0.0, 1.0)
    s_min, s_max = 0.0, 10.0

    observed = 0.5
    for i in range(n_repetitions):
        observed = space_state_updater(observed, s_min, s_max,
                                       dt, *gaussian_parameters)

    assert observed >= s_min
    assert observed <= s_max


def test_space_state_updater_3():
    '''
    '''
    dt = 2.0
    gaussian_parameters = (0.0, 1.0)
    s_min, s_max = 0.0, 10.0

    with pytest.raises(ValueError):
        out_of_bounds_origin = 500.0
        observed = space_state_updater(out_of_bounds_origin,
                                       s_min, s_max, dt, *gaussian_parameters)


def test_draw_random_event_1():
    '''
    '''
    rates = [0.1, 0.1, 0.1]
    transitions_names = ['event1', 'event2', 'event3']
    event = draw_random_event(transitions_names, rates)
    assert event in transitions_names


def test_draw_random_event_2():
    '''
    '''
    rates = [0.1, 0.1]
    transitions_names = ['event1', 'event2', 'event3', 'event4']
    with pytest.raises(ValueError):
        event = draw_random_event(transitions_names, rates)


def test_run_simulation_1():
    '''
    '''
    t_0, time_limit = 30.0, 30.0
    x_0, y_0 = 0.0, 0.0
    x_min, x_max = 0.0, 20.0
    y_min, y_max = -10.0, 10.0
    result = run_simulation(x_0, y_0, x_min, x_max,
                   y_min, y_max, t_0, time_limit)

    assert result == []


def test_run_simulation_2():
    '''
    '''
    t_0, time_limit = 0.0, 30.0
    x_0, y_0 = 0.0, 0.0
    x_min, x_max = 0.0, 20.0
    y_min, y_max = -10.0, 10.0
    seed = 420
    result_1 = run_simulation(x_0, y_0, x_min, x_max,
                              y_min, y_max, t_0, time_limit, seed)
    result_2 = run_simulation(x_0, y_0, x_min, x_max,
                              y_min, y_max, t_0, time_limit, seed)

    assert result_1 == result_2
