from simulation import *
from analysis import *
import pytest


##############################################################################
# tests for simulation.py
def test_gaussian_distribution_1():
    '''Testing reproducibility of the distribution: to do so some values are
       generated from a seed, the default_rng of numpy and the normal
       distribution are then called with that default generator.
       Then the rng is reset and the distribution to be tested is called with
       that rng to check that the generated values are the same of those
       generated with the standard method.
    '''
    rng = np.random.default_rng(42)
    mu, sigma = 0.0, 1.0
    expected = [rng.normal(mu, sigma), rng.normal(mu, sigma),
                rng.normal(mu, sigma), rng.normal(mu, sigma)]

    # the rng must be set again or the next calls
    # to the distribution will proceed with the
    # following pseudo-random numbers in the sequence,
    # while we want to check that the same distribution
    # values are drawn from the same pseudo-random numbers
    rng = np.random.default_rng(42)
    observed = [gaussian_distribution(mu, sigma, rng),
                gaussian_distribution(mu, sigma, rng),
                gaussian_distribution(mu, sigma, rng),
                gaussian_distribution(mu, sigma, rng)]
    for i in range(len(expected)):
        assert observed[i] == pytest.approx(expected[i])


def test_gaussian_distribution_2():
    '''Testing that, with a 0.0 standard deviation, the gaussian gives back
       the mu value as output.
    '''
    mu = 2.0
    observed = gaussian_distribution(mu, 0.0)
    assert observed == pytest.approx(mu)


def test_exponential_distribution_1():
    '''Testing reproducibility of the distribution: to do so some values are
       generated from a seed, the default_rng of numpy and the normal
       distribution are then called with that default generator.
       Then the rng is reset and the distribution to be tested is called with
       that rng to check that the generated values are the same of those
       generated with the standard method.
    '''
    rng = np.random.default_rng(69)
    beta = 1.0
    expected = [rng.exponential(beta), rng.exponential(beta),
                rng.exponential(beta), rng.exponential(beta)]

    # the rng must be set again or the next calls
    # to the distribution will proceed with the
    # following pseudo-random numbers in the sequence,
    # while we want to check that the same distribution
    # values are drawn from the same pseudo-random numbers
    rng = np.random.default_rng(69)
    observed = [exponential_distribution(beta, rng),
                exponential_distribution(beta, rng),
                exponential_distribution(beta, rng),
                exponential_distribution(beta, rng)]
    assert observed == expected


def test_exponential_distribution_2():
    '''Testing that, with a 0.0 beta value (which is like an infinite rate
       value, since beta = 1/rate), the exponential distribution gives back 0.0.
    '''
    beta = 0.0

    observed = exponential_distribution(0.0)
    assert observed == pytest.approx(0.0)


def test_brownian_formula_1():
    '''Testing that the formula gives the expected value with the same values
       computed in another way.
    '''
    dt = 2.0
    seed = 20
    rng = np.random.default_rng(seed)
    gaussian = gaussian_distribution(0.0, 1.0, rng)
    previous_point = 3.0
    expected = 3.0 + gaussian * 1.41421356

    # the rng must be set again or the next call
    # to the distribution will proceed with the
    # following pseudo-random numbers in the sequence,
    # while we want to check the formula with the
    # same generated value
    rng = np.random.default_rng(seed)
    observed = brownian_formula(previous_point, dt, rng)
    assert observed == pytest.approx(expected)


def test_brownian_formula_2():
    '''Testing that by giving a zero time interval dt then the point will
       retain the old position.
    '''
    dt = 0.0
    previous_point = 1.0

    observed = brownian_formula(previous_point, dt)
    assert observed == pytest.approx(previous_point)


def test_brownian_formula_3():
    '''Testing that, by giving a negative time interval dt, brownian_formula()
       raises a ValueError.
    '''
    with pytest.raises(ValueError):
        dt = -1.0
        brownian_formula(1.0, dt)


def test_brownian_formula_4():
    '''Testing that the function has proper replicability by calling it
       multiple times and then doing it again with the rng reset and
       comparing the results stored in different lists.
    '''
    n_repetitions = 100
    list1, list2 = [], []
    seed = 900
    dt = 1.0

    rng = np.random.default_rng(seed)
    observed = 0.0
    for index in range(n_repetitions):
        observed = brownian_formula(observed, dt,  rng)
        list1.append(observed)

    # the rng must be set again or the next call
    # to the distribution will proceed with the
    # following pseudo-random numbers in the sequence,
    # while we want to check the formula with the
    # same generated value
    rng = np.random.default_rng(seed)
    # observed must also be reset
    observed = 0.0
    for index in range(n_repetitions):
        observed = brownian_formula(observed, dt, rng)
        list2.append(observed)

    assert list1 == list2


def test_space_state_updater_1():
    '''Testing that, with a zero time interval dt, the space updater leaves the
       point unchanged.
    '''
    point = 0.1
    initial_point = point
    point = space_state_updater(point, 0.0, 1.0, 0.0)

    assert point == pytest.approx(initial_point)


def test_space_state_updater_2():
    '''Testing that no matter how many times the space updater is called, the
       final position will always be between the interval boundaries.
    '''
    n_repetitions = 100
    dt = 2.0
    gaussian_parameters = (0.0, 1.0)
    s_min, s_max = 0.0, 10.0

    observed = 0.5
    for i in range(n_repetitions):
        observed = space_state_updater(observed, s_min, s_max,
                                       dt)

    assert observed >= s_min
    assert observed <= s_max


def test_space_state_updater_3():
    '''Testing that, with an origin that is out of bounds, the space updater
       raises a ValueError.
    '''
    dt = 2.0
    gaussian_parameters = (0.0, 1.0)
    s_min, s_max = 0.0, 10.0

    with pytest.raises(ValueError):
        out_of_bounds_origin = 500.0
        observed = space_state_updater(out_of_bounds_origin,
                                       s_min, s_max, dt)


def test_draw_random_event_1():
    '''Testing that the event generated by this function is always one of the
       given transition names.
    '''
    rates = [0.1, 0.1, 0.1]
    transitions_names = ['event1', 'event2', 'event3']
    event = draw_random_event(transitions_names, rates)
    assert event in transitions_names


def test_draw_random_event_2():
    '''Testing that, by passing a different number of rates and events, the
       function throws a ValueError.
    '''
    rates = [0.1, 0.1]
    transitions_names = ['event1', 'event2', 'event3', 'event4']
    with pytest.raises(ValueError):
        event = draw_random_event(transitions_names, rates)


def test_is_float_strictly_lesser_1():
    '''Testing that value_1 is less than value_2 as expected, and the
       function recognizes it.
    '''
    value_1 = 5.0
    value_2 = 10.0
    assert is_float_strictly_lesser(value_1, value_2) is True


def test_is_float_strictly_lesser_2():
    '''Testing that value_2 is more than value_1 as expected, and the
       function recognizes it.
    '''
    value_1 = 5.0
    value_2 = 10.0
    assert is_float_strictly_lesser(value_2, value_1) is False


def test_is_float_strictly_lesser_3():
    '''Testing that the value is not strictly less than itself, and
       the function recognizes it.
    '''
    value = 10.0
    assert is_float_strictly_lesser(value, value) is False


def test_run_simulation_1():
    '''Testing that, by giving an initial time already at the time limit value,
       the simulation returns a list with only the same number of bacteria
       initialized from the start.
    '''
    t_0, time_limit = 30.0, 30.0
    x_0, y_0 = 0.0, 0.0
    x_min, x_max = 0.0, 20.0
    y_min, y_max = -10.0, 10.0
    start_bacteria = 2
    result = run_simulation(x_0, y_0, x_min, x_max,
                   y_min, y_max, t_0, time_limit, start_bacteria)

    assert len(result) == start_bacteria


def test_run_simulation_2():
    '''Testing reproducibility of the simulation: a seed is fixed that will be
       passed to the numpy default_rng inside the simulation body. The
       simulation is called two times with the same inputs and stored in two
       different lists. The test checks that the lists have the same elements.
    '''
    t_0, time_limit = 0.0, 10.0
    x_0, y_0 = 0.0, 0.0
    x_min, x_max = 0.0, 20.0
    y_min, y_max = -10.0, 10.0
    start_bacteria, seed = 2, 420
    result_1 = run_simulation(x_0, y_0, x_min, x_max,
                              y_min, y_max, t_0, time_limit,
                              start_bacteria, seed)
    result_2 = run_simulation(x_0, y_0, x_min, x_max,
                              y_min, y_max, t_0, time_limit,
                              start_bacteria, seed)

    assert result_1 == result_2


def test_date_name_file_1():
    '''Testing that the result is effectively a string.
    '''
    result = date_name_file()
    assert type(result) == str


def test_date_name_file_2():
    '''Testing that the function will raise a TypeError if an extension
       that isn't a string is passed.
    '''
    with pytest.raises(TypeError):
        result = date_name_file(33.3)


def test_save_data_json_1(tmp_path):
    '''
    '''
    data = [1, 2, 3, 4]
    results_folder = str(tmp_path) + '/'
    save_data_json(data, 'test.json', results_folder)

    filepath = tmp_path / 'test.json'
    with open(filepath, 'r') as json_file:
        observed = json.load(json_file)
        assert observed == data


##############################################################################
# tests for analysis.py
def test_load_data_json_1(tmp_path):
    '''
    '''
    filepath = tmp_path / 'test.json'
    data = {'1': 1, '2': 2, '3': 3}
    with open(filepath, 'a') as json_file:
        json.dump(data, json_file, indent=4)

    str_path = str(filepath)
    observed = load_data_json(str_path)
    assert observed == data


def test_parse_data_dictionary_1():
    '''
    '''
    with pytest.raises(ValueError):
        parse_data_dictionary({}, 1)
