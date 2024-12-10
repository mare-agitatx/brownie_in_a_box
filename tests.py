import br_simulation as brs
import br_analysis as bra
import pytest
import json
import numpy as np


################################################################################
# tests for br_simulation.py
def test_gaussian_distribution_1():
    '''
    Testing reproducibility of the distribution: to do so, some values are
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
    observed = [brs.gaussian_distribution(mu, sigma, rng),
                brs.gaussian_distribution(mu, sigma, rng),
                brs.gaussian_distribution(mu, sigma, rng),
                brs.gaussian_distribution(mu, sigma, rng)]
    for i in range(len(expected)):
        assert observed[i] == pytest.approx(expected[i])


def test_gaussian_distribution_2():
    '''
    Testing that, with a 0.0 standard deviation, the gaussian gives back
    the mu value as output.
    '''
    mu = 2.0
    observed = brs.gaussian_distribution(mu, 0.0)
    assert observed == pytest.approx(mu)


def test_exponential_distribution_1():
    '''
    Testing reproducibility of the distribution: to do so, some values are
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
    observed = [brs.exponential_distribution(beta, rng),
                brs.exponential_distribution(beta, rng),
                brs.exponential_distribution(beta, rng),
                brs.exponential_distribution(beta, rng)]
    assert observed == expected


def test_exponential_distribution_2():
    '''
    Testing that, with a 0.0 beta value (which is like an infinite rate
    value, since beta = 1/rate), the exponential distribution gives back 0.0.
    '''
    beta = 0.0

    observed = brs.exponential_distribution(0.0)
    assert observed == pytest.approx(0.0)


def test_uniform_distribution_1():
    '''
    Testing reproducibility of the distribution: to do so, some values are
    generated from a seed, the default_rng of numpy and the normal
    distribution are then called with that default generator.
    Then the rng is reset and the distribution to be tested is called with
    that rng to check that the generated values are the same of those
    generated with the standard method.
    '''
    rng = np.random.default_rng(911)
    min, max = 0.0, 1.0
    expected = [rng.uniform(min, max), rng.uniform(min, max),
                rng.uniform(min, max), rng.uniform(min, max)]

    # the rng must be set again or the next calls
    # to the distribution will proceed with the
    # following pseudo-random numbers in the sequence,
    # while we want to check that the same distribution
    # values are drawn from the same pseudo-random numbers
    rng = np.random.default_rng(911)
    observed = [brs.uniform_distribution(min, max, rng),
                brs.uniform_distribution(min, max, rng),
                brs.uniform_distribution(min, max, rng),
                brs.uniform_distribution(min, max, rng)]
    assert observed == expected


def test_brownian_formula_1d_1():
    '''
    Testing that the formula gives the expected value with the same values
    computed in another way.
    '''
    dt = 2.0
    seed = 20
    rng = np.random.default_rng(seed)
    gaussian = brs.gaussian_distribution(0.0, 1.0, rng)
    previous_point = 3.0
    expected = 3.0 + gaussian * np.sqrt(dt).astype(float)

    # the rng must be set again or the next call
    # to the distribution will proceed with the
    # following pseudo-random numbers in the sequence,
    # while we want to check the formula with the
    # same generated value
    rng = np.random.default_rng(seed)
    observed = brs.brownian_formula_1d(previous_point, dt, rng)
    assert observed == pytest.approx(expected)


def test_brownian_formula_1d_2():
    '''
    Testing that by giving a zero time interval dt then the point will
    retain the old position.
    '''
    dt = 0.0
    previous_point = 1.0

    observed = brs.brownian_formula_1d(previous_point, dt)
    assert observed == pytest.approx(previous_point)


def test_brownian_formula_1d_3():
    '''
    Testing that, by giving a negative time interval dt, brownian_formula()
    raises a ValueError.
    '''
    with pytest.raises(ValueError):
        dt = -1.0
        brs.brownian_formula_1d(1.0, dt)


def test_brownian_formula_1d_4():
    '''
    Testing that the function has proper replicability by calling it
    multiple times and then doing it again with the rng reset, and then
    comparing the results stored in different lists.
    '''
    n_repetitions = 100
    list1, list2 = [], []
    seed = 900
    dt = 1.0

    rng = np.random.default_rng(seed)
    observed = 0.0
    for index in range(n_repetitions):
        observed = brs.brownian_formula_1d(observed, dt,  rng)
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
        observed = brs.brownian_formula_1d(observed, dt, rng)
        list2.append(observed)

    assert list1 == list2


def test_is_point_out_of_boundaries_1():
    '''
    Checking that if the point is outside the bounds then the boolean from 
    the function will be True.
    '''
    point_to_check = 50.0
    s_min, s_max = 0.0, 1.0
    result = brs.is_point_out_of_boundaries(point_to_check, s_min, s_max)
    assert result is True


def test_is_point_out_of_boundaries_2():
    '''
    Checking that if the point is inside the bounds then the boolean from 
    the function will be False.
    '''
    point_to_check = 0.5
    s_min, s_max = 0.0, 1.0
    result = brs.is_point_out_of_boundaries(point_to_check, s_min, s_max)
    assert result is False


def test_enforce_boundaries_1d_1():
    '''
    Testing that no matter how many times the brownian formula is called, the
    final position will always be between the interval boundaries by means of
    the function that enforces the border.
    '''
    n_repetitions = 100
    dt = 2.0
    gaussian_parameters = (0.0, 1.0)
    s_min, s_max = 0.0, 10.0

    observed = 0.5
    for i in range(n_repetitions):
        observed = brs.brownian_formula_1d(observed, dt)
        observed = brs.enforce_boundaries_1d(observed, s_min, s_max)

    assert observed > s_min
    assert observed < s_max


def test_space_state_updater_1():
    '''
    Testing that, with a zero time interval dt, the space updater leaves the
    point unchanged.
    '''
    x, y, z = 0.1, 0.1, 0.1
    x_min, y_min, z_min = -10.0, -10.0, -10.0
    x_max, y_max, z_max = 10.0, 10.0, 10.0
    x_0, y_0, z_0 = x, y, z
    dt = 0.0
    point = brs.space_state_updater(x, y, z, x_min, y_min, z_min,
                                    x_max, y_max, z_max, dt)

    assert x == pytest.approx(x_0)
    assert y == pytest.approx(y_0)
    assert z == pytest.approx(z_0)
    

def test_space_state_updater_2():
    '''
    Testing that, with an origin that is out of bounds, the space updater
    raises a ValueError.
    '''
    dt = 2.0
    x, y, z = 300.0, 800.0, 550.0
    x_min, y_min, z_min = -10.0, -10.0, -10.0
    x_max, y_max, z_max = 10.0, 10.0, 10.0

    with pytest.raises(ValueError):
        point = brs.space_state_updater(x, y, z, x_min, y_min, z_min,
                                        x_max, y_max, z_max, dt)


def test_draw_random_event_1():
    '''
    Testing that the event generated by this function is always one of the
    given transition names.
    '''
    rates = [0.1, 0.1, 0.1]
    transitions_names = ['event1', 'event2', 'event3']
    event = brs.draw_random_event(transitions_names, rates)
    assert event in transitions_names


def test_draw_random_event_2():
    '''
    Testing that, by passing a different number of rates and events, the
    function throws a ValueError.
    '''
    rates = [0.1, 0.1]
    transitions_names = ['event1', 'event2', 'event3', 'event4']
    with pytest.raises(ValueError):
        event = brs.draw_random_event(transitions_names, rates)


def test_is_float_strictly_lesser_1():
    '''
    Testing that, if value_1 is less than value_2 as expected, the
    function recognizes it.
    '''
    value_1 = 5.0
    value_2 = 10.0
    assert brs.is_float_strictly_lesser(value_1, value_2) is True


def test_is_float_strictly_lesser_2():
    '''
    Testing that, if value_2 is more than value_1 as expected, the
    function recognizes it.
    '''
    value_1 = 5.0
    value_2 = 10.0
    assert brs.is_float_strictly_lesser(value_2, value_1) is False


def test_is_float_strictly_lesser_3():
    '''
    Testing that the value is not strictly less than itself, and
    the function recognizes it.
    '''
    value = 10.0
    assert brs.is_float_strictly_lesser(value, value) is False


def test_run_simulation_1():
    '''
    Testing that, by giving an initial time already at the time limit value,
    the simulation returns a list with the same number of bacteria
    initialized from the start.
    '''
    t_0, time_limit = 30.0, 30.0
    x_0, y_0, z_0 = 0.0, 0.0, 0.0
    x_min, y_min, z_min = -10.0, -10.0, -10.0
    x_max, y_max, z_max = 10.0, 10.0, 10.0
    death_coeff, reprod_coeff, move_coeff = 0.1, 0.1, 1.0
    initial_bacteria_number, bacteria_limit = 2, 1000000
    bact_list, flag, max_time = brs.run_simulation(x_0, y_0, z_0, x_min, y_min,
                                                   z_min, x_max, y_max, z_max,
                                                   t_0, time_limit,
                                                   death_coeff, reprod_coeff,
                                                   move_coeff,
                                                   initial_bacteria_number,
                                                   bacteria_limit)

    assert len(bact_list) == initial_bacteria_number


def test_run_simulation_2():
    '''
    Testing that, by giving an initial time already at the time limit value,
    the simulation returns a max time reached that is t_0.
    '''
    t_0, time_limit = 30.0, 30.0
    x_0, y_0, z_0 = 0.0, 0.0, 0.0
    x_min, y_min, z_min = -10.0, -10.0, -10.0
    x_max, y_max, z_max = 10.0, 10.0, 10.0
    death_coeff, reprod_coeff, move_coeff = 0.1, 0.1, 1.0
    initial_bacteria_number, bacteria_limit = 2, 1000000
    bact_list, flag, max_time = brs.run_simulation(x_0, y_0, z_0, x_min, y_min,
                                                   z_min, x_max, y_max, z_max,
                                                   t_0, time_limit,
                                                   death_coeff, reprod_coeff,
                                                   move_coeff,
                                                   initial_bacteria_number,
                                                   bacteria_limit)

    assert max_time == pytest.approx(t_0)


def test_run_simulation_3():
    '''
    Testing reproducibility of the simulation: a seed is fixed that will be
    passed to the numpy default_rng inside the simulation body. The
    simulation is called two times with the same inputs and stored in two
    different lists. The test checks that the lists have the same elements.
    '''
    t_0, time_limit = 0.0, 10.0
    x_0, y_0, z_0 = 0.0, 0.0, 0.0
    x_min, y_min, z_min = -10.0, -10.0, -10.0
    x_max, y_max, z_max = 10.0, 10.0, 10.0
    death_coeff, reprod_coeff, move_coeff = 0.1, 0.1, 1.0
    initial_bacteria_number, bacteria_limit, seed = 2, 1000000, 420
    result_1 = brs.run_simulation(x_0, y_0, z_0, x_min, y_min,
                                  z_min, x_max, y_max, z_max,
                                  t_0, time_limit, death_coeff, reprod_coeff,
                                  move_coeff, initial_bacteria_number,
                                  bacteria_limit, seed)
    result_2 = brs.run_simulation(x_0, y_0, z_0, x_min, y_min,
                                  z_min, x_max, y_max, z_max,
                                  t_0, time_limit, death_coeff, reprod_coeff,
                                  move_coeff, initial_bacteria_number,
                                  bacteria_limit, seed)

    assert result_1 == result_2


def test_date_name_file_1():
    '''
    Testing that the name produced by the function is effectively a string.
    '''
    result = brs.date_name_file()
    assert type(result) == str


def test_date_name_file_2():
    '''
    Testing that the function will raise a TypeError if an extension
    that isn't a string is passed.
    '''
    with pytest.raises(TypeError):
        result = brs.date_name_file(33.3)


def test_save_data_json_1(tmp_path):
    '''
    Testing that the function will properly save a list in .json format, by
    saving it and then reloading the file to check that the saved list is
    the same as the original one.
    tmp_path is employed in this test, a temporary folder system that works
    under pytest.
    '''
    data = [1, 2, 3, 4]
    results_folder = str(tmp_path) + '/'
    brs.save_data_json(data, 'test.json', results_folder)

    filepath = tmp_path / 'test.json'
    with open(filepath, 'r') as json_file:
        observed = json.load(json_file)
        assert observed == data


################################################################################
# tests for br_analysis.py
def test_load_data_json_1(tmp_path):
    '''
    Testing that the function will properly load a dictionary by first
    saving it and then loading it with the said function and checking that
    the loaded dictionary is equal to the original one.
    tmp_path is employed in this test, a temporary folder system that works
    under pytest.
    '''
    filepath = tmp_path / 'test.json'
    data = {'1': 1, '2': 2, '3': 3}
    with open(filepath, 'a') as json_file:
        json.dump(data, json_file, indent=4)

    str_path = str(filepath)
    observed = bra.load_data_json(str_path)
    assert observed == data


def test_parse_data_dict_1():
    '''
    Testing that with an empty dictionary as input, the parsing function will
    raise a ValueError.
    '''
    with pytest.raises(ValueError):
        bra.parse_data_dict({})
