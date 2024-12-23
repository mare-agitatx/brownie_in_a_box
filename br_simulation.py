import numpy as np
import json
import sys
from enum import Enum
from datetime import datetime
import configparser


def gaussian_distribution(mu, sigma, rng=None):
    '''
    It computes a gaussian distribution's value.
    Parameters:
        mu: float, the gaussian average parameter.
        sigma: float, the gaussian standard deviation parameter.
        rng: numpy.random.Generator, the generator of random numbers if passed
             as input, defaults at None if it isn't passed.
    Returns:
        output: float, the computed value of the distribution.
    '''
    if rng is None:
        output = np.random.normal(mu, sigma)
    else:
        random_number_generator = rng
        output = random_number_generator.normal(mu, sigma)

    return output


def exponential_distribution(beta, rng=None):
    '''
    It computes an exponential distribution's value.
    Parameters:
        beta: float, the beta value of the exponential, equal to
              1/rate if the distribution is thought as function of rate.
        rng: numpy.random.Generator, the generator of random numbers if passed
             as input, defaults at None if it isn't passed.
    Returns:
        output: float, the computed value of the distribution.
    '''
    if rng is None:
        output = np.random.exponential(beta)
    else:
        random_number_generator = rng
        output = random_number_generator.exponential(beta)

    return output


def uniform_distribution(min, max, rng=None):
    '''
    It computes an uniform distribution's value.
    Parameters:
        min: float, the minimum of the range of possible values.
        min: float, the maximum of the range of possible values.
        rng: numpy.random.Generator, the generator of random numbers if passed
             as input, defaults at None if it isn't passed.
    Returns:
        output: float, the computed value of the distribution.
    '''
    if rng is None:
        output = np.random.uniform(min, max)
    else:
        random_number_generator = rng
        output = random_number_generator.uniform(min, max)

    return output


def brownian_formula_1d(starting_point, dt, rng=None):
    '''
    A simple formula to implement Brownian motion.
    Parameters:
        starting_point: float, previous position to update.
        dt: float, time interval of the movement.
        rng: numpy.random.Generator, the generator of random numbers if passed
             as input, defaults at None if it isn't passed.
    Returns:
        new_point: float, the new position computed.
    Raises:
        ValueError: if dt is negative.
    '''
    if dt < 0.0:
        raise ValueError('Given a negative time gap to dt.')

    gaussian_term = gaussian_distribution(0.0, 1.0, rng)
    new_point = starting_point + np.sqrt(dt).astype(float) * gaussian_term
    return new_point


def is_point_out_of_boundaries(point_to_check, interval_min, interval_max):
    '''
    A boolean check function for a point in an interval.
    Parameters:
        point_to_check: float, the point to check.
        interval_min: float, the minimum of the interval.
        interval_max: float, the maximum of the interval.
    Returns:
        check: boolean, the result of the comparisons.
    '''
    check = (point_to_check < interval_min or point_to_check > interval_max)
    return check


def enforce_boundaries_1d(point_to_check, interval_min, interval_max):
    '''
    It updates a point's position if that point lies outside of the given
    boundaries, reflecting it back inside the interval, else it leaves
    the position unchanged.
    Parameters:
        point_to_check: float, the point to check.
        interval_min: float, the minimum of the interval.
        interval_max: float, the maximum of the interval.
    Returns:
        point_checked: float, the point eventually reflected back in the
                       interval or else left unscathed.
    '''
    if point_to_check < interval_min:
        gap = abs(point_to_check - interval_min)
        reflected_position = interval_min + gap
        point_checked = reflected_position
    elif point_to_check > interval_max:
        gap = abs(interval_max - point_to_check)
        reflected_position = interval_max - gap
        point_checked = reflected_position
    else:
        point_checked = point_to_check

    return point_checked


def space_state_updater(x, y, z, x_min, y_min, z_min,
                        x_max, y_max, z_max, dt, rng=None):
    '''
    It updates a given point position by drawing randomly a direction on a
    spherical surface (to implement isotropy) and then it jumps on that 
    direction by a random gap drawn by the brownian formula. These calculations
    are made in spherical coordinates and then projected in cartesian ones.
    Parameters:
        x, y, z: floats, the coordinates the function works on.
        x_min, y_min, z_min: floats, interval minimums.
        x_max, y_max, z_max: floats, interval maximums.
        dt: float, the interval of time in which the transition happens.
        rng: numpy.random.Generator, the generator of random numbers if passed
             as input, defaults at None if it isn't passed.
    Returns:
        x, y, z: floats, the coordinates the function works on.
    Raises:
        ValueError: if the coordinates at the start are out of bounds.
    '''
    if is_point_out_of_boundaries(x, x_min, x_max) is True:
        raise ValueError('The given x coordinate is out of bounds.')
    if is_point_out_of_boundaries(y, y_min, y_max) is True:
        raise ValueError('The given y coordinate is out of bounds.')
    if is_point_out_of_boundaries(z, z_min, z_max) is True:
        raise ValueError('The given z coordinate is out of bounds.')

    # draw a direction on a sphere with uniform distribution for
    # said direction; this is to implement isotropy, AKA no
    # favorite direction in space. Apparently, if one generates
    # both phi and theta directly from the uniform distribution,
    # then the final spherical shape will be biased on the poles:
    # the solution is then to draw randomly the cosine of theta
    # to obtain a uniform sphere
    phi = uniform_distribution(0, 2 * np.pi, rng)
    cos_of_theta = uniform_distribution(-1, 1, rng)

    # sin_of_theta is calculated as plus the square root
    # of a function of the cosine since theta is always between 0 and pi
    # so its sine is always positive
    sin_of_theta = np.sqrt(1 - cos_of_theta ** 2).astype(float)

    # draw a radius from the brownian motion,
    # so the leap forward is on the direction
    # drawn before and is regulated by the
    # brownian formula, then it is projected
    # on the three coordinates
    dr = brownian_formula_1d(0, dt, rng)
    x += dr * sin_of_theta * np.cos(phi)
    y += dr * sin_of_theta * np.sin(phi)
    z += dr * cos_of_theta

    # checking that new values are inside
    # boundaries and if not, those
    # boundaries are then enforced
    x = enforce_boundaries_1d(x, x_min, x_max)
    y = enforce_boundaries_1d(y, y_min, y_max)
    z = enforce_boundaries_1d(z, z_min, z_max)

    return x, y, z


class Event(Enum):
    DEATH = 'death'
    REPRODUCTION = 'reproduction'
    MOVEMENT = 'movement'
    BIRTH = 'birth'


def death(death_coefficient):
    '''
    Probability for the death event.
    Parameters:
        death_coefficient: float, influences the final proability.
    Returns:
        death_coefficient: float, probability value.
    '''
    return death_coefficient


def reproduction(reproduction_coefficient):
    '''
    Probability for the reproduction event.
    Parameters:
        reproduction_coefficient: float, influences the final proability.
    Returns:
        reproduction_coefficient: float, probability value.
    '''
    return reproduction_coefficient


def movement(movement_coefficient):
    '''
    Probability for the movement event.
    Parameters:
        movement_coefficient: float, influences the final proability.
    Returns:
        movement_coefficient: float, probability value.
    '''
    return movement_coefficient


def draw_random_event(transition_names, transition_rates, rng=None):
    '''
    It draws randomly an event string given names and rates.
    Parameters:
        transition_names = iterable of strings, the events to draw from.
        transition_rates = iterable of floats, the rates related to
                           probabilities of those events happening,
                           not normalized.
        rng = numpy.random.Generator, the generator of random numbers if passed
              as input, defaults at None if it isn't passed.
    Returns:
        event: string, the event drawn randomly.
    Raises:
        ValueError: if the iterables of transition_names and transition_rates
                    have different lengths.
    '''
    if len(transition_names) != len(transition_rates):
        raise ValueError('Names and rates for the transitions ' +
                         'must be equal in number.')

    # normalizing the rates
    normalized_rates = np.cumsum(transition_rates)
    normalized_rates /= normalized_rates[-1]

    # drawing the random probability
    probability = uniform_distribution(0, 1, rng)

    # drawing the event from the probability
    index = np.searchsorted(normalized_rates, probability)
    event = transition_names[index]
    return event


def run_simulation(x_0, y_0, z_0, x_min, y_min, z_min, x_max, y_max, z_max,
                   t_0, time_limit, death_coefficient,
                   reproduction_coefficient, movement_coefficient,
                   initial_bacteria_number, bacteria_limit, seed=None):
    '''
    The function that runs the main simulation of the bacteria.
    Parameters:
        x_0, y_0 z_0: floats, coordinates of the origin.
        x_min, y_min, z_min: floats, interval minimums for the coordinates.
        x_max, y_max, z_max: floats, interval maximums for the coordinates.
        t_0, time_limit: floats, first instant and last instant
                         of the simulation.
        initial_bacteria_number: integer, number of bacteria born at t_0.
        bacteria_limit: integer, if more bacteria than this value are
                        spawned the simulation is halted.
        seed: integer if given, will be passed to the default number
              generator of numpy. If not passed will default to None
              and the rng won't be called.
    Returns:
        bacteria: list of dictionaries, the data generated by the simulation.
        flag: string, describes the way the simulation ended.
        max_time_reached: float, the last time value reached by any bacteria.
    Raises:
        ValueError: if an event is drawn randomly that isn't found in
                    transition_names.
    '''
    # initializing some variables
    times_to_print = np.linspace(t_0, time_limit, 11).tolist()
    active_bacteria_counter = initial_bacteria_number
    dead_bacteria_counter = 0
    transition_names = ('death', 'reproduction', 'movement')
    bacteria = []
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = None

    # defining here active_times or else
    # the assignment 'old_active_times = active_times'
    # in the while loop will raise an error. Also, by
    # fixing it to t_0, if the while loop skips the evolution
    # of bacteria for any reason, the simulation will end
    # with max_time_reached = t_0 which is the only
    # logical conclusion
    active_times = [t_0]

    # initialize bacteria as list of dictionaries;
    # states are represented by dictionaries in a list
    # in a voice of the bacterium dictionary
    for identifier_index in range(1, initial_bacteria_number +1):
        bacterium = {'bacterium_id': identifier_index,
                     'mother_id': 'init by user',
                     'states': []}

        rates = (death(death_coefficient),
                 reproduction(reproduction_coefficient),
                 movement(movement_coefficient))
        total_rate = sum(rates)
        dt = exponential_distribution(1/total_rate, rng)

        first_state = {'time_of_observation': t_0,
                       'interval_of_residency': dt,
                       'x': x_0, 'y': y_0, 'z': z_0,
                       'event': 'birth'}
        bacterium['states'].append(first_state)
        bacteria.append(bacterium)

    while active_bacteria_counter != 0:
        active_bacteria_counter = 0
        old_active_times = active_times
        active_times = []

        for bacterium in bacteria:
            last_state = bacterium['states'][-1]

            # print a time check every ten 
            # parts of the total time, in
            # order to have an output of the
            # running simulation, especially
            # should it happen that it gets stuck;
            # the current time and the alive bacteria
            # are printed
            for time in times_to_print:
                if max(old_active_times) > time:
                    print(f't={time:.2f} has been reached',
                          f'out of {time_limit:.2f}.')
                    print(f'The alive bacteria as of now are {len(bacteria)}',
                          f'out of {bacteria_limit}.')
                    times_to_print.remove(time)

            # ignore dead bacteria
            is_dead = (last_state['event'] == 'death')
            if is_dead is True:
                continue

            # computing new interval of residency
            rates = (death(death_coefficient),
                     reproduction(reproduction_coefficient),
                     movement(movement_coefficient))
            total_rate = sum(rates)
            dt = exponential_distribution(1/total_rate, rng)

            # computing the time of observation
            previous_time = last_state['time_of_observation']
            previous_dt = last_state['interval_of_residency']
            observed_time = previous_time + previous_dt

            # checking that the total time of this state
            # doesn't exceed the time_limit: if it doesn't,
            # the active counter keeping the while loop
            # running is increased... when every bacteria
            # that is still alive reaches the max time,
            # this counter will remain to 0, the while loop
            # will stop and the simulation will finish.
            # if instead total_time is over the time
            # limit, the bacterium will be skipped
            # from being updated; the bacterium also
            # will be skipped if its time is very close
            # to the limit as the inclusion of float values
            # deemed close by numpy.isclose() may
            # sometimes produce weird values and
            # precision errors
            total_time = observed_time + dt
            if total_time < time_limit:
                active_bacteria_counter += 1
                active_times.append(total_time)
            elif np.isclose(total_time, time_limit):
                continue
            elif total_time > time_limit:
                continue

            # computing the transition event
            event = draw_random_event(transition_names, rates, rng)

            # computing x and y, dependent of the drawn event
            if event == Event.REPRODUCTION.value:
                x, y, z = last_state['x'], last_state['y'], last_state['z']
            elif event == Event.DEATH.value:
                x, y, z = last_state['x'], last_state['y'], last_state['z']
            elif event == Event.MOVEMENT.value:
                x, y, z = space_state_updater(last_state['x'], last_state['y'],
                                              last_state['z'], x_min, y_min,
                                              z_min, x_max, y_max, z_max,
                                              dt, rng)
            else:
                raise ValueError('Unrecognized transition.')

            # registering spatial coordinates, times and
            # transition type into the state dictionary
            bacterium_state = {'time_of_observation': observed_time,
                               'interval_of_residency': dt,
                               'x': x, 'y': y, 'z': z,
                               'event': event}
            bacterium['states'].append(bacterium_state)

            # if the event is a reproduction, a new
            # bacterium is created in the same
            # position, and its mother id is registered
            if event == Event.REPRODUCTION.value:
                identifier_index += 1
                mother_index = bacterium['bacterium_id']
                new_bacterium = {'bacterium_id': identifier_index,
                                 'mother_id': mother_index,
                                 'states': []}
                new_state = {'time_of_observation': observed_time,
                             'interval_of_residency': dt,
                             'x': x, 'y': y, 'z': z,
                             'event': 'birth'}
                new_bacterium['states'].append(new_state)
                bacteria.append(new_bacterium)
            # if the event is death, the death counter
            # is increased
            elif event == Event.DEATH.value:
                dead_bacteria_counter += 1

        N = len(bacteria)
        # checking if every bacterium died:
        # the simulation is then stopped
        if N == dead_bacteria_counter:
            print('Simulation done. All bacteria died.')
            flag = 'all_dead'
            max_time = max(active_times)
            return bacteria, flag, max_time

        # checking if the total bacteria is
        # above the bacteria_limit: if yes
        # the simulation is halted
        if N > bacteria_limit:
            print('Simulation halted. Too many bacteria spawned.')
            flag = 'halted_simulation'
            max_time = max(active_times)
            return bacteria, flag, max_time

    print('Simulation done. Time limit reached.')
    flag = 'time_limit'
    max_time = max(old_active_times)
    return bacteria, flag, max_time


def date_name_file(extension=None):
    '''
    Produces a string with the numerical date and time to be employed as a
    filename, and optionally adds an extension.
    Parameters:
        extension: string if given, a string added to the end of the date
                   string. If not passed will default to None.
    Returns:
        filename: string, the filename to emply.
    Raises:
        TypeError: if the extension passed is not a string type.
    '''
    if extension is not None:
        if type(extension) != str:
            raise TypeError('The extension given must be a valid string.')

    # datetime object containing current date and time
    now = datetime.now()

    # YYYY_mm_dd_H_M_S format for the date string
    date_string = now.strftime('%Y_%m_%d_%H_%M_%S')

    if extension is None:
        filename = date_string
    else:
        filename = date_string + extension

    return filename


def save_data_json(data, filename, results_folder=None):
    '''
    Saves some data in json format with a given filename and folder.
    Parameters:
        data: anything that can be saved in json.
        filename: string, the name given to the saved file.
        results_folder: string if given, the subfolder where the file
                        will be placed. If not passed will default to None.
    '''
    if results_folder is not None:
        filepath = results_folder + filename
    else:
        filepath = filename
    with open(filepath, 'a') as json_file:
        json.dump(data, json_file, indent=4)


###############################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('No input given. Exiting...')
        print('Please input the configuration filename on the command line.')
        sys.exit()

    # import the parameters from CLI specified input file
    config = configparser.ConfigParser()
    input_file = sys.argv[1]
    config.read(input_file)

    print('Collecting parameters...')
    x_0 = float(config.get('parameters', 'x_0'))
    y_0 = float(config.get('parameters', 'y_0'))
    z_0 = float(config.get('parameters', 'y_0'))
    x_min = float(config.get('parameters', 'x_min'))
    z_min = float(config.get('parameters', 'z_min'))
    y_min = float(config.get('parameters', 'y_min'))
    x_max = float(config.get('parameters', 'x_max'))
    y_max = float(config.get('parameters', 'y_max'))
    z_max = float(config.get('parameters', 'z_max'))
    t_0 = float(config.get('parameters', 't_0'))
    time_limit = float(config.get('parameters', 'time_limit'))
    death_coefficient = float(config.get('parameters', 'death_coefficient'))
    reproduction_coefficient = float(config.get('parameters',
                                                'reproduction_coefficient'))
    movement_coefficient = float(config.get('parameters',
                                            'movement_coefficient'))
    initial_bacteria_number = int(config.get('parameters',
                                             'initial_bacteria_number'))
    bacteria_limit = int(config.get('parameters',
                                    'bacteria_limit'))
    seed = int(config.get('parameters', 'seed'))

    # prepare the dictionaries for the data
    parameters_dict = {'t_0': t_0, 'time_limit': time_limit,
                       'x_0': x_0, 'y_0': y_0, 'z_0': z_0,
                       'x_min': x_min, 'y_min': y_min,
                       'z_min': z_min, 'x_max': x_max,
                       'y_max': y_max, 'z_max': z_max,
                       'death_coefficient': death_coefficient,
                       'reproduction_coefficient': reproduction_coefficient,
                       'movement_coefficient': movement_coefficient,
                       'initial_bacteria_number': initial_bacteria_number,
                       'bacteria_limit': bacteria_limit,
                       'seed': seed}
    simulation_dict = {'parameters': parameters_dict, 'result_flag': None,
                       'max_time_reached': None, 'bacteria_final_number': None,
                       'bacteria_list': []}
    print('Parameters collected.')

    print('Starting the simulation...')
    bacteria, flag, max_time = run_simulation(x_0, y_0, z_0, x_min, y_min,
                                              z_min, x_max, y_max, z_max,
                                              t_0, time_limit,
                                              death_coefficient,
                                              reproduction_coefficient,
                                              movement_coefficient,
                                              initial_bacteria_number,
                                              bacteria_limit, seed)
    print('Bacteria total: ', len(bacteria))
    simulation_dict['bacteria_final_number'] = len(bacteria)
    simulation_dict['bacteria_list'] = bacteria
    simulation_dict['result_flag'] = flag
    simulation_dict['max_time_reached'] = max_time

    print('Saving the result...')
    filename = date_name_file('.json')
    results_folder = config.get('folders', 'results_folder')
    save_data_json(simulation_dict, filename, results_folder)
    print('Result saved.')
