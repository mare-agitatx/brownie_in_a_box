import numpy as np
from enum import Enum
import json
from datetime import datetime


RESULTS_FOLDER = 'results/'


def gaussian_distribution(mu, sigma, rng=None):
    '''It computes a gaussian distribution's value.
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
    '''It computes an exponential distribution's value.
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


def brownian_formula(point_at_t_minus_one, dt, rng=None):
    '''A simple formula to implement Brownian motion.
    Parameters:
        point_at_t_minus_one: float, previous position to update.
        dt: float, time interval of the movement.
        rng: numpy.random.Generator, the generator of random numbers if passed
             as input, defaults at None if it isn't passed.
    Returns:
        point_at_t: float, the new position computed.
    Raises:
        ValueError: if dt is negative.
    '''
    if dt < 0.:
        raise ValueError('Given a negative time gap to dt.')

    gaussian_term = gaussian_distribution(0.0, 1.0, rng)
    point_at_t = point_at_t_minus_one + float(np.sqrt(dt)) * gaussian_term
    return point_at_t


def space_state_updater(point_to_update, interval_min, interval_max,
                        dt, rng=None):
    '''Updater function to compute the next position for the bacterium in the
       simulation. The new value is constrained to stay inside the interval
       bounds and borders are reflective: the point will be bounced back
       if it tries to cross the boundary.
    Parameters:
        point_to_update: float, the initial point to be computed.
        interval_min: float, the minimum value that new_point can have.
        interval_max: float, the maximum value that new_point can have.
        dt: float, time interval of the movement.
        rng: numpy.random.Generator, the generator of random numbers if passed
             as input, defaults at None if it isn't passed.
    Returns:
        new_point: float, the new position computed.
    Raises:
        ValueError: if point_to_update is less than interval_min
                    or more than interval_max.
    '''
    if point_to_update < interval_min or point_to_update > interval_max:
        raise ValueError('The given point is out of bounds.')

    new_point = brownian_formula(point_to_update, dt, rng)

    if new_point < interval_min:
        gap = abs(new_point - interval_min)
        reflected_position = interval_min + gap
        new_point = reflected_position

    if new_point > interval_max:
        gap = abs(interval_max - new_point)
        reflected_position = interval_max - gap
        new_point = reflected_position

    return new_point


class Transition(Enum):
    DEATH = 'death'
    REPRODUCTION = 'reproduction'
    MOVEMENT = 'movement'
    BIRTH = 'birth'


def death():
    '''Probability for the death event.
    Returns: float, probability value.
    '''
    return 0.005


def reproduction():
    '''Probability for the reproduction event.
    Returns: float, probability value.
    '''
    return 0.3


def movement():
    '''Probability for the movement event.
    Returns: float, probability value.
    '''
    return 1


def draw_random_event(transition_names, transition_rates, rng=None):
    '''It draws randomly an event string given names and rates.
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
    if rng is None:
        probability = np.random.random_sample()
    else:
        random_number_generator = rng
        probability = random_number_generator.random()
    # drawing the event from the probability
    index = np.searchsorted(normalized_rates, probability)
    event = transition_names[index]
    return event


def run_simulation(x_0, y_0, x_min, x_max,
                   y_min, y_max, t_0, time_limit, seed=None):
    '''The function that runs the main simulation of the bacterium.
    Parameters:
        x_0, y_0: floats, coordinates of the origin
        x_min, x_max: floats, interval limits for the x coordinate
        y_min, y_max: floats, interval limits for the y coordinate
        t_0, time_limit: floats, first instant and last instant
                         of the simulation
        seed: integer if given, will be passed to the default number
              generator of numpy. If not passed will default to None
              and the rng won't be called
    Returns:
        bacterium_states: list of dictionaries, the data generated
                          by the simulation
    Raises:
        ValueError: if an event is drawn randomly that isn't found in
                    transition_names
    '''
    transitions = [death, reproduction, movement]
    transition_names = ['death', 'reproduction', 'movement']

    # initializing the variables
    bacterium_states = []
    total_time = t_0
    x, y = x_0, y_0
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = None

    while total_time < time_limit:
        rates = [f() for f in transitions]
        total_rate = sum(rates)
        dt = exponential_distribution(1/total_rate, rng)
        # drawing the event
        event = draw_random_event(transition_names, rates, rng)

        bacterium = {'time_of_observation': total_time,
                     'interval_of_residency': dt,
                     'x': x, 'y': y,
                     'event': event}
        bacterium_states.append(bacterium)
        total_time += dt

        if event == Transition.REPRODUCTION.value:
            # same coordinates get stored again, the bacterium
            # doesn't move
            continue
            # here goes the code for spawning another bacterium, WIP
        elif event == Transition.DEATH.value:
            # same coordinates get stored again, the bacterium
            # doesn't move
            break
            # here goes the code for removing the bacterium
            # from the total count, WIP
        elif event == Transition.MOVEMENT.value:
            x = space_state_updater(x, x_min, x_max,
                                    dt, rng)
            y = space_state_updater(y, y_min, y_max,
                                    dt, rng)
        else:
            raise ValueError('Unrecognized transition.')
    return bacterium_states


def date_name_file(extension=None):
    '''Blah blah.
    Parameters:

    Returns:

    Raises:

    '''
    if extension is not None:
        if type(extension) != str:
            raise TypeError('The extension given must be a valid string.')

    # datetime object containing current date and time
    now = datetime.now()
    # YYYY_mm_dd_H_M_S
    date_string = now.strftime('%Y_%m_%d_%H_%M_%S')
    if extension is None:
        filename = date_string
    else:
        filename = date_string + extension
    return filename


def save_data_json(data, filename, results_folder=None):
    '''Blah blah.
    Parameters:

    Returns:

    Raises:

    '''
    if results_folder is not None:
        filepath = results_folder + filename
    else:
        filepath = filename
    with open(filepath, 'a') as json_file:
        json.dump(data, json_file, indent=4)


###############################################################################
if __name__ == "__main__":
    # set the simulation parameters
    t_0, time_limit = 0.0, 300.0
    x_0, y_0 = 0.0, 0.0
    x_min, x_max = 0.0, 20.0
    y_min, y_max = -10.0, 10.0
    seed = 400
    parameters_dict = {'t_0': t_0, 'time_limit': time_limit,
                       'x_0': x_0, 'y_0': y_0,
                       'x_min': x_min, 'x_max': x_max,
                       'y_min': y_min, 'y_max': y_max,
                       'seed': seed}
    simulation_dict = {'parameters': parameters_dict, 'bacteria':[]}

    print('Starting the simulation...')
    result = run_simulation(x_0, y_0, x_min, x_max,
                            y_min, y_max, t_0, time_limit, seed)
    print('Simulation done.\nSteps of the simulation =', len(result))
    print('Saving the result...')
    filename = date_name_file('.json')
    simulation_dict['bacteria'] = result
    save_data_json(simulation_dict, filename, RESULTS_FOLDER)
    print('Result saved.')
