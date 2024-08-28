import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import typing
from enum import Enum


def brownian_formula(point_at_t_minus_one, dt, gaussian_term):
    '''A simple formula to implement Brownian motion.
    Parameters:
    Returns:
    Raises:
    '''
    if dt < 0.:
        raise ValueError('Given a negative time gap to dt.')

    point_at_t = point_at_t_minus_one + float(np.sqrt(dt)) * gaussian_term
    return point_at_t


def gaussian_distribution(mu, sigma, rng=None):
    '''It computes a gaussian distribution's value.
    Parameters:
    Returns:
    '''
    if rng is None:
        gaussian = np.random.normal
    else:
        random_number_generator = rng
        gaussian = random_number_generator.normal

    return gaussian(mu, sigma)

def exponential_distribution(beta, rng=None):
    '''It computes an exponential distribution's value.
    Parameters:
    Returns:
    '''
    if rng is None:
        output = np.random.exponential(beta)
    else:
        random_number_generator = rng
        output = random_number_generator.exponential(beta)

    return output


def space_state_updater(point_to_update, interval_min, interval_max,
                        dt, *gaussian_parameters):
    '''Something something.
    Parameters:
    Returns:
    '''
    if point_to_update < interval_min or point_to_update > interval_max:
        raise ValueError('The given point is out of bounds.')

    gaussian_term = gaussian_distribution(*gaussian_parameters)
    new_point = brownian_formula(point_to_update, dt, gaussian_term)

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


class Bacterium(typing.NamedTuple):
    x: float
    y: float
    time_of_observation: float
    time_of_residency: float
    transition: Transition


def death():
    return 0.005


def reproduction():
    return 0.3


def movement():
    return 1


def draw_random_event(transitions_names, rates, rng=None):
    if len(transitions_names) != len(rates):
        raise ValueError('Names and rates for the transitions ' +
                         'must be equal in number.')

    normalized_rates = np.cumsum(rates)
    normalized_rates /= normalized_rates[-1]
    if rng is None:
        p = np.random.random_sample()
    else:
        random_number_generator = rng
        p = random_number_generator.random()
    idx = np.searchsorted(normalized_rates, p)
    event = transitions_names[idx]
    return event


def run_simulation(x_0, y_0, x_min, x_max,
                   y_min, y_max, t_0, time_limit, seed=None):
    transitions = [death, reproduction, movement]
    transitions_names = ['death', 'reproduction', 'movement']

    # initializing the variables
    bacterium_states = []
    total_time = t_0
    x, y = x_0, y_0
    mu, sigma = 0.0, 1.0
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = None
    gaussian_parameters = (mu, sigma, rng)

    while total_time < time_limit:
        rates = [f() for f in transitions]
        total_rate = sum(rates)
        dt = exponential_distribution(1/total_rate, rng)
        # drawing the event
        event = draw_random_event(transitions_names, rates, rng)

        bacterium = Bacterium(x, y, total_time, dt, event)
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
                                    dt, *gaussian_parameters)
            y = space_state_updater(y, y_min, y_max,
                                    dt, *gaussian_parameters)
        else:
            raise ValueError('Unrecognized transition.')
    return bacterium_states


def plots(times, x_coordinates, y_coordinates, x_min, x_max, y_min, y_max):

    fig1, ax1 = plt.subplots()
    ax1.set_title('Time evolution of the coordinates')
    ax1.plot(times, x_coordinates, label='x coordinate', color='orange')
    ax1.plot(times, y_coordinates, label='y coordinate', color='cyan')
    ax1.hlines([x_min, x_max], times[0], times[-1],
                label='x boundaries', color='orange', linestyle='dotted')
    ax1.hlines([y_min, y_max],  times[0], times[-1],
                label='y boundaries', color='cyan', linestyle='dotted')
    ax1.legend(loc='best')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Space')

    fig2, ax2 = plt.subplots()
    ax2.set_title('Trajectory on the xy plane')
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


    plt.show()


###############################################################################
if __name__ == "__main__":
    t_0, time_limit = 0.0, 300.0
    x_0, y_0 = 0.0, 0.0
    x_min, x_max = 0.0, 20.0
    y_min, y_max = -10.0, 10.0
    result = run_simulation(x_0, y_0, x_min, x_max,
                            y_min, y_max, t_0, time_limit)
    print('Simulation done.\nSteps of the simulation =', len(result))

    print('Unpacking the space and time values into lists...')
    x_coordinates, y_coordinates, times = [], [], []
    for state in result:
        x_coordinates.append(state.x)
        y_coordinates.append(state.y)
        times.append(state.time_of_observation)

    print('Plotting the lists.')
    plots(times, x_coordinates, y_coordinates, x_min, x_max, y_min, y_max)



