import br_analysis as bra
import numpy as np
import matplotlib.pyplot as plt
import sys


def counters_at_some_time(bacteria_list, time_of_interest):
    '''
    Calculates two spatial distributions for the given bacteria input,
    distinguishing between dead bacteria and live bacteria.
    Parameters:
        bacteria_list: list of dictionaries, the entries represent bacteria.
        time_of_interest: the time at which the distributions are evaluated.
    Returns:
        output_dict: dictionary, has four entries that are lists with two
                     lists for x and y of live bacteria and two other lists
                     for coordinates of dead bacteria.
    '''
    move_count, repro_count, death_count = 0, 0, 0
    for bacterium_dict in bacteria_list:
        x, y, z, event = bra.space_value_at_some_time(bacterium_dict,
                                                      time_of_interest)
        if event is None:
            continue
        elif event == 'birth':
            continue # we count births already in reproductions
        elif event == 'death':
            death_count += 1
        elif event == 'reproduction':
            repro_count += 1
        elif event == 'movement':
            move_count += 1

    return move_count, repro_count, death_count


def bacteria_counters_evolution(bacteria_list, t_0, t_f, n_steps=1000):
    movements, deaths, reproductions = [], [], []
    times = np.linspace(t_0, t_f, n_steps).astype(float)
    counts_at_t = counters_at_some_time 
    
    for time in times:
        move_count, repro_count, death_count = counts_at_t(bacteria_list, time)
        movements.append(move_count)
        deaths.append(death_count)
        reproductions.append(repro_count)
        
    return times, movements, deaths, reproductions
    
def bacteria_number_evolution(bacteria_list, t_0, t_f, n_steps=1000):
    alive_bacteria = []
    times = np.linspace(t_0, t_f, n_steps).astype(float)
    
    space_distrib = bra.space_distribution_at_some_time
    for time in times:
        live_xyz, dead_xyz = space_distrib(bacteria_list, time)
        alive_bact_count = len(live_xyz)
        alive_bacteria.append(alive_bact_count)
    
    return alive_bacteria
    
def time_graphs_routine(times, movements, deaths, reproductions, numbers):
    plt.figure(1)
    ax = plt.axes()
    plt.title('Deaths')
    plt.ylabel('Number')
    plt.xlabel('Time')
    plt.plot(times, deaths)
    
    plt.figure(2)
    ax = plt.axes()
    plt.title('Reproductions')
    plt.ylabel('Number')
    plt.xlabel('Time')
    plt.plot(times, reproductions)
    
    plt.figure(3)
    ax = plt.axes()
    plt.title('Movements')
    plt.ylabel('Number')
    plt.xlabel('Time')
    plt.plot(times, movements)
    
    plt.figure(4)
    ax = plt.axes()
    plt.title('Number of alive bacteria')
    plt.ylabel('Number')
    plt.xlabel('Time')
    plt.plot(times, numbers)
    
    plt.figure(5)
    ax = plt.axes()
    plt.title('Events over time')
    plt.ylabel('Number')
    plt.xlabel('Time')
    plt.plot(times, reproductions, label='Reproductions')
    plt.plot(times, movements, label='Movements')
    plt.plot(times, deaths, label='Deaths')
    plt.legend()
    
    plt.figure(6)
    ax = plt.axes()
    plt.title('Events over time and alive bacteria')
    plt.ylabel('Number')
    plt.xlabel('Time')
    plt.plot(times, reproductions, label='Reproductions')
    plt.plot(times, movements, label='Movements')
    plt.plot(times, deaths, label='Deaths')
    plt.plot(times, numbers, label='Alive bacteria')
    plt.legend()
    
    plt.show()
    
    
################################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('No input given. Exiting...')
        print('Please input the data filename on the command line.')
        sys.exit()
    
    print('Loading the file...')
    filepath = sys.argv[1]
    analysis_data = bra.load_data_json(filepath)
    print('File loaded.')
    
    print('Parsing the data...')
    parameters_dict, bacteria_list = bra.parse_data_dict(analysis_data)
    if parameters_dict['result_flag'] != 'halted_simulation':
        t_f = parameters_dict['time_limit']
    else:
        t_f = parameters_dict['max_time_reached']
    t_0 = parameters_dict['t_0']
    
    bact_counters = bacteria_counters_evolution
    bact_numbers = bacteria_number_evolution
    print('Plotting the data...')
    times, movements, deaths, reproductions = bact_counters(bacteria_list, 
                                                            t_0, t_f)
    numbers = bact_numbers(bacteria_list, t_0, t_f)
    time_graphs_routine(times, movements, deaths, reproductions, numbers)
