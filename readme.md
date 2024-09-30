# Brownie in a box

## Welcome

This code simulates brownian motion in a 2d box for many bacteria that
can die, reproduce or keep moving by that brownian motion. Reproduction means
that the bacterium reproducing will not move that turn and in the same
position a bacterium will be spawned, added to the simulated bacteria and
will be evolved in the next turn. Death means that the bacterium stops
being evolved once the 'death' event is registered. Movement means that a
new position will be calculated and updated accordingly.

The code is split in simulation.py, the script to generate the simulation data,
analysis.py, the script to analyze data precedently generated, and tests.py, a
test suite meant to be run with pytest.

As of now, the documentation and the code itself are a WORK IN PROGRESS!

## Dependencies

numpy, json, sys, enum, datetime, configparser for simulation.py.

numpy, json, sys, matplotlib for analysis.py.

everything listed in simulation.py, analysis.py and also pytest for tests.py.

## Installation

Just clone the repository; one way to do so is running the command:

```
git clone https://github.com/mare-agitatx/brownie_in_a_box.git
```

By doing so in the local folder a new folder is created with the code and
the .git repo files.

## Usage

To run the simulation, you need to specify in a configuration text file the
values of the parameters. In the repo it is included as sim_config.txt but you
are free to run the simulation with any other file with the same internal
structure. When running the script, run it with python and specify in the
command line after the script name the configuration file. The script will
create a json file containing the data generated by simulation.py and will
place it in a subfolder specified in the configuration file.
As an example, let's suppose you wanted to run with sim_config.txt as
configuration file; you could do so with the command:
```
python simulation.py sim_config.txt
```

To run the analysis over the data, run the script analysis.py with python and
specify the json file to be analized in the command line after the script name.
As an example, let's suppose you wanted to analize the file example.json in the
subdirectory results/ contained in the working directory. You could do so with:
```
python analysis.py results/example.json
```

To run the test suite, run tests.py with pytest.
```
pytest tests.py
```
