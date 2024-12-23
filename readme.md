# Brownie in a box

## Welcome

This code simulates brownian motion in a 3D box for many bacteria that
can die, reproduce or keep moving by that brownian motion. Reproduction means
that the bacterium reproducing will not move that turn and in the same
position a new bacterium will be spawned, added to the simulated bacteria and
will be evolved in the next turn. Death means that the bacterium stops
being evolved once the 'death' event is registered. Movement means that a
new position will be calculated and updated accordingly by drawing randomly
a direction in a spherical reference and then drawing the jump distance by
the brownian formula. The simulation is in the form of a continuous time
random walk, so duration of events for the bacteria is also randomly drawn
by means of an exponential distribution.

The code is split in br_simulation.py, the script to generate the simulation
data, br_analysis.py, a module with functions employed for analysis, 
br_distributions.py which analyzes the distributions of the data by means of 
histograms and pdfs for reference, br_3dplot.py to plot the
spatial configurations from precedently generated data in three dimensions
and tests.py, a test suite meant to be run with pytest.

## Dependencies

For the totality of the scripts, numpy, scipy, matplotlib, json, sys, enum,
datetime, configparser and pytest are required. Below are listed the specific
requirements for every script.

numpy, json, sys, enum, datetime, configparser for br_simulation.py.

numpy, scipy, json, sys, matplotlib for br_analysis.py.

everything listed for br_analysis.py implicitly and also explicitly
numpy, scipy, json, sys, matplotlib for br_distributions.py

everything listed for br_analysis.py implicitly and also explicitly
numpy, sys, matplotlib for br_3dplot.py

everything listed for br_simulation.py and br_analysis.py implicitly and also
explicitly pytest for tests.py.

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
structure and if you wanted to alter the parameters you could just take
sim_config.txt, change the values are needed and even rename it if desired
(but then remember to use the new name in the command line).
When running the script, run it with python and specify in the
command line after the script name the configuration file. The script will
create a json file containing the data generated by br_simulation.py and will
place it in a subfolder specified in the configuration file.
As an example, let's suppose you wanted to run with sim_config.txt as
configuration file; you could do so with the command:
```
python br_simulation.py sim_config.txt
```

To view the distributions over the data, run br_distributions.py with python and
specify the json file to be analized in the command line after the script name.
As an example, let's suppose you wanted to analize the file example.json in the
subdirectory results/ contained in the working directory.
You could do so with:
```
python br_distributions.py results/example.json
```

To plot the 3d scatter plots, run the script br_3dplots.py with python and
specify the json file to be analized in the command line after the script name.
With a similar setting from the example above for analysis.py:
```
python br_3dplot.py results/example.json
```

To run the test suite, run tests.py with pytest.
```
pytest tests.py
```
