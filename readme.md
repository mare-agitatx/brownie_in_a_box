# Brownie in a box

## Welcome

This code simulates brownian motion in a 2d box for a single bacterium that
can die, reproduce (which now means just stopping there for a simulation turn)
or keep moving by that brownian motion. The results are then plotted

As of now, the documentation and the code itself are a WORK IN PROGRESS

## Dependencies

numpy, enum, json, datetime for simulation.py

json, matplotlib for analysis.py

everything listed simulation.py and analysis.py, and then pytest for tests.py

## Installation

Just clone the repository; one way to do so is running the command:

```
git clone https://github.com/mare-agitatx/brownie_in_a_box.git
```

By doing so in the local folder a new folder is created with the code and
the .git repo files

## Usage

To run the simulation, open the script simulation.py and specify the folder
where the data will be saved in the RESULTS_FOLDER variable at the top of
the file, then run it with python. The script will create a json file
containing the data generated by the simulation.
To run the analysis over the data, open the script analysis.py  and specify
the filepath to the json file that must be analized in the FILEPATH variable at
the top of the file, then run it with python.
To run the test suite, run tests.py with pytest.
The commands to run the scripts are the following:

```
python simulation.py
python analysis.py
pytest tests.py
```
