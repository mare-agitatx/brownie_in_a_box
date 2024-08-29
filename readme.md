# Brownie in a box

## Welcome

This code simulates brownian motion in a 2d box for a single bacterium that
can die, reproduce (which now means just stopping there for a simulation turn)
or keep moving by that brownian motion. The results are then plotted

As of now, the documentation and the code itself are a WORK IN PROGRESS

## Dependencies

numpy, matplotlib for brownie_box.py

everything listed for brownie_box.py and pytest for tests.py

## Installation

Just clone the repository; one way to do so is running the command:

```
git clone https://github.com/mare-agitatx/brownie_in_a_box.git
```

By doing so in the local folder a new folder is created with the code and
the .git repo files

## Usage

To run the simulation, run the script brownie_box.py with python. To run the
test suite, run tests.py with pytest; this is achieved with the commands:

```
python brownie_in_a_box.py
pytest tests.py
```
