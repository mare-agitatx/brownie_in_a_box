import numpy as np
import pytest


def motion_rule(time):
    point = 1.0 * time
    return point


def time_state():
    return [1.0, 2.0, 3.0]


def space_state(time_state):
    output_state = []
    for t in time_state:
        point = motion_rule(t)
        output_state.append(point)
    return output_state


def run_simulation():
    times = time_state()
    x_coordinates = space_state(times)
    y_coordinates = space_state(times)


################################################################################
if __name__ == "__main__":
    run_simulation()


################################################################################
def test_motion_rule_1():
    time = 2.
    expected = 2.
    observed = motion_rule(time)
    assert expected == pytest.approx(observed)


def test_motion_rule_2():
    time = 0.
    expected = 0.
    observed = motion_rule(time)
    assert expected == pytest.approx(observed)


def test_time_state():
    expected = [1.0, 2.0, 3.0]
    observed = time_state()
    assert expected == pytest.approx(observed)


def test_space_state():
    time_state = []
    expected = []
    observed = space_state(time_state)
    assert expected == observed
