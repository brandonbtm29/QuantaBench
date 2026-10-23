import numpy as np

from quantabench.gui.selection import SelectionModel, measure


def test_measure():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], float)
    assert np.isclose(measure(p, [0, 1])[1], 1.0)
    assert np.isclose(measure(p, [0, 1, 2])[1], 90.0)
    assert np.isclose(abs(measure(p, [0, 1, 2, 3])[1]), 90.0)
    assert measure(p, [0]) is None


def test_selection_signal():
    s = SelectionModel()
    seen = []
    s.changed.connect(seen.append)
    s.select([3, 1])
    s.toggle(3)
    s.select([5], additive=True)
    s.clear()
    assert seen == [[3, 1], [1], [1, 5], []]
