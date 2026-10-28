import numpy as np
import pytest
from strainsight.sampling import sample_masked_grid
from strainsight.strain import axial_engineering_strain


def test_sample_points_are_inside_mask():
    mask = np.zeros((20, 30), dtype=bool)
    mask[4:15, 7:25] = True
    points = sample_masked_grid(mask, 25)
    assert len(points) == 25
    assert all(mask[int(y), int(x)] for x, y in points)
    assert len(set(map(tuple, points))) == 25


def test_strain():
    original = np.array([[0., 0.], [100., 0.]])
    deformed = np.array([[10., 0.], [120., 0.]])
    assert axial_engineering_strain(original, deformed) == pytest.approx(0.1)


def test_empty_mask():
    with pytest.raises(ValueError):
        sample_masked_grid(np.zeros((4, 4)), 2)
