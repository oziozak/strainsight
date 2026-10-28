"""Synthetic sanity check for masked initialization and axial strain."""
import numpy as np
from .sampling import sample_masked_grid
from .strain import axial_engineering_strain


def main() -> None:
    mask = np.zeros((80, 180), dtype=bool)
    mask[15:65, 20:160] = True
    pts = sample_masked_grid(mask, budget=100)
    left, right = np.array([[30., 40.], [150., 40.]])
    stretch = 1.08
    transformed = np.array([[30., 40.], [30. + (150. - 30.) * stretch, 40.]])
    print(f"Sampled {len(pts)} points in specimen mask")
    print(f"Known axial strain: {stretch - 1:.3f}; estimated: {axial_engineering_strain(np.array([left, right]), transformed):.3f}")


if __name__ == "__main__":
    main()
