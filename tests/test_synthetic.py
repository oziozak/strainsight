import numpy as np
import pytest

from strainsight.dataset import frozen_dataset, generator_identity, sample_case
from strainsight.specimen import dog_bone
from strainsight.synthetic import TensileMotion, render_sequence


def test_frozen_dataset_has_disjoint_splits_from_one_distribution():
    dataset = frozen_dataset()
    assert len(dataset.dev) == 8
    assert len(dataset.test) == 24
    assert not {case.seed for case in dataset.dev} & {case.seed for case in dataset.test}
    for split in (dataset.dev, dataset.test):
        assert {case.kind for case in split} == {"uniform", "necking", "smooth"}
        for case in split:
            assert 16 <= case.band_fwhm <= 40
            assert case.band_fwhm >= 2 * max(dataset.baseline_spacings)
            assert 0.08 <= case.uniform_strain <= 0.12
            assert 0.2 <= case.peak_extra_strain <= 0.6
            assert 1.5 <= case.speckle_size <= 5
            assert case.frame_count == 32
            assert case.scored_frames == (4, 8, 12, 16, 20, 24, 28)
            assert 12 <= case.motion.grip_displacement <= 28
            assert sample_case(case.seed) == case
    assert frozen_dataset() == dataset


def test_necking_band_has_the_specified_green_lagrange_peak_and_fwhm():
    motion = TensileMotion("necking", uniform_strain=0.1, band_fwhm=16,
                           band_center=240, peak_extra_strain=0.4)
    points = np.array([[232., 192.], [240., 192.], [248., 192.]])
    np.testing.assert_allclose(motion.strain_at(points, 1)[:, 0, 0], [0.3, 0.5, 0.3], atol=1e-12)
    np.testing.assert_allclose(motion.forward(points, 0), points, atol=1e-12)
    np.testing.assert_allclose(motion.strain_at(points, 0), 0, atol=1e-12)


@pytest.mark.parametrize("kind", ["uniform", "necking", "smooth"])
def test_deformation_is_invertible_incompressible_and_matches_spatial_truth(kind):
    motion = TensileMotion(kind, 0.1, 24, 280, 0.4)
    points = np.array([[66., 192.], [446., 192.], [270.123, 212.], [280., 192.]])
    for time in (0., 0.5, 1.):
        deformed = motion.forward(points, time)
        np.testing.assert_allclose(motion.inverse(deformed, time), points, atol=1e-8)
        gradient = motion.deformation_gradient(points, time)
        # Equal contraction in y and z gives volume preservation in 3D.
        np.testing.assert_allclose(np.linalg.det(gradient) * gradient[:, 1, 1], 1, atol=1e-12)
        epsilon = 1e-3
        numerical = np.stack([(motion.forward(points + delta, time) -
                                motion.forward(points - delta, time)) / (2 * epsilon)
                               for delta in (np.array([epsilon, 0]), np.array([0, epsilon]))], axis=-1)
        np.testing.assert_allclose(gradient, numerical, atol=1e-8)
    ends = motion.forward(points[:2], 1)[:, 0] - points[:2, 0]
    np.testing.assert_allclose(ends[0], -ends[1], atol=1e-10)
    assert 12 <= ends[1] <= 28


def test_rendering_is_reproducible_and_emits_dense_reference_ground_truth():
    specimen = dog_bone()
    motion = TensileMotion("necking", 0.1, 16, 240, 0.4)
    sequence = render_sequence(specimen, motion, 2, seed=17, speckle_size=1.5,
                               noise_sigma=2, blur_sigma=0.5)
    repeated = render_sequence(specimen, motion, 2, seed=17, speckle_size=1.5,
                               noise_sigma=2, blur_sigma=0.5)
    np.testing.assert_array_equal(sequence.frames, repeated.frames)
    assert sequence.frames.shape == (2, 384, 512)
    assert sequence.frames.dtype == np.uint8
    assert sequence.max_inversion_residual <= 0.01
    assert np.std(sequence.frames[0][specimen.specimen_mask]) > 10
    assert not np.array_equal(sequence.frames[0], sequence.frames[1])
    truth = sequence.ground_truth(1)
    assert truth.displacement.shape == (384, 512, 2)
    assert truth.strain.shape == (384, 512, 2, 2)
    assert np.isnan(truth.strain[~specimen.specimen_mask]).all()
    np.testing.assert_allclose(truth.strain[192, [232, 240, 248], 0, 0], [0.3, 0.5, 0.3], atol=1e-12)
    np.testing.assert_allclose(sequence.ground_truth(0).displacement[specimen.specimen_mask], 0, atol=1e-12)
    with pytest.raises(ValueError, match="frame"):
        sequence.ground_truth(2)


def test_frozen_generator_identity_is_a_content_hash_and_detects_changed_sources(monkeypatch):
    from pathlib import Path

    identity = generator_identity()
    assert identity.version == "tensile-v2"
    assert len(identity.sha256) == 64
    int(identity.sha256, 16)
    original_read = Path.read_bytes

    def changed_read(path):
        data = original_read(path)
        return data + b"\n# changed generator\n" if path.name == "synthetic.py" else data

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    with pytest.raises(ValueError, match="frozen generator content hash"):
        generator_identity()


def test_manifest_json_key_order_does_not_change_frozen_draws(monkeypatch):
    import json
    from pathlib import Path

    expected = frozen_dataset()
    original_read = Path.read_text

    def sorted_json(path, *args, **kwargs):
        data = original_read(path, *args, **kwargs)
        if path.name == "synthetic_manifest.json":
            return json.dumps(json.loads(data), sort_keys=True)
        return data

    monkeypatch.setattr(Path, "read_text", sorted_json)
    assert frozen_dataset() == expected


@pytest.mark.parametrize("speckle_size", [1.5, 5.0])
def test_optical_blur_and_framewise_sensor_noise_affect_the_rendered_images(speckle_size):
    specimen = dog_bone()
    motion = TensileMotion("uniform")
    clean = render_sequence(specimen, motion, 2, 23, speckle_size)
    blurred = render_sequence(specimen, motion, 2, 23, speckle_size, blur_sigma=0.8)
    noisy = render_sequence(specimen, motion, 2, 23, speckle_size, noise_sigma=3)
    sharp_gauge = clean.frames[0, 170:214, 140:370].astype(float)
    blurred_gauge = blurred.frames[0, 170:214, 140:370].astype(float)
    assert np.var(np.diff(blurred_gauge, axis=1)) < np.var(np.diff(sharp_gauge, axis=1))
    # The background is a constant 24; noise sigma is in 8-bit grey levels.
    noise = noisy.frames[:, :100].astype(float) - clean.frames[:, :100]
    assert 2.9 < noise.std() < 3.1
    assert abs(noise.mean()) < 0.05
    assert not np.array_equal(noise[0], noise[1])
