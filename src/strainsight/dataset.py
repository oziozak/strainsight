"""The frozen synthetic distribution and dev/test seed manifest."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np

from .specimen import dog_bone
from .synthetic import GENERATOR_VERSION, SyntheticSequence, TensileMotion, render_sequence

CaseKind = Literal["uniform", "necking", "smooth"]


@dataclass(frozen=True)
class GeneratorIdentity:
    version: str
    sha256: str


def _generator_sha256(manifest: dict) -> str:
    digest = hashlib.sha256()
    for name in ("dataset.py", "specimen.py", "synthetic.py"):
        digest.update(name.encode() + b"\0")
        digest.update(Path(__file__).with_name(name).read_bytes() + b"\0")
    payload = {key: value for key, value in manifest.items() if key != "generator_sha256"}
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _validated_manifest() -> tuple[dict, GeneratorIdentity]:
    manifest = json.loads(Path(__file__).with_name("synthetic_manifest.json").read_text())
    actual = _generator_sha256(manifest)
    if manifest.get("generator_sha256") != actual or manifest["generator_version"] != GENERATOR_VERSION:
        raise ValueError("frozen generator content hash mismatch; deliberately version and refreeze the generator")
    return manifest, GeneratorIdentity(GENERATOR_VERSION, actual)


def generator_identity() -> GeneratorIdentity:
    """Reject changes to the committed generator freeze before any method runs."""
    return _validated_manifest()[1]


@dataclass(frozen=True)
class SyntheticCase:
    seed: int
    kind: CaseKind
    uniform_strain: float
    band_fwhm: float
    band_center: float
    peak_extra_strain: float
    speckle_size: float
    noise_sigma: float
    blur_sigma: float
    frame_count: int = 32
    score_every: int = 4

    @property
    def scored_frames(self) -> tuple[int, ...]:
        return tuple(range(self.score_every, self.frame_count, self.score_every))

    @property
    def motion(self) -> TensileMotion:
        return TensileMotion(self.kind, self.uniform_strain, self.band_fwhm,
                             self.band_center, self.peak_extra_strain)

    def render(self) -> SyntheticSequence:
        generator_identity()
        return render_sequence(dog_bone(), self.motion, self.frame_count, self.seed,
                               self.speckle_size, self.noise_sigma, self.blur_sigma)


@dataclass(frozen=True)
class FrozenDataset:
    dev: tuple[SyntheticCase, ...]
    test: tuple[SyntheticCase, ...]
    baseline_spacings: tuple[float, ...]
    identity: GeneratorIdentity


def sample_case(seed: int) -> SyntheticCase:
    """Draw from the same PCG64 distribution regardless of split membership."""
    manifest, _ = _validated_manifest()
    return _sample_case(seed, manifest["distribution"])


def _sample_case(seed: int, distribution: dict) -> SyntheticCase:
    rng = np.random.Generator(np.random.PCG64(seed))
    while True:
        kind = rng.choice(distribution["kinds"], p=distribution["kind_probabilities"])
        values = {name: float(rng.uniform(*distribution["uniform_ranges"][name]))
                  for name in distribution["parameter_order"]}
        case = SyntheticCase(seed, kind, values["uniform_strain"], values["band_fwhm"],
                             values["band_center"], values["peak_extra_strain"],
                             values["speckle_size"], values["noise_sigma"], values["blur_sigma"])
        lower, upper = distribution["grip_displacement_range"]
        if lower <= case.motion.grip_displacement <= upper:
            return case


def frozen_dataset() -> FrozenDataset:
    manifest, identity = _validated_manifest()
    dev = tuple(_sample_case(seed, manifest["distribution"]) for seed in manifest["dev_seeds"])
    test = tuple(_sample_case(seed, manifest["distribution"]) for seed in manifest["test_seeds"])
    return FrozenDataset(dev, test, tuple(manifest["baseline_spacings"]), identity)
