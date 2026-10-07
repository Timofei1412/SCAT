from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
import copy
import random
import numpy as np

import SCAT.AERO.main as main


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    low: float
    high: float
    baseline: float


def build_parameter_specs_for_config(config: main.AircraftConfig) -> List[ParameterSpec]:
    """
    Построить список спецификаций параметров для всех поверхностей (кроме фюзеляжа).
    Для каждой секции поверхности добавляем: x, y, z, chord, twist.
    """
    specs: List[ParameterSpec] = []

    surfaces = [
        ("wing", config.wing),
        ("canard", config.canard),
        ("htail", config.htail),
        ("vtail", config.vtail),
    ]

    for surf_name, surface in surfaces:
        if not surface.enabled:
            continue
        for idx, sec in enumerate(surface.sections):
            base_x = sec.x
            base_y = sec.y
            base_z = sec.z
            base_chord = sec.chord
            base_twist = sec.twist

            # shifts: allow moderate absolute shifts around baseline
            specs.append(ParameterSpec(f"{surf_name}_s{idx}_x", base_x - 0.8, base_x + 0.8, base_x))
            specs.append(ParameterSpec(f"{surf_name}_s{idx}_y", base_y - 0.8, base_y + 0.8, base_y))
            specs.append(ParameterSpec(f"{surf_name}_s{idx}_z", base_z - 0.5, base_z + 0.5, base_z))

            # chord: relative scaling but expressed as absolute chord value ranges
            min_chord = max(0.05, base_chord * 0.4)
            max_chord = max(0.06, base_chord * 2.0)
            specs.append(ParameterSpec(f"{surf_name}_s{idx}_chord", min_chord, max_chord, base_chord))

            # twist: allow +/- 15 deg around baseline
            specs.append(ParameterSpec(f"{surf_name}_s{idx}_twist", base_twist - 15.0, base_twist + 15.0, base_twist))

    return specs


def baseline_vector_from_specs(specs: List[ParameterSpec]) -> np.ndarray:
    return np.array([spec.baseline for spec in specs], dtype=float)


def vector_to_dict(specs: List[ParameterSpec], vector: np.ndarray) -> Dict[str, float]:
    return {specs[i].name: float(vector[i]) for i in range(len(specs))}


def clamp_vector(specs: List[ParameterSpec], vector: np.ndarray) -> np.ndarray:
    clamped = vector.copy()
    for i, spec in enumerate(specs):
        clamped[i] = float(np.clip(clamped[i], spec.low, spec.high))
    return clamped


def sample_random_vector(specs: List[ParameterSpec], rng: random.Random) -> np.ndarray:
    return np.array([rng.uniform(spec.low, spec.high) for spec in specs], dtype=float)


def apply_vector_to_config(base_config: main.AircraftConfig, specs: List[ParameterSpec], vector: np.ndarray) -> main.AircraftConfig:
    """
    Применить вектор параметров к копии конфигурации и вернуть новый конфиг.
    Поддерживает только поверхности (wing, canard, htail, vtail); фюзеляж не меняется.
    """
    cfg = copy.deepcopy(base_config)
    vec = clamp_vector(specs, vector)
    name_to_value = {spec.name: float(vec[i]) for i, spec in enumerate(specs)}

    surfaces = {
        "wing": cfg.wing,
        "canard": cfg.canard,
        "htail": cfg.htail,
        "vtail": cfg.vtail,
    }

    for surf_name, surface in surfaces.items():
        if not surface.enabled:
            continue
        for idx, sec in enumerate(surface.sections):
            key_x = f"{surf_name}_s{idx}_x"
            key_y = f"{surf_name}_s{idx}_y"
            key_z = f"{surf_name}_s{idx}_z"
            key_chord = f"{surf_name}_s{idx}_chord"
            key_twist = f"{surf_name}_s{idx}_twist"

            if key_x in name_to_value:
                sec.x = name_to_value[key_x]
            if key_y in name_to_value:
                sec.y = name_to_value[key_y]
            if key_z in name_to_value:
                sec.z = name_to_value[key_z]
            if key_chord in name_to_value:
                sec.chord = name_to_value[key_chord]
            if key_twist in name_to_value:
                sec.twist = name_to_value[key_twist]

    return cfg
