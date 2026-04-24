from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import SCAT.AERO.main as main
import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    low: float
    high: float
    baseline: float


PARAMETER_SPECS = [
    ParameterSpec("wing_x_shift", -0.8, 0.8, 0.0),
    ParameterSpec("wing_z_shift", -0.35, 0.35, 0.0),
    ParameterSpec("wing_span_scale", 0.70, 1.50, 1.0),
    ParameterSpec("wing_sweep_scale", 0.60, 1.60, 1.0),
    ParameterSpec("wing_dihedral_scale", 0.40, 1.80, 1.0),
    ParameterSpec("wing_chord_scale", 0.65, 1.45, 1.0),
    ParameterSpec("wing_root_chord_scale", 0.70, 1.40, 1.0),
    ParameterSpec("wing_mid_chord_scale", 0.60, 1.50, 1.0),
    ParameterSpec("wing_tip_chord_scale", 0.50, 1.60, 1.0),
    ParameterSpec("wing_twist_bias", -4.0, 4.0, 0.0),
    ParameterSpec("wing_twist_scale", 0.50, 1.50, 1.0),
    ParameterSpec("canard_x_shift", -1.0, 1.0, 0.0),
    ParameterSpec("canard_y_shift", -0.20, 0.40, 0.0),
    ParameterSpec("canard_z_shift", -0.25, 0.25, 0.0),
    ParameterSpec("canard_span_scale", 0.60, 1.60, 1.0),
    ParameterSpec("canard_sweep_scale", 0.60, 1.60, 1.0),
    ParameterSpec("canard_chord_global_scale", 0.60, 1.60, 1.0),
    ParameterSpec("canard_chord_scale", 0.60, 1.60, 1.0),
    ParameterSpec("canard_twist_bias", -5.0, 5.0, 0.0),
    ParameterSpec("canard_twist_scale", 0.50, 1.60, 1.0),
    ParameterSpec("htail_x_shift", -1.0, 1.0, 0.0),
    ParameterSpec("htail_z_shift", -0.30, 0.30, 0.0),
    ParameterSpec("htail_span_scale", 0.60, 1.60, 1.0),
    ParameterSpec("htail_sweep_scale", 0.60, 1.60, 1.0),
    ParameterSpec("htail_chord_global_scale", 0.60, 1.60, 1.0),
    ParameterSpec("htail_chord_scale", 0.60, 1.60, 1.0),
    ParameterSpec("htail_twist_bias", -5.0, 5.0, 0.0),
    ParameterSpec("htail_twist_scale", 0.50, 1.60, 1.0),
    ParameterSpec("vtail_x_shift", -1.0, 1.0, 0.0),
    ParameterSpec("vtail_z_shift", -0.40, 0.40, 0.0),
    ParameterSpec("vtail_height_scale", 0.60, 1.60, 1.0),
    ParameterSpec("vtail_sweep_scale", 0.60, 1.60, 1.0),
    ParameterSpec("vtail_chord_global_scale", 0.60, 1.60, 1.0),
    ParameterSpec("vtail_chord_scale", 0.60, 1.60, 1.0),
]

PARAMETER_INDEX = {spec.name: index for index, spec in enumerate(PARAMETER_SPECS)}


def baseline_vector() -> np.ndarray:
    return np.array([spec.baseline for spec in PARAMETER_SPECS], dtype=float)


def vector_to_dict(vector: np.ndarray) -> dict[str, float]:
    return {spec.name: float(vector[index]) for index, spec in enumerate(PARAMETER_SPECS)}


def clamp_vector(vector: np.ndarray) -> np.ndarray:
    clamped = vector.copy()
    for index, spec in enumerate(PARAMETER_SPECS):
        clamped[index] = np.clip(clamped[index], spec.low, spec.high)
    return clamped


def load_config(config_path: str | None) -> main.AircraftConfig:
    if config_path is None:
        return main.default_config()
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return main.config_from_dict(data)


def save_config(config: main.AircraftConfig, output_path: str) -> None:
    Path(output_path).write_text(
        json.dumps(main.config_to_dict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def fuselage_length(config: main.AircraftConfig) -> float:
    xs = [section.x for section in config.fuselage_sections]
    return max(xs) - min(xs) if xs else 0.0


def ensure_surface_ready(surface: main.SurfaceConfig, default_name: str) -> None:
    surface.enabled = True
    if not surface.name.strip():
        surface.name = default_name
    if len(surface.sections) < 2:
        raise ValueError(f"Поверхность '{default_name}' должна иметь минимум 2 секции.")


def transform_surface_sections(
    sections: list[main.SurfaceSection],
    x_shift: float,
    y_shift: float,
    span_scale: float,
    sweep_scale: float,
    z_shift: float,
    z_scale: float,
    global_chord_scale: float,
    chord_scales: list[float],
    twist_bias: float,
    twist_scale: float,
) -> list[main.SurfaceSection]:
    transformed = []
    root_x = sections[0].x
    root_y = sections[0].y
    root_z = sections[0].z

    for index, section in enumerate(sections):
        local_chord_scale = chord_scales[min(index, len(chord_scales) - 1)]
        transformed.append(
            main.SurfaceSection(
                x=root_x + x_shift + (section.x - root_x) * sweep_scale,
                y=root_y + y_shift + (section.y - root_y) * span_scale,
                z=root_z + z_shift + (section.z - root_z) * z_scale,
                chord=max(0.08, section.chord * global_chord_scale * local_chord_scale),
                twist=section.twist * twist_scale + twist_bias,
            )
        )
    return transformed


def apply_design_vector(base_config: main.AircraftConfig, vector: np.ndarray) -> main.AircraftConfig:
    config = copy.deepcopy(base_config)
    vector = clamp_vector(vector)
    params = vector_to_dict(vector)

    ensure_surface_ready(config.wing, "Main Wing")
    ensure_surface_ready(config.canard, "Canard")

    config.wing.sections = transform_surface_sections(
        sections=config.wing.sections,
        x_shift=params["wing_x_shift"],
        y_shift=0.0,
        span_scale=params["wing_span_scale"],
        sweep_scale=params["wing_sweep_scale"],
        z_shift=params["wing_z_shift"],
        z_scale=params["wing_dihedral_scale"],
        global_chord_scale=params["wing_chord_scale"],
        chord_scales=[
            params["wing_root_chord_scale"],
            params["wing_mid_chord_scale"],
            params["wing_tip_chord_scale"],
        ],
        twist_bias=params["wing_twist_bias"],
        twist_scale=params["wing_twist_scale"],
    )

    config.canard.sections = transform_surface_sections(
        sections=config.canard.sections,
        x_shift=params["canard_x_shift"],
        y_shift=params["canard_y_shift"],
        span_scale=params["canard_span_scale"],
        sweep_scale=params["canard_sweep_scale"],
        z_shift=params["canard_z_shift"],
        z_scale=1.0,
        global_chord_scale=params["canard_chord_global_scale"],
        chord_scales=[params["canard_chord_scale"]] * len(config.canard.sections),
        twist_bias=params["canard_twist_bias"],
        twist_scale=params["canard_twist_scale"],
    )

    if config.htail.enabled and len(config.htail.sections) >= 2:
        ensure_surface_ready(config.htail, "Horizontal Tail")
        config.htail.sections = transform_surface_sections(
            sections=config.htail.sections,
            x_shift=params["htail_x_shift"],
            y_shift=0.0,
            span_scale=params["htail_span_scale"],
            sweep_scale=params["htail_sweep_scale"],
            z_shift=params["htail_z_shift"],
            z_scale=1.0,
            global_chord_scale=params["htail_chord_global_scale"],
            chord_scales=[params["htail_chord_scale"]] * len(config.htail.sections),
            twist_bias=params["htail_twist_bias"],
            twist_scale=params["htail_twist_scale"],
        )

    if config.vtail.enabled and len(config.vtail.sections) >= 2:
        ensure_surface_ready(config.vtail, "Vertical Tail")
        config.vtail.sections = transform_surface_sections(
            sections=config.vtail.sections,
            x_shift=params["vtail_x_shift"],
            y_shift=0.0,
            span_scale=1.0,
            sweep_scale=params["vtail_sweep_scale"],
            z_shift=params["vtail_z_shift"],
            z_scale=params["vtail_height_scale"],
            global_chord_scale=params["vtail_chord_global_scale"],
            chord_scales=[params["vtail_chord_scale"]] * len(config.vtail.sections),
            twist_bias=0.0,
            twist_scale=1.0,
        )

    return config


def geometry_penalty(config: main.AircraftConfig) -> float:
    penalty = 0.0
    for surface in (config.wing, config.canard, config.htail):
        if not surface.enabled:
            continue
        ys = [section.y for section in surface.sections]
        chords = [section.chord for section in surface.sections]

        for left, right in zip(ys, ys[1:]):
            if right <= left:
                penalty += 15.0 + abs(right - left) * 20.0
        for left, right in zip(chords, chords[1:]):
            if right > left * 1.35:
                penalty += 5.0 + (right - left) * 4.0

    wing_tip = config.wing.sections[-1]
    canard_tip = config.canard.sections[-1]
    if wing_tip.y <= canard_tip.y:
        penalty += 10.0
    if config.htail.enabled and config.htail.sections[-1].y <= 0.4:
        penalty += 5.0
    if config.vtail.enabled:
        zs = [section.z for section in config.vtail.sections]
        if any(right <= left for left, right in zip(zs, zs[1:])):
            penalty += 10.0
    return penalty


def surface_half_span(surface: main.SurfaceConfig) -> float:
    ys = [section.y for section in surface.sections]
    return max(ys) - min(ys) if ys else 0.0


def surface_planform_area(surface: main.SurfaceConfig) -> float:
    if len(surface.sections) < 2:
        return 0.0
    area = 0.0
    for left, right in zip(surface.sections[:-1], surface.sections[1:]):
        dy = abs(right.y - left.y)
        area += 0.5 * (left.chord + right.chord) * dy
    return area * (2.0 if surface.symmetric else 1.0)


def surface_mean_chord(surface: main.SurfaceConfig) -> float:
    area = surface_planform_area(surface)
    span = surface_half_span(surface) * (2.0 if surface.symmetric else 1.0)
    if span <= 1e-9:
        return 0.0
    return area / span


def surface_aspect_ratio(surface: main.SurfaceConfig) -> float:
    area = surface_planform_area(surface)
    full_span = surface_half_span(surface) * (2.0 if surface.symmetric else 1.0)
    if area <= 1e-9:
        return 0.0
    return full_span**2 / area


def surface_metrics(surface: main.SurfaceConfig) -> dict[str, float]:
    return {
        "half_span": surface_half_span(surface),
        "full_span": surface_half_span(surface) * (2.0 if surface.symmetric else 1.0),
        "area": surface_planform_area(surface),
        "mean_chord": surface_mean_chord(surface),
        "aspect_ratio": surface_aspect_ratio(surface),
        "root_chord": surface.sections[0].chord if surface.sections else 0.0,
        "tip_chord": surface.sections[-1].chord if surface.sections else 0.0,
    }


def evaluate_design(
    config: main.AircraftConfig,
    velocity: float = 50.0,
    target_cl: float = 0.55,
    alpha_min: float = -4.0,
    alpha_max: float = 12.0,
    alpha_samples: int = 21,
) -> dict:
    airplane = main.build_airplane(config)
    alphas = np.linspace(alpha_min, alpha_max, alpha_samples)
    candidates = []
    wing_metrics = surface_metrics(config.wing)
    canard_metrics = surface_metrics(config.canard)
    htail_metrics = surface_metrics(config.htail) if config.htail.enabled else {}
    vtail_metrics = surface_metrics(config.vtail) if config.vtail.enabled else {}

    for alpha in alphas:
        op_point = main.asb.OperatingPoint(velocity=velocity, alpha=float(alpha))
        aero = main.asb.AeroBuildup(airplane=airplane, op_point=op_point).run_with_stability_derivatives()

        cl = float(np.atleast_1d(aero["CL"])[0])
        cd = float(np.atleast_1d(aero["CD"])[0])
        cm = float(np.atleast_1d(aero["Cm"])[0])
        cma = float(np.atleast_1d(aero["Cma"])[0])
        x_np = float(np.atleast_1d(aero["x_np"])[0])
        ld = cl / max(cd, 1e-6)
        cl_error = abs(cl - target_cl)

        score = (
            ld
            - 18.0 * cl_error
            - 0.22 * abs(cm)
            - 0.02 * abs(alpha)
            - max(0.0, cma) * 0.5
        )

        candidates.append(
            {
                "alpha": float(alpha),
                "CL": cl,
                "CD": cd,
                "Cm": cm,
                "Cma": cma,
                "x_np": x_np,
                "L_over_D": ld,
                "cl_error": cl_error,
                "point_score": score,
            }
        )

    best_point = max(candidates, key=lambda item: item["point_score"])
    penalty = geometry_penalty(config)
    final_score = best_point["point_score"] - penalty

    return {
        "score": final_score,
        "geometry_penalty": penalty,
        "best_point": best_point,
        "alpha_sweep": candidates,
        "geometry": {
            "wing": wing_metrics,
            "canard": canard_metrics,
            "htail": htail_metrics,
            "vtail": vtail_metrics,
            "fuselage_length": fuselage_length(config),
        },
    }


def sample_random_vector(rng: random.Random) -> np.ndarray:
    return np.array([rng.uniform(spec.low, spec.high) for spec in PARAMETER_SPECS], dtype=float)


def generate_dataset(
    base_config: main.AircraftConfig,
    sample_count: int,
    output_path: str,
    seed: int,
    velocity: float,
    target_cl: float,
) -> None:
    rng = random.Random(seed)
    output = Path(output_path)
    records = []

    baseline = baseline_vector()
    configs = [baseline]
    for _ in range(sample_count - 1):
        configs.append(sample_random_vector(rng))

    for index, vector in enumerate(configs, start=1):
        config = apply_design_vector(base_config, vector)
        evaluation = evaluate_design(config, velocity=velocity, target_cl=target_cl)
        record = {
            "index": index,
            "parameters": vector_to_dict(vector),
            "score": evaluation["score"],
            "geometry_penalty": evaluation["geometry_penalty"],
            "best_point": evaluation["best_point"],
        }
        records.append(record)
        print(
            f"[dataset] {index:04d}/{sample_count} "
            f"score={evaluation['score']:.3f} "
            f"LD={evaluation['best_point']['L_over_D']:.3f} "
            f"alpha={evaluation['best_point']['alpha']:.2f}"
        )

    output.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")
    print(f"Датасет сохранён: {output}")


def dataset_to_arrays(dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
    lines = Path(dataset_path).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    if not records:
        raise ValueError("Датасет пустой.")

    xs = []
    ys = []
    for record in records:
        params = record["parameters"]
        xs.append([float(params[spec.name]) for spec in PARAMETER_SPECS])
        ys.append(float(record["score"]))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def polynomial_features(x: np.ndarray) -> np.ndarray:
    features = [np.ones((x.shape[0], 1)), x, x**2]
    return np.concatenate(features, axis=1)


def fit_surrogate(dataset_path: str) -> dict:
    x, y = dataset_to_arrays(dataset_path)
    phi = polynomial_features(x)
    weights, *_ = np.linalg.lstsq(phi, y, rcond=None)
    predictions = phi @ weights
    rmse = float(np.sqrt(np.mean((predictions - y) ** 2)))
    return {
        "weights": weights,
        "rmse": rmse,
        "x": x,
        "y": y,
        "predictions": predictions,
    }


def surrogate_predict(weights: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return polynomial_features(vectors) @ weights


def optimize_design(
    base_config: main.AircraftConfig,
    iterations: int,
    population: int,
    elite_fraction: float,
    seed: int,
    velocity: float,
    target_cl: float,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    mean = baseline_vector()
    sigmas = np.array([(spec.high - spec.low) / 4.0 for spec in PARAMETER_SPECS], dtype=float)
    elite_count = max(2, int(population * elite_fraction))

    best_vector = mean.copy()
    best_evaluation = evaluate_design(apply_design_vector(base_config, best_vector), velocity=velocity, target_cl=target_cl)

    for iteration in range(1, iterations + 1):
        population_vectors = []
        population_scores = []
        population_evaluations = []

        population_vectors.append(best_vector.copy())

        while len(population_vectors) < population:
            sample = rng.normal(mean, sigmas)
            population_vectors.append(clamp_vector(sample))

        for vector in population_vectors:
            config = apply_design_vector(base_config, vector)
            evaluation = evaluate_design(config, velocity=velocity, target_cl=target_cl)
            population_scores.append(evaluation["score"])
            population_evaluations.append(evaluation)

        ranked = sorted(
            zip(population_scores, population_vectors, population_evaluations),
            key=lambda item: item[0],
            reverse=True,
        )
        elites = ranked[:elite_count]

        elite_vectors = np.array([item[1] for item in elites], dtype=float)
        mean = elite_vectors.mean(axis=0)
        sigmas = np.maximum(elite_vectors.std(axis=0), 0.03)

        if elites[0][0] > best_evaluation["score"]:
            best_vector = elites[0][1].copy()
            best_evaluation = elites[0][2]

        iteration_scores = [item[0] for item in ranked]
        print(
            f"[opt] iter={iteration:02d}/{iterations} "
            f"best={best_evaluation['score']:.3f} "
            f"iter_best={ranked[0][0]:.3f} "
            f"iter_mean={statistics.mean(iteration_scores):.3f}"
        )

    return best_vector, best_evaluation


def optimize_with_surrogate(
    dataset_path: str,
    base_config: main.AircraftConfig,
    proposal_count: int,
    evaluate_top_k: int,
    seed: int,
    velocity: float,
    target_cl: float,
) -> tuple[np.ndarray, dict]:
    model = fit_surrogate(dataset_path)
    rng = np.random.default_rng(seed)

    candidates = np.array(
        [sample_random_vector(random.Random(int(rng.integers(0, 10_000_000)))) for _ in range(proposal_count)]
    )
    predictions = surrogate_predict(model["weights"], candidates)
    best_indices = np.argsort(predictions)[-evaluate_top_k:][::-1]

    best_vector = baseline_vector()
    best_evaluation = evaluate_design(apply_design_vector(base_config, best_vector), velocity=velocity, target_cl=target_cl)
    for rank, index in enumerate(best_indices, start=1):
        vector = candidates[index]
        evaluation = evaluate_design(apply_design_vector(base_config, vector), velocity=velocity, target_cl=target_cl)
        print(
            f"[surrogate] rank={rank:02d}/{evaluate_top_k} "
            f"pred={predictions[index]:.3f} "
            f"real={evaluation['score']:.3f}"
        )
        if evaluation["score"] > best_evaluation["score"]:
            best_vector = vector
            best_evaluation = evaluation

    return best_vector, best_evaluation


def print_evaluation(label: str, evaluation: dict) -> None:
    point = evaluation["best_point"]
    geometry = evaluation.get("geometry", {})
    wing = geometry.get("wing", {})
    canard = geometry.get("canard", {})
    print(label)
    print(f"  score            : {evaluation['score']:.3f}")
    print(f"  geometry_penalty : {evaluation['geometry_penalty']:.3f}")
    print(f"  alpha            : {point['alpha']:.3f} deg")
    print(f"  CL               : {point['CL']:.4f}")
    print(f"  CD               : {point['CD']:.4f}")
    print(f"  L/D              : {point['L_over_D']:.4f}")
    print(f"  Cm               : {point['Cm']:.4f}")
    print(f"  Cma              : {point['Cma']:.4f}")
    print(f"  x_np             : {point['x_np']:.4f}")
    if wing:
        print(
            "  wing size        : "
            f"span={wing['full_span']:.3f}, area={wing['area']:.3f}, "
            f"root={wing['root_chord']:.3f}, tip={wing['tip_chord']:.3f}, AR={wing['aspect_ratio']:.3f}"
        )
    if canard:
        print(
            "  canard size      : "
            f"span={canard['full_span']:.3f}, area={canard['area']:.3f}, "
            f"root={canard['root_chord']:.3f}, tip={canard['tip_chord']:.3f}, AR={canard['aspect_ratio']:.3f}"
        )


def write_best_outputs(
    base_config: main.AircraftConfig,
    best_vector: np.ndarray,
    output_json: str | None,
    output_python: str | None,
) -> main.AircraftConfig:
    best_config = apply_design_vector(base_config, best_vector)
    if output_json is not None:
        save_config(best_config, output_json)
        print(f"Лучшая конфигурация сохранена: {output_json}")
    if output_python is not None:
        Path(output_python).write_text(main.generate_python_script(best_config), encoding="utf-8")
        print(f"Python-скрипт сохранён: {output_python}")
    return best_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Оптимизация крыла и ПГО для модели AeroSandbox.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=str, default=None, help="Путь к JSON-конфигурации самолёта.")
    common.add_argument("--velocity", type=float, default=50.0, help="Скорость анализа, м/с.")
    common.add_argument("--target-cl", type=float, default=0.55, help="Целевой коэффициент подъёмной силы.")
    common.add_argument("--seed", type=int, default=42, help="Сид для генератора случайных чисел.")

    sample = subparsers.add_parser("sample", parents=[common], help="Сгенерировать датасет случайных геометрий.")
    sample.add_argument("--samples", type=int, default=150, help="Количество примеров в датасете.")
    sample.add_argument("--output", type=str, default="dataset.jsonl", help="Куда сохранить датасет.")

    optimize = subparsers.add_parser("optimize", parents=[common], help="Прямая оптимизация через CEM.")
    optimize.add_argument("--iterations", type=int, default=12, help="Количество итераций оптимизации.")
    optimize.add_argument("--population", type=int, default=32, help="Популяция на итерацию.")
    optimize.add_argument("--elite-fraction", type=float, default=0.25, help="Доля элитных кандидатов.")
    optimize.add_argument("--output-json", type=str, default="best_config.json", help="Куда сохранить лучший JSON.")
    optimize.add_argument("--output-python", type=str, default="best_aircraft.py", help="Куда сохранить Python-код.")

    surrogate = subparsers.add_parser("surrogate", parents=[common], help="Суррогатный поиск по готовому датасету.")
    surrogate.add_argument("--dataset", type=str, required=True, help="Путь к датасету JSONL.")
    surrogate.add_argument("--proposal-count", type=int, default=400, help="Сколько кандидатов быстро оценивать суррогатом.")
    surrogate.add_argument("--evaluate-top-k", type=int, default=20, help="Сколько лучших суррогатных кандидатов считать точно.")
    surrogate.add_argument("--output-json", type=str, default="surrogate_best_config.json", help="Куда сохранить лучший JSON.")
    surrogate.add_argument("--output-python", type=str, default="surrogate_best_aircraft.py", help="Куда сохранить Python-код.")

    return parser


def main_cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_config = load_config(args.config)

    baseline_eval = evaluate_design(base_config, velocity=args.velocity, target_cl=args.target_cl)
    print_evaluation("Базовая конструкция", baseline_eval)

    if args.command == "sample":
        generate_dataset(
            base_config=base_config,
            sample_count=args.samples,
            output_path=args.output,
            seed=args.seed,
            velocity=args.velocity,
            target_cl=args.target_cl,
        )
        return

    if args.command == "optimize":
        best_vector, best_evaluation = optimize_design(
            base_config=base_config,
            iterations=args.iterations,
            population=args.population,
            elite_fraction=args.elite_fraction,
            seed=args.seed,
            velocity=args.velocity,
            target_cl=args.target_cl,
        )
        print_evaluation("Лучшая найденная конструкция", best_evaluation)
        best_config = write_best_outputs(base_config, best_vector, args.output_json, args.output_python)
        print("Лучшие параметры:")
        for name, value in vector_to_dict(best_vector).items():
            print(f"  {name}: {value:.4f}")
        print(f"Итоговая длина фюзеляжа: {fuselage_length(best_config):.3f}")
        return

    if args.command == "surrogate":
        model = fit_surrogate(args.dataset)
        print(f"RMSE суррогатной модели: {model['rmse']:.3f}")
        best_vector, best_evaluation = optimize_with_surrogate(
            dataset_path=args.dataset,
            base_config=base_config,
            proposal_count=args.proposal_count,
            evaluate_top_k=args.evaluate_top_k,
            seed=args.seed,
            velocity=args.velocity,
            target_cl=args.target_cl,
        )
        print_evaluation("Лучшая конструкция через суррогат", best_evaluation)
        write_best_outputs(base_config, best_vector, args.output_json, args.output_python)
        return


if __name__ == "__main__":
    main_cli()
