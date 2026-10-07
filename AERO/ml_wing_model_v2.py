"""
ML v2: суррогат по per-section параметрам (control_v2). Фюзеляж не входит в вектор признаков.

Цель обучения: смесь score и efficiency из датасета (устойчивее к выбросам, чем один score).
Сохранение: только keras.Sequential — без кастомных подклассов Model (корректный .keras zip).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers
    keras = tf.keras
    HAS_TF = True
except Exception:
    HAS_TF = False

import SCAT.AERO.wing_optimizer as wing_opt
import SCAT.AERO.optimizer as base_opt
import SCAT.AERO.control_v2 as control_v2
import SCAT.AERO.main as main


def _max_wing_span_from_config(cfg: main.AircraftConfig) -> float:
    return wing_opt.extract_wing_metrics(cfg.wing).span


def build_surrogate_mlp(input_dim: int, hidden_dims: tuple[int, ...] = (128, 256, 128)) -> keras.Model:
    """Полносвязная сеть; первый слой — Normalization (адаптация на train X)."""
    reg = regularizers.l2(1e-5)
    blocks: list = [layers.Normalization(axis=-1, input_shape=(input_dim,))]
    for dim in hidden_dims:
        blocks.append(layers.Dense(dim, activation="relu", kernel_regularizer=reg))
        blocks.append(layers.Dropout(0.12))
    blocks.append(layers.Dense(1))
    return keras.Sequential(blocks)


def load_dataset_for_specs(dataset_path: str, specs: List[control_v2.ParameterSpec], only_valid: bool = False):
    lines = Path(dataset_path).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    xs = []
    ys = []
    filtered = []
    for rec in records:
        if only_valid and not rec.get("is_valid", False):
            continue
        params = rec.get("parameters", {})
        try:
            x = [float(params[spec.name]) for spec in specs]
        except KeyError as exc:
            raise KeyError(
                "Dataset missing expected parameter keys for current base config specs. "
                "Regenerate dataset with the same aircraft JSON / default_config as training."
            ) from exc
        xs.append(x)
        score = float(rec.get("score", 0.0))
        eff = float(rec.get("efficiency", 0.0))
        if np.isfinite(eff) and 0.0 < eff < 400.0:
            # Нормируем L/D к шкале ~[-3, 3] для баланса с score
            y_val = 0.55 * score + 0.45 * np.tanh(eff / 45.0) * 3.0
        else:
            y_val = score
        ys.append(y_val)
        filtered.append(rec)
    return np.array(xs, dtype=float), np.array(ys, dtype=float), filtered


def create_and_train_model_v2(
    dataset_path: str,
    base_config_path: str | None,
    model_path: str = "wing_model_v2.keras",
    epochs: int = 120,
    batch_size: int = 32,
    validation_split: float = 0.15,
    learning_rate: float = 8e-4,
    only_valid: bool = False,
):
    if not HAS_TF:
        raise ImportError("TensorFlow required")

    base_config = base_opt.load_config(base_config_path)
    specs = control_v2.build_parameter_specs_for_config(base_config)

    print(f"Using {len(specs)} specs (per-section parameters)")

    x, y, records = load_dataset_for_specs(dataset_path, specs, only_valid=only_valid)
    if len(x) < 8:
        raise ValueError(f"Слишком мало примеров после фильтрации: {len(x)}. Увеличьте датасет или отключите --only-valid.")

    y_mean = float(y.mean())
    y_std = float(y.std() if y.std() > 1e-8 else 1.0)
    y_norm = (y - y_mean) / y_std

    model = build_surrogate_mlp(x.shape[1])
    model.layers[0].adapt(x)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.Huber(delta=1.25),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=18, restore_best_weights=True, min_delta=1e-4
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        x,
        y_norm,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=min(batch_size, max(8, len(x) // 4)),
        callbacks=callbacks,
        verbose=1,
    )

    meta_path = str(Path(model_path).with_suffix(".meta.json"))
    meta = {
        "y_mean": y_mean,
        "y_std": y_std,
        "input_dim": int(x.shape[1]),
        "spec_count": len(specs),
        "train_samples": int(len(x)),
        "only_valid": only_valid,
        "history_epochs": len(history.history.get("loss", [])),
    }
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Meta saved to {meta_path}")

    print(f"Saving model to {model_path}")
    try:
        model.save(model_path)
    except Exception as exc:
        # Fallback: веса + конфиг Sequential всё равно сериализуемы
        alt = str(Path(model_path).with_suffix(".weights.h5"))
        model.save_weights(alt)
        Path(str(Path(model_path).with_suffix(".json"))).write_text(model.to_json(), encoding="utf-8")
        print(f"Keras zip save failed ({exc}); saved weights to {alt} and architecture JSON next to it.")
        raise

    return model, meta


def load_surrogate_v2(model_path: str) -> keras.Model:
    """Загрузка обученной Sequential-модели (без custom_objects)."""
    return keras.models.load_model(model_path, compile=False)


def load_v2_training_meta(model_path: str) -> dict | None:
    """Метаданные нормализации цели (y_mean, y_std) рядом с .keras."""
    meta_path = Path(model_path).with_suffix(".meta.json")
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def optimize_with_neural_network_v2(
    model: keras.Model,
    base_config: main.AircraftConfig,
    proposal_count: int = 2000,
    evaluate_top_k: int = 40,
    seed: int = 42,
    velocity: float = 50.0,
    target_cl: float = 0.55,
    y_mean: float = 0.0,
    y_std: float = 1.0,
):
    specs = control_v2.build_parameter_specs_for_config(base_config)
    max_span = _max_wing_span_from_config(base_config)
    rng = random.Random(seed)
    candidates = np.array([control_v2.sample_random_vector(specs, rng) for _ in range(proposal_count)])

    preds = model(candidates, training=False).numpy().flatten()
    std = y_std if abs(y_std) > 1e-12 else 1.0
    preds = preds * std + y_mean
    best_indices = np.argsort(preds)[-evaluate_top_k:][::-1]

    best_vector = control_v2.baseline_vector_from_specs(specs)
    best_eval = wing_opt.evaluate_wing_design(
        control_v2.apply_vector_to_config(base_config, specs, best_vector),
        velocity=velocity,
        target_cl=target_cl,
        max_wing_span=max_span,
    )

    for rank, idx in enumerate(best_indices, start=1):
        vec = candidates[idx]
        cfg = control_v2.apply_vector_to_config(base_config, specs, vec)
        evalr = wing_opt.evaluate_wing_design(
            cfg, velocity=velocity, target_cl=target_cl, max_wing_span=max_span
        )
        status = "✓" if evalr.is_valid(max_wing_span=max_span) else "✗"
        print(
            f"[{status}] rank={rank:3d}/{evaluate_top_k} pred={preds[idx]:8.3f} "
            f"real={evalr.score:8.3f} L/D={evalr.efficiency:6.3f}"
        )
        if evalr.score > best_eval.score:
            best_eval = evalr
            best_vector = vec.copy()

    return best_vector, best_eval


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML v2 для оптимизации по-секционных параметров")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--dataset", required=True)
    t.add_argument(
        "--base-config",
        required=False,
        default=None,
        help="Путь к JSON конфигу. Если пропущен — будет использован дефолтный конфиг.",
    )
    t.add_argument("--output", default="wing_model_v2.keras")
    t.add_argument("--epochs", type=int, default=120)
    t.add_argument(
        "--only-valid",
        action="store_true",
        help="Обучать только на is_valid=true (меньше шума, нужно больше строк в датасете).",
    )

    o = sub.add_parser("optimize")
    o.add_argument("--model", required=True)
    o.add_argument("--base-config", required=False, default=None)
    o.add_argument("--proposals", type=int, default=2000)
    o.add_argument("--topk", type=int, default=40)

    args = parser.parse_args()
    if args.cmd == "train":
        create_and_train_model_v2(
            args.dataset,
            args.base_config,
            model_path=args.output,
            epochs=args.epochs,
            only_valid=args.only_valid,
        )
    elif args.cmd == "optimize":
        model = load_surrogate_v2(args.model)
        meta = load_v2_training_meta(args.model) or {}
        base_cfg = base_opt.load_config(args.base_config)
        vec, evaluation = optimize_with_neural_network_v2(
            model,
            base_cfg,
            proposal_count=args.proposals,
            evaluate_top_k=args.topk,
            y_mean=float(meta.get("y_mean", 0.0)),
            y_std=float(meta.get("y_std", 1.0)),
        )
        print("\nBest evaluation:")
        print(f" score={evaluation.score:.3f} L/D={evaluation.efficiency:.3f}")
