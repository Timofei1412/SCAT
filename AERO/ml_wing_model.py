"""
ML модель для оптимизации крыла.
Использует нейросеть для прямого предсказания оптимальной геометрии.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import random

# Попытаемся использовать TensorFlow/Keras, если не доступна - падём со статусом
try:
    import tensorflow as tf
    # Универсальный импорт: работает и с tf.keras, и с standalone keras
    try:
        from tensorflow.keras import layers, models, saving
        keras = tf.keras
    except ImportError:
        import keras
        from keras import layers, models, saving
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️  TensorFlow не установлен. Установи: pip install tensorflow")

import SCAT.AERO.optimizer as base_opt
import SCAT.AERO.wing_optimizer as wing_opt


# =============================================================================
# 🔧 РЕГИСТРАЦИЯ КАСТОМНОГО КЛАССА ДЛЯ KERAS 3.x
# =============================================================================
@keras.saving.register_keras_serializable(package="AERO", name="WingOptimizationNN")
class WingOptimizationNN(keras.Model):
    """
    Нейросеть для оптимизации геометрии крыла.
    
    Входы: параметры геометрии крыла (PARAMETER_SPECS)
    Выходы: предсказанный скор оптимизации
    """
    
    def __init__(self, input_dim: int = None, hidden_dims: list = None, **kwargs):
        # Поддержка загрузки: input_dim может прийти из config
        if input_dim is None:
            input_dim = len(base_opt.PARAMETER_SPECS)
        if hidden_dims is None:
            hidden_dims = [256, 512, 256, 128]
            
        super().__init__(**kwargs)
        
        # Нормализационный слой
        self.input_norm = layers.Normalization()
        
        # Основная сеть
        self.dense_layers = []
        self.dropout_layers = []
        self.batch_norm_layers = []
        
        for hidden_dim in hidden_dims:
            self.dense_layers.append(layers.Dense(hidden_dim, activation=None))
            self.batch_norm_layers.append(layers.BatchNormalization())
            self.dropout_layers.append(layers.Dropout(0.3))
        
        # Выходной слой
        self.output_layer = layers.Dense(1, activation=None)
        
        # Сохраняем конфигурацию для сериализации
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
    
    def get_config(self):
        """Сериализация конфигурации для сохранения модели"""
        config = super().get_config()
        config.update({
            "input_dim": self._input_dim,
            "hidden_dims": self._hidden_dims,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Десериализация конфигурации при загрузке модели"""
        return cls(**config)
    
    def adapt(self, x: np.ndarray):
        """Адаптировать нормализационный слой к данным."""
        self.input_norm.adapt(x)
    
    def call(self, x, training=False):
        x = self.input_norm(x)
        
        for dense, bn, dropout in zip(self.dense_layers, self.batch_norm_layers, self.dropout_layers):
            x = dense(x)
            x = bn(x, training=training)
            x = layers.ReLU()(x)
            x = dropout(x, training=training)
        
        return self.output_layer(x)


# =============================================================================
# 🎓 ОБУЧЕНИЕ МОДЕЛИ
# =============================================================================
def create_and_train_model(
    dataset_path: str,
    model_path: str = "wing_model.keras",
    validation_split: float = 0.2,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    only_valid: bool = False,
) -> Tuple[keras.Model, dict]:
    """
    Создать и обучить модель предсказания оптимальности крыла.
    """
    if not HAS_TF:
        raise ImportError("TensorFlow требуется для обучения моделей")
    
    print(f"📊 Загрузка датасета: {dataset_path}")
    x, y, records = wing_opt.dataset_to_arrays_extended(dataset_path, only_valid=only_valid)
    
    print(f"✓ Загружено {len(x)} примеров")
    print(f"  Средний скор: {y.mean():.3f}, std: {y.std():.3f}")
    print(f"  Min: {y.min():.3f}, Max: {y.max():.3f}")
    
    # Нормализовать целевую переменную
    y_mean = y.mean()
    y_std = y.std()
    y_norm = (y - y_mean) / (y_std + 1e-8)
    
    # Создать модель
    print("\n🧠 Создание нейросети...")
    model = WingOptimizationNN(input_dim=x.shape[1], hidden_dims=[256, 512, 256, 128])
    
    # Адаптировать нормализацию к данным
    model.adapt(x)
    
    # Скомпилировать
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )
    
    # Обучить
    print(f"🚀 Обучение на {epochs} эпохах...")
    history = model.fit(
        x, y_norm,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
    )
    
    # Сохранить модель
    print(f"\n💾 Сохранение модели: {model_path}")
    model.save(model_path)
    
    # Оценить на тестовых данных
    predictions = model(x, training=False).numpy().flatten()
    train_mae = np.mean(np.abs(predictions - y_norm))
    print(f"✓ Training MAE (normalized): {train_mae:.4f}")
    
    # Денормализовать для отчёта
    predictions_denorm = predictions * (y_std + 1e-8) + y_mean
    denorm_mae = np.mean(np.abs(predictions_denorm - y))
    print(f"✓ Training MAE (denormalized): {denorm_mae:.4f}")
    
    history_dict = {
        "loss": [float(x) for x in history.history.get('loss', [])],
        "val_loss": [float(x) for x in history.history.get('val_loss', [])],
        "mae": [float(x) for x in history.history.get('mean_absolute_error', [])],
        "val_mae": [float(x) for x in history.history.get('val_mean_absolute_error', [])],
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "final_mae": float(denorm_mae),
    }
    
    return model, history_dict


# =============================================================================
# 📥 ЗАГРУЗКА МОДЕЛИ (ИСПРАВЛЕНА)
# =============================================================================
def load_trained_model(model_path: str) -> keras.Model:
    """
    Загрузить сохранённую модель с поддержкой кастомных классов.
    """
    if not HAS_TF:
        raise ImportError("TensorFlow требуется для работы с моделями")
    
    # Явно передаём custom_objects, чтобы Keras нашёл наш класс
    custom_objects = {
        'WingOptimizationNN': WingOptimizationNN,
    }
    
    try:
        # Пробуем загрузить с custom_objects
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=True)
        return model
    except Exception as e:
        # Если не получилось — пробуем без компиляции
        print(f"⚠️  Первая попытка загрузки не удалась: {e}")
        print("   Пробуем загрузить без компиляции...")
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return model


# =============================================================================
# 🤖 ОПТИМИЗАЦИЯ ЧЕРЕЗ НЕЙРОСЕТЬ
# =============================================================================
def optimize_with_neural_network(
    model: keras.Model,
    base_config,
    proposal_count: int = 5000,
    evaluate_top_k: int = 50,
    seed: int = 42,
    velocity: float = 50.0,
    target_cl: float = 0.55,
    min_wing_area: float = 0.3,
) -> Tuple[np.ndarray, wing_opt.DesignEvaluation]:
    """
    Оптимизировать геометрию крыла используя обученную нейросеть.
    """
    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    
    print(f"🤖 Генерация {proposal_count} кандидатов через нейросеть...")
    
    # Сгенерировать кандидатов
    candidates = np.array([
        base_opt.sample_random_vector(rng)
        for _ in range(proposal_count)
    ])
    
    # Быстрая оценка через нейросеть
    nn_predictions = model(candidates, training=False).numpy().flatten()
    
    # Выбрать топ-k по предсказаниям нейросети
    best_indices = np.argsort(nn_predictions)[-evaluate_top_k:][::-1]
    
    print(f"🔍 Детальная оценка топ-{evaluate_top_k} кандидатов...")
    
    best_vector = base_opt.baseline_vector()
    best_evaluation = wing_opt.evaluate_wing_design(
        base_opt.apply_design_vector(base_config, best_vector),
        velocity=velocity,
        target_cl=target_cl,
        min_wing_area=min_wing_area,
    )
    
    for rank, index in enumerate(best_indices, start=1):
        vector = candidates[index]
        config = base_opt.apply_design_vector(base_config, vector)
        evaluation = wing_opt.evaluate_wing_design(
            config,
            velocity=velocity,
            target_cl=target_cl,
            min_wing_area=min_wing_area,
        )
        
        status = "✓" if evaluation.is_valid(min_wing_area) else "✗"
        print(
            f"[{status}] rank={rank:2d}/{evaluate_top_k} "
            f"nn_pred={nn_predictions[index]:8.3f} "
            f"real={evaluation.score:8.3f} "
            f"eff={evaluation.efficiency:6.3f}"
        )
        
        if evaluation.score > best_evaluation.score:
            best_vector = vector
            best_evaluation = evaluation
    
    return best_vector, best_evaluation


# =============================================================================
# 📊 АНАЛИЗ МОДЕЛИ
# =============================================================================
def analyze_model_predictions(
    model: keras.Model,
    dataset_path: str,
) -> dict:
    """Анализировать предсказания модели на датасете."""
    x, y, records = wing_opt.dataset_to_arrays_extended(dataset_path)
    
    predictions = model(x, training=False).numpy().flatten()
    
    # Найти средние и стандартные отклонения из датасета для денормализации
    y_mean = y.mean()
    y_std = y.std()
    
    predictions_denorm = predictions * (y_std + 1e-8) + y_mean
    
    mae = np.mean(np.abs(predictions_denorm - y))
    rmse = np.sqrt(np.mean((predictions_denorm - y) ** 2))
    r2 = 1.0 - (np.sum((predictions_denorm - y) ** 2) / np.sum((y - y_mean) ** 2))
    
    # Найти лучшие и худшие предсказания
    errors = np.abs(predictions_denorm - y)
    best_predictions = np.argsort(errors)[:5]
    worst_predictions = np.argsort(errors)[-5:]
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "best_predictions": [
            {
                "index": int(i),
                "true": float(y[i]),
                "pred": float(predictions_denorm[i]),
                "error": float(errors[i]),
            }
            for i in best_predictions
        ],
        "worst_predictions": [
            {
                "index": int(i),
                "true": float(y[i]),
                "pred": float(predictions_denorm[i]),
                "error": float(errors[i]),
            }
            for i in worst_predictions
        ],
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML модель для оптимизации крыла")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    train_cmd = subparsers.add_parser("train", help="Обучить модель")
    train_cmd.add_argument("--dataset", type=str, required=True, help="Путь к датасету JSONL")
    train_cmd.add_argument("--output", type=str, default="wing_model.keras", help="Где сохранить модель")
    train_cmd.add_argument("--epochs", type=int, default=100, help="Количество эпох")
    train_cmd.add_argument("--batch-size", type=int, default=32, help="Размер батча")
    train_cmd.add_argument("--only-valid", action="store_true", help="Обучать только на валидных конфигах")
    
    analyze_cmd = subparsers.add_parser("analyze", help="Анализировать предсказания модели")
    analyze_cmd.add_argument("--model", type=str, required=True, help="Путь к модели")
    analyze_cmd.add_argument("--dataset", type=str, required=True, help="Путь к датасету JSONL")
    
    args = parser.parse_args()
    
    if args.command == "train":
        model, history = create_and_train_model(
            dataset_path=args.dataset,
            model_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            only_valid=args.only_valid,
        )
        print(f"\n✅ Модель обучена и сохранена: {args.output}")
    
    elif args.command == "analyze":
        model = load_trained_model(args.model)
        analysis = analyze_model_predictions(model, args.dataset)
        print("\n📊 Анализ предсказаний модели:")
        print(f"MAE:  {analysis['mae']:.4f}")
        print(f"RMSE: {analysis['rmse']:.4f}")
        print(f"R²:   {analysis['r2']:.4f}")