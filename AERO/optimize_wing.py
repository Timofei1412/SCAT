"""
Основной скрипт оптимизации крыла.
Управляет всей цепочкой: генерация датасета -> обучение ML -> оптимизация -> анализ.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import SCAT.AERO.main as main
import SCAT.AERO.optimizer as base_opt
import SCAT.AERO.wing_optimizer as wing_opt


def print_header(title: str):
    """Печать красивого заголовка."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_evaluation(label: str, evaluation: wing_opt.DesignEvaluation):
    """Печать результатов оценки конфигурации."""
    print(f"\n{label}")
    print(f"  Скор:             {evaluation.score:10.3f}")
    print(f"  Эффективность:    {evaluation.efficiency:10.3f} (CL/CD)")
    print(f"  Штраф геометрии:  {evaluation.geometry_penalty:10.3f}")
    print(f"  Нарушение огран.:  {evaluation.constraints_violation:10.3f}")
    
    print(f"\n  Крыло:")
    print(f"    Площадь:        {evaluation.wing.area:10.4f}")
    print(f"    Размах:         {evaluation.wing.span:10.4f}")
    print(f"    Хорда корня:    {evaluation.wing.root_chord:10.4f}")
    print(f"    Хорда законц.:  {evaluation.wing.tip_chord:10.4f}")
    print(f"    Удлинение:      {evaluation.wing.aspect_ratio:10.4f}")
    print(f"    Стреловидн.:    {evaluation.wing.sweep:10.4f}°")
    print(f"    Диэдр. угол:    {evaluation.wing.dihedral:10.4f}°")
    
    print(f"\n  Стабильность:")
    print(f"    Cm (питчинг):   {evaluation.stability.Cm:10.4f}")
    print(f"    Cma (стаб.):    {evaluation.stability.Cma:10.4f} {'✓ устойчив' if evaluation.stability.Cma < -0.01 else '✗ неустойчив'}")
    print(f"    Cmq (демпф.):   {evaluation.stability.Cmq:10.4f}")
    print(f"    Clp (крен демп):{evaluation.stability.Clp:10.4f} {'✓' if wing_opt.StabilityMetrics(0,0,evaluation.stability.Clp,0,0,0,0,0,0).is_stable_roll() else '✗'}")
    print(f"    Cnr (рысканье): {evaluation.stability.Cnr:10.4f} {'✓' if wing_opt.StabilityMetrics(0,0,0,0,evaluation.stability.Cnr,0,0,0,0).is_stable_yaw() else '✗'}")
    
    print(f"\n  Аэродинамика:")
    print(f"    CL:             {evaluation.stability.Cl:10.4f}")
    print(f"    CD:             {evaluation.stability.Cd:10.4f}")


def run_full_optimization(
    config_path: Optional[str] = None,
    dataset_samples: int = 200,
    dataset_path: str = "wing_dataset.jsonl",
    model_path: str = "wing_model.keras",
    velocity: float = 50.0,
    target_cl: float = 0.55,
    min_wing_area: float = 0.3,
    nn_proposal_count: int = 5000,
    nn_evaluate_top_k: int = 50,
    ml_epochs: int = 100,
    skip_dataset: bool = False,
    skip_training: bool = False,
):
    """
    Полная цепочка оптимизации.
    
    Args:
        config_path: путь к базовой конфигурации (None = по умолчанию)
        dataset_samples: количество примеров в датасете
        dataset_path: путь к датасету
        model_path: путь к модели
        velocity: скорость полёта для анализа
        target_cl: целевой CL
        min_wing_area: минимальная площадь крыла
        nn_proposal_count: количество кандидатов для быстрой оценки
        nn_evaluate_top_k: количество лучших для детальной оценки
        ml_epochs: количество эпох обучения
        skip_dataset: пропустить генерацию датасета
        skip_training: пропустить обучение модели
    """
    
    # Загрузить базовую конфигурацию
    print_header("📍 ОПТИМИЗАЦИЯ КРЫЛА")
    print("Загрузка базовой конфигурации...")
    base_config = base_opt.load_config(config_path)
    
    # Оценить базовую конфигурацию
    baseline_eval = wing_opt.evaluate_wing_design(base_config, velocity=velocity, target_cl=target_cl)
    print_evaluation("Базовая конфигурация", baseline_eval)
    
    # ===== ЭТАП 1: Генерация датасета =====
    if not skip_dataset:
        print_header("📊 ЭТАП 1: ГЕНЕРАЦИЯ ДАТАСЕТА")
        dataset_file = Path(dataset_path)
        
        if dataset_file.exists():
            print(f"⚠️  Датасет уже существует: {dataset_path}")
            response = input("Перегенерировать? (y/n): ").strip().lower()
            if response != 'y':
                print("Используем существующий датасет.")
            else:
                wing_opt.generate_wing_dataset(
                    base_config=base_config,
                    sample_count=dataset_samples,
                    output_path=dataset_path,
                    velocity=velocity,
                    target_cl=target_cl,
                    min_wing_area=min_wing_area,
                )
        else:
            wing_opt.generate_wing_dataset(
                base_config=base_config,
                sample_count=dataset_samples,
                output_path=dataset_path,
                velocity=velocity,
                target_cl=target_cl,
                min_wing_area=min_wing_area,
            )
    
    # ===== ЭТАП 2: Обучение ML модели =====
    if not skip_training:
        print_header("🧠 ЭТАП 2: ОБУЧЕНИЕ ML МОДЕЛИ")
        
        model_file = Path(model_path)
        if model_file.exists():
            print(f"⚠️  Модель уже существует: {model_path}")
            response = input("Переобучить? (y/n): ").strip().lower()
            if response != 'y':
                print("Используем существующую модель.")
            else:
                try:
                    import SCAT.AERO.ml_wing_model as ml_model
                    model, history = ml_model.create_and_train_model(
                        dataset_path=dataset_path,
                        model_path=model_path,
                        epochs=ml_epochs,
                    )
                except ImportError as e:
                    print(f"❌ Ошибка: {e}")
                    print("Обучение модели пропущено.")
        else:
            try:
                import SCAT.AERO.ml_wing_model as ml_model
                model, history = ml_model.create_and_train_model(
                    dataset_path=dataset_path,
                    model_path=model_path,
                    epochs=ml_epochs,
                )
            except ImportError as e:
                print(f"❌ Ошибка: {e}")
                print("Обучение модели пропущено.")
    
    # ===== ЭТАП 3: Оптимизация =====
    print_header("🚀 ЭТАП 3: ОПТИМИЗАЦИЯ С ML")
    
    try:
        import SCAT.AERO.ml_wing_model as ml_model
        
        model = ml_model.load_trained_model(model_path)
        best_vector, best_evaluation = ml_model.optimize_with_neural_network(
            model=model,
            base_config=base_config,
            proposal_count=nn_proposal_count,
            evaluate_top_k=nn_evaluate_top_k,
            velocity=velocity,
            target_cl=target_cl,
            min_wing_area=min_wing_area,
        )
        
        print_evaluation("Лучшая найденная конфигурация", best_evaluation)
        
        # Сохранить результат
        best_config = base_opt.apply_design_vector(base_config, best_vector)
        output_json = "optimized_wing.json"
        output_python = "optimized_wing.py"
        
        base_opt.save_config(best_config, output_json)
        Path(output_python).write_text(
            main.generate_python_script(best_config),
            encoding="utf-8"
        )
        
        print(f"\n✅ Результаты сохранены:")
        print(f"   JSON конфиг: {output_json}")
        print(f"   Python код: {output_python}")
        
        # Сохранить параметры
        params_file = "optimized_params.json"
        params = {
            "parameters": base_opt.vector_to_dict(best_vector),
            "score": float(best_evaluation.score),
            "efficiency": float(best_evaluation.efficiency),
            "wing": {
                "area": float(best_evaluation.wing.area),
                "aspect_ratio": float(best_evaluation.wing.aspect_ratio),
                "span": float(best_evaluation.wing.span),
            },
            "stability": {
                "Cma": float(best_evaluation.stability.Cma),
                "Clp": float(best_evaluation.stability.Clp),
                "Cnr": float(best_evaluation.stability.Cnr),
            },
            "valid": best_evaluation.is_valid(min_wing_area),
        }
        Path(params_file).write_text(json.dumps(params, indent=2, ensure_ascii=False))
        print(f"   Параметры: {params_file}")
        
        # Сравнение с базовой конфигурацией
        print(f"\n📈 Сравнение с базовой конфигурацией:")
        improvement = ((best_evaluation.score - baseline_eval.score) / max(abs(baseline_eval.score), 1e-6)) * 100
        print(f"   Улучшение скора: {improvement:+.1f}%")
        print(f"     Базовый:  {baseline_eval.score:8.3f}")
        print(f"     Лучший:   {best_evaluation.score:8.3f}")
        
        eff_improvement = ((best_evaluation.efficiency - baseline_eval.efficiency) / max(baseline_eval.efficiency, 1e-6)) * 100
        print(f"   Улучшение эффективности: {eff_improvement:+.1f}%")
        print(f"     Базовая:  {baseline_eval.efficiency:6.3f}")
        print(f"     Лучшая:   {best_evaluation.efficiency:6.3f}")
        
    except ImportError as e:
        print(f"❌ Ошибка: {e}")
        print("Оптимизация через ML невозможна.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Полная оптимизация крыла самолета с использованием ML"
    )
    
    parser.add_argument("--config", type=str, default=None, help="Путь к базовой конфигурации")
    parser.add_argument("--samples", type=int, default=200, help="Примеров в датасете")
    parser.add_argument("--dataset", type=str, default="wing_dataset.jsonl", help="Путь к датасету")
    parser.add_argument("--model", type=str, default="wing_model.keras", help="Путь к модели")
    parser.add_argument("--velocity", type=float, default=50.0, help="Скорость, м/с")
    parser.add_argument("--target-cl", type=float, default=0.55, help="Целевой CL")
    parser.add_argument("--min-wing-area", type=float, default=0.3, help="Минимальная площадь крыла")
    parser.add_argument("--proposals", type=int, default=5000, help="Кандидатов для быстрой оценки")
    parser.add_argument("--evaluate-top-k", type=int, default=50, help="Лучших для детальной оценки")
    parser.add_argument("--epochs", type=int, default=100, help="Эпох обучения")
    parser.add_argument("--skip-dataset", action="store_true", help="Пропустить генерацию датасета")
    parser.add_argument("--skip-training", action="store_true", help="Пропустить обучение модели")
    
    args = parser.parse_args()
    
    run_full_optimization(
        config_path=args.config,
        dataset_samples=args.samples,
        dataset_path=args.dataset,
        model_path=args.model,
        velocity=args.velocity,
        target_cl=args.target_cl,
        min_wing_area=args.min_wing_area,
        nn_proposal_count=args.proposals,
        nn_evaluate_top_k=args.evaluate_top_k,
        ml_epochs=args.epochs,
        skip_dataset=args.skip_dataset,
        skip_training=args.skip_training,
    )
