"""
Анализ и визуализация результатов оптимизации крыла.
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

import SCAT.AERO.optimizer as base_opt
import SCAT.AERO.wing_optimizer as wing_opt


def analyze_dataset(dataset_path: str) -> dict:
    """Анализировать датасет и вернуть статистику."""
    x, y, records = wing_opt.dataset_to_arrays_extended(dataset_path)
    
    valid_records = [r for r in records if r.get("is_valid", False)]
    valid_indices = np.array([r.get("is_valid", False) for r in records])
    
    analysis = {
        "total_samples": len(records),
        "valid_samples": len(valid_records),
        "invalid_samples": len(records) - len(valid_records),
        "valid_ratio": len(valid_records) / len(records) if records else 0,
        
        "score_stats": {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
            "valid_mean": float(y[valid_indices].mean()) if valid_indices.sum() > 0 else None,
            "valid_max": float(y[valid_indices].max()) if valid_indices.sum() > 0 else None,
        },
        
        "efficiency_stats": {
            "mean": float(np.mean([r.get("efficiency", 0) for r in records])),
            "max": float(np.max([r.get("efficiency", 0) for r in records])),
            "min": float(np.min([r.get("efficiency", 0) for r in records])),
        },
        
        "wing_area_stats": {
            "mean": float(np.mean([r.get("wing", {}).get("area", 0) for r in records])),
            "min": float(np.min([r.get("wing", {}).get("area", 0) for r in records])),
            "max": float(np.max([r.get("wing", {}).get("area", 0) for r in records])),
        },
        
        "stability_stats": {
            "Clp_mean": float(np.mean([r.get("stability", {}).get("Clp", 0) for r in records])),
            "Clp_stable": int(np.sum([r.get("stability", {}).get("Clp", 0) < -0.01 for r in records])),
            "Cnr_mean": float(np.mean([r.get("stability", {}).get("Cnr", 0) for r in records])),
            "Cnr_stable": int(np.sum([r.get("stability", {}).get("Cnr", 0) < -0.01 for r in records])),
        }
    }
    
    return analysis


def plot_dataset_analysis(dataset_path: str, output_path: str = "dataset_analysis.png"):
    """Создать графики анализа датасета."""
    x, y, records = wing_opt.dataset_to_arrays_extended(dataset_path)
    valid_indices = np.array([r.get("is_valid", False) for r in records])
    
    # Извлечь метрики
    efficiencies = np.array([r.get("efficiency", 0) for r in records])
    areas = np.array([r.get("wing", {}).get("area", 0) for r in records])
    clp = np.array([r.get("stability", {}).get("Clp", 0) for r in records])
    cnr = np.array([r.get("stability", {}).get("Cnr", 0) for r in records])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Анализ датасета оптимизации крыла", fontsize=16, fontweight='bold')
    
    # График 1: Распределение скоров
    ax = axes[0, 0]
    ax.hist(y, bins=30, alpha=0.7, label="Все конфиги", edgecolor='black')
    if valid_indices.sum() > 0:
        ax.hist(y[valid_indices], bins=30, alpha=0.7, label="Валидные", edgecolor='black')
    ax.set_xlabel("Скор оптимизации")
    ax.set_ylabel("Количество")
    ax.set_title("Распределение скоров")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # График 2: Эффективность vs Площадь крыла
    ax = axes[0, 1]
    invalid_mask = ~valid_indices
    if invalid_mask.sum() > 0:
        ax.scatter(areas[invalid_mask], efficiencies[invalid_mask], alpha=0.5, s=30, label="Невалидные", color='red')
    if valid_indices.sum() > 0:
        ax.scatter(areas[valid_indices], efficiencies[valid_indices], alpha=0.7, s=30, label="Валидные", color='green')
    ax.set_xlabel("Площадь крыла")
    ax.set_ylabel("Эффективность (CL/CD)")
    ax.set_title("Эффективность vs Площадь")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # График 3: Clp (демпфирование крена)
    ax = axes[0, 2]
    colors = ['green' if clp[i] < -0.01 else 'red' for i in range(len(clp))]
    ax.scatter(range(len(clp)), clp, c=colors, alpha=0.6, s=20)
    ax.axhline(y=-0.01, color='green', linestyle='--', label='Граница устойчивости')
    ax.set_xlabel("Образец")
    ax.set_ylabel("Clp (рад⁻¹)")
    ax.set_title("Демпфирование крена (Clp)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # График 4: Cnr (демпфирование рысканья: Cnr < -0.01, как is_stable_yaw)
    ax = axes[1, 0]
    colors = ['green' if cnr[i] < -0.01 else 'red' for i in range(len(cnr))]
    ax.scatter(range(len(cnr)), cnr, c=colors, alpha=0.6, s=20)
    ax.axhline(y=-0.01, color='green', linestyle='--', label='Граница устойчивости')
    ax.set_xlabel("Образец")
    ax.set_ylabel("Cnr (рад⁻¹)")
    ax.set_title("Стабильность рысканья (Cnr)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # График 5: Clp vs Cnr (крен: Clp < -0.01; рысканье: Cnr < -0.01)
    ax = axes[1, 1]
    stable_roll = clp < -0.01
    stable_yaw = cnr < -0.01
    both_stable = stable_roll & stable_yaw

    ax.scatter(clp[~both_stable], cnr[~both_stable], alpha=0.5, s=20, label="Не все стабильны", color='red')
    ax.scatter(clp[both_stable], cnr[both_stable], alpha=0.7, s=20, label="Оба стабильны", color='green')
    ax.axvline(x=-0.01, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.01, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel("Clp (демпфирование крена)")
    ax.set_ylabel("Cnr (стабильность рысканья)")
    ax.set_title("Взаимосвязь устойчивости крен-рыскание")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # График 6: Валидность по предпочтениям
    ax = axes[1, 2]
    categories = ['Все', 'Валидные', 'Площадь OK', 'Крен OK', 'Рысканье OK']
    counts = [
        len(records),
        valid_indices.sum(),
        (areas >= 0.3).sum(),
        (clp < -0.01).sum(),
        (cnr < -0.01).sum(),
    ]
    ax.bar(categories, counts, alpha=0.7, edgecolor='black')
    ax.set_ylabel("Количество конфигураций")
    ax.set_title("Статистика ограничений")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ График сохранён: {output_path}")
    plt.close()


def print_dataset_analysis(dataset_path: str):
    """Печать анализа датасета."""
    analysis = analyze_dataset(dataset_path)
    
    print("\n" + "=" * 70)
    print("  АНАЛИЗ ДАТАСЕТА")
    print("=" * 70)
    
    print(f"\n📊 Общая статистика:")
    print(f"  Всего образцов:      {analysis['total_samples']}")
    print(f"  Валидных:            {analysis['valid_samples']}")
    print(f"  Невалидных:          {analysis['invalid_samples']}")
    print(f"  Процент валидных:    {analysis['valid_ratio']*100:.1f}%")
    
    print(f"\n🎯 Скоры оптимизации:")
    print(f"  Среднее:             {analysis['score_stats']['mean']:8.3f}")
    print(f"  Std:                 {analysis['score_stats']['std']:8.3f}")
    print(f"  Min:                 {analysis['score_stats']['min']:8.3f}")
    print(f"  Max:                 {analysis['score_stats']['max']:8.3f}")
    if analysis['score_stats']['valid_mean']:
        print(f"  Валидных - среднее:  {analysis['score_stats']['valid_mean']:8.3f}")
        print(f"  Валидных - max:      {analysis['score_stats']['valid_max']:8.3f}")
    
    print(f"\n⚙️  Эффективность:")
    print(f"  Средняя:             {analysis['efficiency_stats']['mean']:8.3f}")
    print(f"  Min:                 {analysis['efficiency_stats']['min']:8.3f}")
    print(f"  Max:                 {analysis['efficiency_stats']['max']:8.3f}")
    
    print(f"\n🪟 Площадь крыла:")
    print(f"  Средняя:             {analysis['wing_area_stats']['mean']:8.3f}")
    print(f"  Min:                 {analysis['wing_area_stats']['min']:8.3f}")
    print(f"  Max:                 {analysis['wing_area_stats']['max']:8.3f}")
    
    print(f"\n🔄 Стабильность:")
    print(f"  Clp - средняя:       {analysis['stability_stats']['Clp_mean']:8.4f}")
    print(f"  Clp - стабильных:    {analysis['stability_stats']['Clp_stable']}/{analysis['total_samples']}")
    print(f"  Cnr - средняя:       {analysis['stability_stats']['Cnr_mean']:8.4f}")
    print(f"  Cnr - стабильных:    {analysis['stability_stats']['Cnr_stable']}/{analysis['total_samples']}")
    
    print()


def find_best_samples(dataset_path: str, top_k: int = 10) -> list:
    """Найти лучшие образцы в датасете."""
    x, y, records = wing_opt.dataset_to_arrays_extended(dataset_path)
    
    # Сортировать по скору
    best_indices = np.argsort(y)[-top_k:][::-1]
    
    best_samples = []
    for rank, idx in enumerate(best_indices, 1):
        record = records[idx]
        best_samples.append({
            'rank': rank,
            'index': record['index'],
            'score': record['score'],
            'efficiency': record['efficiency'],
            'wing_area': record['wing']['area'],
            'clp': record['stability']['Clp'],
            'cnr': record['stability']['Cnr'],
            'is_valid': record['is_valid'],
        })
    
    return best_samples


def print_best_samples(dataset_path: str, top_k: int = 10):
    """Печать лучших образцов датасета."""
    samples = find_best_samples(dataset_path, top_k=top_k)
    
    print("\n" + "=" * 100)
    print(f"  ТОП-{top_k} ЛУЧШИХ КОНФИГУРАЦИЙ")
    print("=" * 100)
    print(f"{'Ранг':<5} {'Индекс':<8} {'Скор':<10} {'Эфф':<8} {'Площадь':<10} {'Clp':<10} {'Cnr':<10} {'Статус':<8}")
    print("-" * 100)
    
    for sample in samples:
        status = "✓ OK" if sample['is_valid'] else "✗ NO"
        print(
            f"{sample['rank']:<5} {sample['index']:<8} "
            f"{sample['score']:>9.3f} {sample['efficiency']:>7.3f} "
            f"{sample['wing_area']:>9.4f} {sample['clp']:>9.4f} {sample['cnr']:>9.4f} "
            f"{status:<8}"
        )
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Анализ результатов оптимизации крыла")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к датасету JSONL")
    parser.add_argument("--plot", action="store_true", help="Создать графики")
    parser.add_argument("--top-k", type=int, default=10, help="Показать топ-k лучших")
    
    args = parser.parse_args()
    
    print_dataset_analysis(args.dataset)
    print_best_samples(args.dataset, top_k=args.top_k)
    
    if args.plot:
        plot_dataset_analysis(args.dataset)
