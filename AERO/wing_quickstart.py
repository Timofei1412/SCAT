#!/usr/bin/env python3
"""
🚀 БЫСТРЫЙ СТАРТ: Оптимизация крыла с ML

Запустить этот скрипт для полной оптимизации:
  python3 wing_quickstart.py

Или используй отдельные команды:
  python3 wing_quickstart.py --step 1  # Только датасет
  python3 wing_quickstart.py --step 2  # Только анализ
  python3 wing_quickstart.py --step 3  # Только обучение модели
  python3 wing_quickstart.py --step 4  # Только оптимизацию
"""

import sys
import subprocess
from pathlib import Path


def print_banner(title):
    """Печать красивого баннера."""
    width = 70
    print("\n" + "█" * width)
    print(f"  {title:^66}  ")
    print("█" * width + "\n")


def run_command(cmd, description):
    """Запустить команду с красивым выводом."""
    print(f"▶ {description}...")
    print(f"  Команда: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    
    if result.returncode != 0:
        print(f"\n❌ Ошибка при выполнении: {description}")
        return False
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Быстрый старт оптимизации крыла"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Запустить только конкретный этап (1-5)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=150,
        help="Количество примеров для датасета"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Количество эпох обучения"
    )
    
    args = parser.parse_args()
    
    print_banner("🛩️ СИСТЕМА ОПТИМИЗАЦИИ КРЫЛА")
    print("""
    Эта система оптимизирует геометрию крыла самолета с использованием:
    • AeroSandbox - аэродинамический анализ
    • TensorFlow - нейросетевая оптимизация
    • Генетические алгоритмы - эволюционная оптимизация
    
    Учитываются ограничения:
    ✓ Минимальная площадь крыла для полета
    ✓ Стабильность по крену (Clp < -0.01)
    ✓ Стабильность по рысканью (Cnr > 0.01)
    ✓ Нейтральность по тангажу (Cm ≈ 0)
    
    Оптимизируется:
    📊 Эффективность (CL/CD)
    """)
    
    # Определить какие этапы запустить
    steps_to_run = [1, 2, 3, 4, 5] if args.step is None else [args.step]
    
    # Этап 1: Генерация датасета
    if 1 in steps_to_run:
        print_banner("ЭТАП 1: ГЕНЕРАЦИЯ ДАТАСЕТА")
        print(f"Будет сгенерировано {args.samples} конфигураций крыла.")
        print("Каждая будет оценена в AeroSandbox (это может занять время)...\n")
        
        cmd = [
            sys.executable, "-m", "SCAT.AERO.wing_optimizer",
            "--samples", str(args.samples),
            "--output", "wing_dataset.jsonl",
            "--velocity", "50.0",
            "--target-cl", "0.55",
            "--min-wing-area", "0.3"
        ]
        
        if not run_command(cmd, "Генерация датасета"):
            print("Остановка.")
            return 1
        
        print("✅ Датасет готов: wing_dataset.jsonl\n")
    
    # Этап 2: Анализ датасета
    if 2 in steps_to_run:
        print_banner("ЭТАП 2: АНАЛИЗ ДАТАСЕТА")
        print("Анализируем распределение параметров и создаём графики...\n")
        
        cmd = [
            sys.executable, "-m", "SCAT.AERO.analyze_wing",
            "--dataset", "wing_dataset.jsonl",
            "--plot",
            "--top-k", "10"
        ]
        
        if not run_command(cmd, "Анализ датасета"):
            print("Остановка.")
            return 1
        
        print("✅ Анализ готов: dataset_analysis.png\n")
    
    # Этап 3: Обучение ML модели
    if 3 in steps_to_run:
        print_banner("ЭТАП 3: ОБУЧЕНИЕ ML МОДЕЛИ")
        print(f"Обучаем нейросеть на {args.epochs} эпохах...")
        print("Это будет быстро на GPU и медленно на CPU.\n")
        
        cmd = [
            sys.executable, "-m", "SCAT.AERO.ml_wing_model",
            "train",
            "--dataset", "wing_dataset.jsonl",
            "--output", "wing_model.keras",
            "--epochs", str(args.epochs),
            "--batch-size", "32"
        ]
        
        if not run_command(cmd, "Обучение модели"):
            print("Остановка.")
            return 1
        
        print("✅ Модель обучена: wing_model.keras\n")
    
    # Этап 4: Анализ модели
    if 4 in steps_to_run:
        print_banner("ЭТАП 4: ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
        print("Проверяем, насколько хорошо модель предсказывает скоры...\n")
        
        cmd = [
            sys.executable, "-m", "SCAT.AERO.ml_wing_model",
            "analyze",
            "--model", "wing_model.keras",
            "--dataset", "wing_dataset.jsonl"
        ]
        
        if not run_command(cmd, "Анализ модели"):
            print("Остановка.")
            return 1
        
        print("✅ Анализ завершён\n")
    
    # Этап 5: Оптимизация
    if 5 in steps_to_run:
        print_banner("ЭТАП 5: ОПТИМИЗАЦИЯ КРЫЛА")
        print("Ищем лучшую конфигурацию крыла используя обученную модель...")
        print("1. Сгенерируем 5000 кандидатов")
        print("2. Быстро оценим их нейросетью")
        print("3. Детально оценим 50 лучших в AeroSandbox")
        print("4. Вернём лучший результат\n")
        
        cmd = [
            sys.executable, "-m", "SCAT.AERO.optimize_wing",
            "--dataset", "wing_dataset.jsonl",
            "--model", "wing_model.keras",
            "--proposals", "5000",
            "--evaluate-top-k", "50",
            "--skip-dataset",
            "--skip-training"
        ]
        
        if not run_command(cmd, "Оптимизация"):
            print("Остановка.")
            return 1
        
        print("✅ Оптимизация завершена!\n")
    
    # Итоговый отчет
    print_banner("✅ ВСЁ ГОТОВО!")
    print("""
    📁 Созданные файлы:
    
    1. wing_dataset.jsonl
       └─ Датасет с 150+ конфигурациями и их оценками
    
    2. dataset_analysis.png
       └─ Графики анализа распределения параметров
    
    3. wing_model.keras
       └─ Обученная нейросеть для предсказания оптимальности
    
    4. optimized_wing.json
       └─ JSON конфигурация лучшей найденной геометрии
    
    5. optimized_wing.py
       └─ Python код для визуализации оптимального крыла
    
    6. optimized_params.json
       └─ Параметры оптимизации в JSON формате
    
    
    🎯 Следующие шаги:
    
    1️⃣  Откройте optimized_wing.py для визуализации крыла:
        python3 optimized_wing.py
    
    2️⃣  Посмотрите на результаты в optimized_params.json
    
    3️⃣  При необходимости измените параметры в optimize_wing.py
        и запустите снова с --skip-dataset --skip-training
    
    
    💡 Для дальнейшего улучшения:
    
    • Увеличьте размер датасета (--samples 300+)
    • Обучите модель дольше (--epochs 150+)
    • Увеличьте число кандидатов для оптимизации (--proposals 10000+)
    • Проанализируйте результаты с analyse_wing.py --plot
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
