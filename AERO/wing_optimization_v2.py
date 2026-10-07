"""
Улучшенная система оптимизации крыла с несколькими режимами.

Режимы:
1. Algorithm - чистая генетическая оптимизация (CEM)
2. AI - нейросетевая оптимизация (если модель доступна)
3. AI+Algorithm - гибридный подход: AI для быстрого поиска + Algorithm для уточнения
"""

from __future__ import annotations

import json
import random
import copy
import statistics
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import SCAT.AERO.main as main
import SCAT.AERO.optimizer as base_opt
import SCAT.AERO.wing_optimizer as wing_opt
import SCAT.AERO.control_v2 as control_v2


# =============================================================================
# 🔧 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ УЛУЧШЕННОЙ ГЕНЕРАЦИИ
# =============================================================================

def sample_random_vector_around_baseline(
    rng: np.random.Generator,
    baseline: Optional[np.ndarray] = None,
    noise_scale: float = 0.15,
) -> np.ndarray:
    """
    Генерировать вектор с небольшим гауссовым шумом вокруг базового дизайна.
    Увеличивает шанс получить валидную конфигурацию.
    """
    if baseline is None:
        baseline = base_opt.baseline_vector()
    noise = rng.normal(0, noise_scale, size=len(base_opt.PARAMETER_SPECS))
    candidate = baseline + noise
    return base_opt.clamp_vector(candidate)


def analyze_constraint_violations(evaluations: List[wing_opt.DesignEvaluation]) -> dict:
    """Анализировать, какие ограничения нарушаются чаще всего."""
    violations = {
        "wing_area": 0,
        "Clp": 0,      # roll damping
        "Cnr": 0,      # yaw damping
        "Cma": 0,      # pitch stiffness
        "total": 0,
    }
    for ev in evaluations:
        if not ev.is_valid():
            violations["total"] += 1
            if ev.wing.area < 0.3:  # пример порога
                violations["wing_area"] += 1
            if ev.stability.Clp >= -0.01:
                violations["Clp"] += 1
            if ev.stability.Cnr >= -0.01:
                violations["Cnr"] += 1
            if ev.stability.Cma >= 0:  # должен быть отрицательным
                violations["Cma"] += 1
    return violations


# =============================================================================
# 🎯 ОСНОВНОЙ ДВИЖОК ОПТИМИЗАЦИИ
# =============================================================================

class WingOptimizationEngine:
    """Единая система оптимизации крыла с несколькими режимами."""
    
    def __init__(
        self,
        base_config: main.AircraftConfig,
        relax_mode: bool = False,
        baseline_focused: bool = True,
    ):
        self.base_config = copy.deepcopy(base_config)
        self.specs_v2 = control_v2.build_parameter_specs_for_config(self.base_config)
        self.max_wing_span_limit = wing_opt.extract_wing_metrics(self.base_config.wing).span
        self.model = None
        self.has_model = False
        self.model_kind = None
        self._v2_y_mean = 0.0
        self._v2_y_std = 1.0
        self.relax_mode = relax_mode
        self.baseline_focused = baseline_focused
        
        # Попытаться загрузить ML модель
        try:
            from tensorflow import keras  # type: ignore
            import SCAT.AERO.ml_wing_model_v2 as ml_model_v2
            model_v2_path = Path(__file__).parent / "wing_model_v2.keras"
            model_v1_path = Path(__file__).parent / "wing_model.keras"
            if model_v2_path.exists():
                self.model = keras.models.load_model(str(model_v2_path), compile=False)
                self.has_model = True
                self.model_kind = "v2"
                meta = ml_model_v2.load_v2_training_meta(str(model_v2_path))
                if meta:
                    self._v2_y_mean = float(meta.get("y_mean", 0.0))
                    self._v2_y_std = float(meta.get("y_std", 1.0))
                print("✓ ML v2 модель загружена")
            elif model_v1_path.exists():
                import SCAT.AERO.ml_wing_model as ml_model
                self.model = ml_model.load_trained_model(str(model_v1_path))
                self.has_model = True
                self.model_kind = "v1"
                print("✓ ML v1 модель загружена")
        except Exception as e:
            print(f"ℹ ML модель недоступна: {e}")

    def _fuselage_signature(self, config: main.AircraftConfig):
        main_fus = tuple((round(s.x, 8), round(s.y, 8), round(s.z, 8), round(s.radius, 8)) for s in config.fuselage_sections)
        extra = tuple(
            tuple((round(s.x, 8), round(s.y, 8), round(s.z, 8), round(s.radius, 8)) for s in chain)
            for chain in config.extra_fuselages
        )
        return main_fus, extra

    def _assert_fuselage_locked(self, config: main.AircraftConfig) -> None:
        if self._fuselage_signature(config) != self._fuselage_signature(self.base_config):
            raise RuntimeError("Обнаружено изменение фюзеляжа: это запрещено правилами оптимизации.")
    
    def optimize(
        self,
        mode: str = "algorithm",
        iterations: int = 10,
        population: int = 20,
        velocity: float = 50.0,
        target_cl: float = 0.55,
        min_wing_area: float = 0.3,
        seed: int = 42,
        relax_constraints: Optional[bool] = None,  # переопределить при вызове
    ) -> dict:
        """
        Оптимизировать конфигурацию крыла в выбранном режиме.
        
        Args:
            relax_constraints: если True — временно ослабить ограничения
                               (полезно при генерации датасета)
        """
        # Локальное переопределение relax_mode
        current_relax = relax_constraints if relax_constraints is not None else self.relax_mode
        
        if mode == "algorithm":
            return self._optimize_algorithm(
                iterations, population, velocity, target_cl, min_wing_area, seed,
                relax_constraints=current_relax,
            )
        elif mode == "ai":
            if not self.has_model:
                return {
                    "error": "ML модель не доступна. Используйте режим 'algorithm' или 'ai+algorithm'.",
                    "mode": "ai",
                }
            return self._optimize_ai(
                iterations, population, velocity, target_cl, min_wing_area, seed,
                relax_constraints=current_relax,
            )
        elif mode == "ai+algorithm":
            return self._optimize_hybrid(
                iterations, population, velocity, target_cl, min_wing_area, seed,
                relax_constraints=current_relax,
            )
        else:
            return {"error": f"Неизвестный режим: {mode}"}
    
    def _get_evaluation_params(self, velocity, target_cl, min_wing_area, relax_constraints: bool):
        """Возвращает параметры оценки с возможным ослаблением ограничений."""
        if relax_constraints:
            # Ослабляем ограничения на 30-50% для поиска "почти валидных" конфигураций
            return {
                "velocity": velocity,
                "target_cl": target_cl * 0.85,      # легче достичь
                "min_wing_area": min_wing_area * 0.6,  # разрешаем меньшие крылья
                "max_wing_span": self.max_wing_span_limit,
            }
        return {
            "velocity": velocity,
            "target_cl": target_cl,
            "min_wing_area": min_wing_area,
            "max_wing_span": self.max_wing_span_limit,
        }
    
    def _optimize_algorithm(
        self,
        iterations: int,
        population: int,
        velocity: float,
        target_cl: float,
        min_wing_area: float,
        seed: int,
        relax_constraints: bool = False,
    ) -> dict:
        """Алгоритмическая оптимизация (CEM) на целевой функции wing_optimizer."""
        
        print(f"\n🧬 РЕЖИМ: АЛГОРИТМ (CEM)")
        print(f"   Итерации: {iterations}, Популяция: {population}")
        if relax_constraints:
            print(f"   ⚠️  Ограничения ОСЛАБЛЕНЫ (relax_mode)")
        
        best_vector, best_evaluation = self._cem_optimize_with_constraints(
            start_vector=base_opt.baseline_vector(),
            iterations=iterations,
            population=population,
            velocity=velocity,
            target_cl=target_cl,
            min_wing_area=min_wing_area,
            seed=seed,
            elite_fraction=0.25,
            relax_constraints=relax_constraints,
        )

        optimized_config = base_opt.apply_design_vector(self.base_config, best_vector)
        return self._build_result_payload(
            mode="algorithm",
            optimized_config=optimized_config,
            best_vector=best_vector,
            best_evaluation=best_evaluation,
            min_wing_area=min_wing_area,
        )

    def _cem_optimize_with_constraints(
        self,
        start_vector: np.ndarray,
        iterations: int,
        population: int,
        velocity: float,
        target_cl: float,
        min_wing_area: float,
        seed: int,
        elite_fraction: float = 0.25,
        relax_constraints: bool = False,
    ) -> tuple[np.ndarray, wing_opt.DesignEvaluation]:
        rng = np.random.default_rng(seed)
        mean = base_opt.clamp_vector(np.array(start_vector, dtype=float))
        sigmas = np.array([(spec.high - spec.low) / 4.0 for spec in base_opt.PARAMETER_SPECS], dtype=float)
        elite_count = max(2, int(population * elite_fraction))

        # Начальная точка — базовый вектор (обычно валидный)
        best_vector = mean.copy()
        eval_params = self._get_evaluation_params(velocity, target_cl, min_wing_area, relax_constraints)
        best_evaluation = wing_opt.evaluate_wing_design(
            base_opt.apply_design_vector(self.base_config, best_vector),
            **eval_params,
        )

        for iteration in range(1, iterations + 1):
            population_vectors = [best_vector.copy()]
            
            while len(population_vectors) < population:
                # 🔧 УЛУЧШЕНИЕ: 70% — локальный сэмплинг, 30% — глобальный
                if self.baseline_focused and rng.random() < 0.7:
                    sample = sample_random_vector_around_baseline(
                        rng, baseline=mean, noise_scale=0.12
                    )
                else:
                    sample = rng.normal(mean, sigmas)
                population_vectors.append(base_opt.clamp_vector(sample))

            ranked = []
            eval_params = self._get_evaluation_params(velocity, target_cl, min_wing_area, relax_constraints)
            
            for vector in population_vectors:
                config = base_opt.apply_design_vector(self.base_config, vector)
                evaluation = wing_opt.evaluate_wing_design(config, **eval_params)
                ranked.append((evaluation.score, vector, evaluation))

            ranked.sort(key=lambda item: item[0], reverse=True)
            elites = ranked[:elite_count]
            elite_vectors = np.array([item[1] for item in elites], dtype=float)

            mean = elite_vectors.mean(axis=0)
            sigmas = np.maximum(elite_vectors.std(axis=0), 0.03)

            if elites[0][0] > best_evaluation.score:
                best_vector = elites[0][1].copy()
                best_evaluation = elites[0][2]

            # Логирование с анализом валидности
            valid_count = sum(
                1 for _, _, ev in ranked if ev.is_valid(min_wing_area=min_wing_area, max_wing_span=self.max_wing_span_limit)
            )
            iteration_scores = [item[0] for item in ranked]
            print(
                f"[opt-v2] iter={iteration:02d}/{iterations} "
                f"best={best_evaluation.score:.3f} "
                f"iter_best={ranked[0][0]:.3f} "
                f"iter_mean={statistics.mean(iteration_scores):.3f} "
                f"valid={valid_count}/{population}"
            )

            # 🔧 Адаптивное сужение поиска, если все невалидны
            if valid_count == 0 and iteration % 5 == 0:
                print(f"   ⚠️  Все конфигурации невалидны — сужаем поиск вокруг лучшего")
                sigmas *= 0.7  # уменьшаем разброс

        return best_vector, best_evaluation

    def _build_result_payload(
        self,
        mode: str,
        optimized_config: main.AircraftConfig,
        best_vector: np.ndarray,
        best_evaluation: wing_opt.DesignEvaluation,
        min_wing_area: float,
    ) -> dict:
        return {
            "mode": mode,
            "optimized_config": main.config_to_dict(optimized_config),
            "evaluation": {
                "mode": mode,
                "score": float(best_evaluation.score),
                "efficiency": float(best_evaluation.efficiency),
                "CL": float(best_evaluation.stability.Cl),
                "CD": float(best_evaluation.stability.Cd),
                "Cm": float(best_evaluation.stability.Cm),
                "Cma": float(best_evaluation.stability.Cma),
                "Clp": float(best_evaluation.stability.Clp),
                "Cnr": float(best_evaluation.stability.Cnr),
                "wing_area": float(best_evaluation.wing.area),
                "constraints_violation": float(best_evaluation.constraints_violation),
                "is_valid": best_evaluation.is_valid(
                    min_wing_area=min_wing_area,
                    max_wing_span=self.max_wing_span_limit,
                ),
                "fuselage_locked": True,
            },
            "chart6": wing_opt.stability_chart6_payload(
                best_evaluation,
                min_wing_area=min_wing_area,
                max_wing_span=self.max_wing_span_limit,
            ),
            "parameters": base_opt.vector_to_dict(best_vector),
        }

    def _build_result_payload_v2(
        self,
        mode: str,
        optimized_config: main.AircraftConfig,
        best_vector: np.ndarray,
        best_evaluation: wing_opt.DesignEvaluation,
        min_wing_area: float,
    ) -> dict:
        return {
            "mode": mode,
            "optimized_config": main.config_to_dict(optimized_config),
            "evaluation": {
                "mode": mode,
                "score": float(best_evaluation.score),
                "efficiency": float(best_evaluation.efficiency),
                "CL": float(best_evaluation.stability.Cl),
                "CD": float(best_evaluation.stability.Cd),
                "Cm": float(best_evaluation.stability.Cm),
                "Cma": float(best_evaluation.stability.Cma),
                "Clp": float(best_evaluation.stability.Clp),
                "Cnr": float(best_evaluation.stability.Cnr),
                "wing_area": float(best_evaluation.wing.area),
                "constraints_violation": float(best_evaluation.constraints_violation),
                "is_valid": best_evaluation.is_valid(
                    min_wing_area=min_wing_area,
                    max_wing_span=self.max_wing_span_limit,
                ),
                "fuselage_locked": True,
            },
            "chart6": wing_opt.stability_chart6_payload(
                best_evaluation,
                min_wing_area=min_wing_area,
                max_wing_span=self.max_wing_span_limit,
            ),
            "parameters": control_v2.vector_to_dict(self.specs_v2, best_vector),
        }

    def _cem_optimize_v2(
        self,
        start_vector: np.ndarray,
        iterations: int,
        population: int,
        velocity: float,
        target_cl: float,
        min_wing_area: float,
        seed: int,
        elite_fraction: float = 0.25,
        relax_constraints: bool = False,
    ) -> tuple[np.ndarray, wing_opt.DesignEvaluation]:
        rng = np.random.default_rng(seed)
        mean = control_v2.clamp_vector(self.specs_v2, np.array(start_vector, dtype=float))
        sigmas = np.array([(spec.high - spec.low) / 5.0 for spec in self.specs_v2], dtype=float)
        elite_count = max(2, int(population * elite_fraction))
        eval_params = self._get_evaluation_params(velocity, target_cl, min_wing_area, relax_constraints)

        best_vector = mean.copy()
        best_cfg = control_v2.apply_vector_to_config(self.base_config, self.specs_v2, best_vector)
        self._assert_fuselage_locked(best_cfg)
        best_evaluation = wing_opt.evaluate_wing_design(best_cfg, **eval_params)

        for iteration in range(1, iterations + 1):
            population_vectors = [best_vector.copy()]
            mutation_prob = max(0.08, 0.30 - 0.02 * iteration)
            mutation_scale = max(0.02, 0.14 - 0.01 * iteration)
            crossover_count = max(2, population // 5)

            # Эволюционное скрещивание вокруг текущего лучшего решения.
            for _ in range(crossover_count):
                mate = control_v2.clamp_vector(self.specs_v2, rng.normal(mean, sigmas))
                alpha = rng.uniform(0.25, 0.75, size=len(self.specs_v2))
                child = alpha * best_vector + (1.0 - alpha) * mate
                # Редкие целевые мутации для выхода из локального минимума.
                mut_mask = rng.random(len(self.specs_v2)) < mutation_prob
                if np.any(mut_mask):
                    child[mut_mask] += rng.normal(0.0, mutation_scale, size=np.sum(mut_mask))
                population_vectors.append(control_v2.clamp_vector(self.specs_v2, child))

            while len(population_vectors) < population:
                sample = rng.normal(mean, sigmas)
                population_vectors.append(control_v2.clamp_vector(self.specs_v2, sample))

            ranked = []
            for vector in population_vectors:
                cfg = control_v2.apply_vector_to_config(self.base_config, self.specs_v2, vector)
                self._assert_fuselage_locked(cfg)
                evaluation = wing_opt.evaluate_wing_design(cfg, **eval_params)
                # Упор на эффективность: слегка поднимаем вклад L/D в ранжировании.
                ranking_score = evaluation.score + 0.25 * evaluation.efficiency
                ranked.append((ranking_score, vector, evaluation))

            ranked.sort(key=lambda item: item[0], reverse=True)
            elites = ranked[:elite_count]
            elite_vectors = np.array([item[1] for item in elites], dtype=float)
            mean = elite_vectors.mean(axis=0)
            sigmas = np.maximum(elite_vectors.std(axis=0), 0.02)

            if elites[0][2].score > best_evaluation.score:
                best_vector = elites[0][1].copy()
                best_evaluation = elites[0][2]

            print(
                f"[opt-v2-ai] iter={iteration:02d}/{iterations} "
                f"best={best_evaluation.score:.3f} "
                f"iter_best={ranked[0][2].score:.3f} "
                f"iter_eff={ranked[0][2].efficiency:.3f}"
            )

        return best_vector, best_evaluation
    
    def _optimize_ai(
        self,
        iterations: int,
        population: int,
        velocity: float,
        target_cl: float,
        min_wing_area: float,
        seed: int,
        relax_constraints: bool = False,
    ) -> dict:
        """Чистая AI оптимизация (нейросеть)."""
        
        if not self.has_model:
            return {"error": "ML модель не загружена"}
        
        print(f"\n🤖 РЕЖИМ: AI (НЕЙРОСЕТЬ)")
        print(f"   Предложений: {population * 50}, Оценить топ: {max(5, population // 4)}")
        if relax_constraints:
            print(f"   ⚠️  Ограничения ОСЛАБЛЕНЫ (relax_mode)")
        
        try:
            eval_params = self._get_evaluation_params(velocity, target_cl, min_wing_area, relax_constraints)
            if self.model_kind == "v2":
                import SCAT.AERO.ml_wing_model_v2 as ml_model_v2

                # AI этап: широкий поиск по нейросети.
                ai_vector, ai_eval = ml_model_v2.optimize_with_neural_network_v2(
                    model=self.model,
                    base_config=self.base_config,
                    proposal_count=population * 80,
                    evaluate_top_k=max(8, population // 2),
                    seed=seed,
                    velocity=eval_params["velocity"],
                    target_cl=eval_params["target_cl"],
                    y_mean=self._v2_y_mean,
                    y_std=self._v2_y_std,
                )
                # Уточнение: локальный CEM вокруг лучшего AI-кандидата.
                best_vector, best_evaluation = self._cem_optimize_v2(
                    start_vector=ai_vector,
                    iterations=max(2, iterations // 2),
                    population=max(10, population),
                    velocity=velocity,
                    target_cl=target_cl,
                    min_wing_area=min_wing_area,
                    seed=seed + 1,
                    relax_constraints=relax_constraints,
                )
                if ai_eval.score > best_evaluation.score:
                    best_vector, best_evaluation = ai_vector, ai_eval

                optimized_config = control_v2.apply_vector_to_config(self.base_config, self.specs_v2, best_vector)
                self._assert_fuselage_locked(optimized_config)
                return self._build_result_payload_v2(
                    mode="ai",
                    optimized_config=optimized_config,
                    best_vector=best_vector,
                    best_evaluation=best_evaluation,
                    min_wing_area=min_wing_area,
                )

            import SCAT.AERO.ml_wing_model as ml_model
            best_vector, best_evaluation = ml_model.optimize_with_neural_network(
                model=self.model,
                base_config=self.base_config,
                proposal_count=population * 50,
                evaluate_top_k=max(5, population // 4),
                seed=seed,
                **eval_params,
            )
            optimized_config = base_opt.apply_design_vector(self.base_config, best_vector)
            self._assert_fuselage_locked(optimized_config)
            return self._build_result_payload(
                mode="ai",
                optimized_config=optimized_config,
                best_vector=best_vector,
                best_evaluation=best_evaluation,
                min_wing_area=min_wing_area,
            )
        except Exception as e:
            return {"error": f"Ошибка в AI оптимизации: {str(e)}"}
    
    def _optimize_hybrid(
        self,
        iterations: int,
        population: int,
        velocity: float,
        target_cl: float,
        min_wing_area: float,
        seed: int,
        relax_constraints: bool = False,
    ) -> dict:
        """Гибридная оптимизация: AI для быстрого поиска + Algorithm для уточнения."""
        
        print(f"\n🔄 РЕЖИМ: ГИБРИДНЫЙ (AI + АЛГОРИТМ)")
        print(f"   Этап 1 (AI): поиск перспективных регионов")
        print(f"   Этап 2 (CEM): локальное уточнение")
        if relax_constraints:
            print(f"   ⚠️  Ограничения ОСЛАБЛЕНЫ (relax_mode)")
        
        results = {}
        start_vector_legacy = base_opt.baseline_vector()
        start_vector_v2 = control_v2.baseline_vector_from_specs(self.specs_v2)
        
        # Этап 1: AI поиск
        if self.has_model:
            print("\n   → Запуск AI оптимизации...")
            ai_result = self._optimize_ai(
                iterations=1,
                population=population // 2,
                velocity=velocity,
                target_cl=target_cl,
                min_wing_area=min_wing_area,
                seed=seed,
                relax_constraints=relax_constraints,
            )
            
            if "error" not in ai_result:
                results["ai_stage"] = ai_result
                if self.model_kind == "v2":
                    start_vector_v2 = np.array(
                        [float(ai_result["parameters"].get(spec.name, spec.baseline)) for spec in self.specs_v2],
                        dtype=float,
                    )
                else:
                    start_vector_legacy = np.array(
                        [float(ai_result["parameters"].get(spec.name, spec.baseline)) for spec in base_opt.PARAMETER_SPECS],
                        dtype=float,
                    )
                print(f"   ✓ AI найдена перспективная конфигурация (score={ai_result['evaluation']['score']:.3f})")
        
        # Этап 2: Algorithm уточнение (CEM локальный поиск)
        print("\n   → Запуск локального уточнения (CEM)...")
        if self.model_kind == "v2":
            best_vector, best_evaluation = self._cem_optimize_v2(
                start_vector=start_vector_v2,
                iterations=iterations // 2 + 1,
                population=population,
                velocity=velocity,
                target_cl=target_cl,
                min_wing_area=min_wing_area,
                seed=seed,
                relax_constraints=relax_constraints,
            )
            optimized_config = control_v2.apply_vector_to_config(self.base_config, self.specs_v2, best_vector)
            self._assert_fuselage_locked(optimized_config)
            cem_result = self._build_result_payload_v2(
                mode="ai+algorithm",
                optimized_config=optimized_config,
                best_vector=best_vector,
                best_evaluation=best_evaluation,
                min_wing_area=min_wing_area,
            )
        else:
            best_vector, best_evaluation = self._cem_optimize_with_constraints(
                start_vector=start_vector_legacy,
                iterations=iterations // 2 + 1,
                population=population,
                velocity=velocity,
                target_cl=target_cl,
                min_wing_area=min_wing_area,
                seed=seed,
                relax_constraints=relax_constraints,
            )
            optimized_config = base_opt.apply_design_vector(self.base_config, best_vector)
            self._assert_fuselage_locked(optimized_config)
            cem_result = self._build_result_payload(
                mode="ai+algorithm",
                optimized_config=optimized_config,
                best_vector=best_vector,
                best_evaluation=best_evaluation,
                min_wing_area=min_wing_area,
            )

        results["algorithm_stage"] = cem_result
        results["mode"] = "ai+algorithm"
        results["optimized_config"] = cem_result["optimized_config"]
        results["evaluation"] = cem_result["evaluation"]
        results["parameters"] = cem_result["parameters"]
        if "chart6" in cem_result:
            results["chart6"] = cem_result["chart6"]

        return results


# =============================================================================
# 📊 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def evaluate_current_design(
    config: main.AircraftConfig,
    velocity: float = 50.0,
    target_cl: float = 0.55,
    min_wing_area: float = 0.3,
    relax_constraints: bool = False,
) -> dict:
    """Оценить текущую конфигурацию крыла."""
    try:
        eval_params = {
            "velocity": velocity,
            "target_cl": target_cl * 0.85 if relax_constraints else target_cl,
            "min_wing_area": min_wing_area * 0.6 if relax_constraints else min_wing_area,
            "max_wing_span": wing_opt.extract_wing_metrics(config.wing).span,
        }
        evaluation = wing_opt.evaluate_wing_design(config, **eval_params)
        
        return {
            "score": float(evaluation.score),
            "efficiency": float(evaluation.efficiency),
            "CL": float(evaluation.stability.Cl),
            "CD": float(evaluation.stability.Cd),
            "Cm": float(evaluation.stability.Cm),
            "Cma": float(evaluation.stability.Cma),
            "Cmq": float(evaluation.stability.Cmq),
            "Clp": float(evaluation.stability.Clp),
            "Cnr": float(evaluation.stability.Cnr),
            "wing_area": float(evaluation.wing.area),
            "wing_span": float(evaluation.wing.span),
            "wing_aspect_ratio": float(evaluation.wing.aspect_ratio),
            "geometry_penalty": float(evaluation.geometry_penalty),
            "constraints_violation": float(evaluation.constraints_violation),
            "is_valid": evaluation.is_valid(
                min_wing_area=min_wing_area,
                max_wing_span=eval_params["max_wing_span"],
            ),
            "stability_status": {
                "roll_stable": evaluation.stability.is_stable_roll(),
                "yaw_stable": evaluation.stability.is_stable_yaw(),
                "pitch_neutral": abs(evaluation.stability.Cm) < 0.05,
            }
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# 🚀 ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Улучшенная оптимизация крыла с несколькими режимами")
    parser.add_argument("--mode", type=str, default="algorithm", choices=["algorithm", "ai", "ai+algorithm"],
                       help="Режим оптимизации")
    parser.add_argument("--config", type=str, default=None, help="JSON конфигурация")
    parser.add_argument("--iterations", type=int, default=10, help="Итерации")
    parser.add_argument("--population", type=int, default=20, help="Популяция")
    parser.add_argument("--velocity", type=float, default=50.0, help="Скорость")
    parser.add_argument("--target-cl", type=float, default=0.55, help="Целевой CL")
    parser.add_argument("--min-wing-area", type=float, default=0.3, help="Минимум площади")
    parser.add_argument("--relax", action="store_true", help="Ослабить ограничения (для генерации датасета)")
    parser.add_argument("--no-baseline-focus", action="store_true", help="Отключить локальный сэмплинг")
    
    args = parser.parse_args()
    
    base_config = base_opt.load_config(args.config)
    engine = WingOptimizationEngine(
        base_config,
        relax_mode=args.relax,
        baseline_focused=not args.no_baseline_focus,
    )
    
    result = engine.optimize(
        mode=args.mode,
        iterations=args.iterations,
        population=args.population,
        velocity=args.velocity,
        target_cl=args.target_cl,
        min_wing_area=args.min_wing_area,
        relax_constraints=args.relax,
    )
    
    print("\n" + "="*70)
    print(json.dumps(result, indent=2, ensure_ascii=False))