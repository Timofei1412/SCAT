"""
Расширенная система оптимизации крыла с учетом аэродинамической стабильности.
Включает ML-модель для прямого предсказания оптимальной геометрии.

ИСПРАВЛЕНИЯ:
- Нормализация экстремальных производных (Cma, Cmq, Cm) перед штрафами
- Мягкие штрафы вместо экстремальных
- is_valid() требует устойчивость по крену (Clp) и рысканью (Cnr) с порогом 0.01
- Параметр relax_mode для генерации датасета
- Линейный клиппинг вместо tanh для сохранения градиентов ML
- ✅ ПРАВИЛО 1: Площадь крыла должна быть ≥ 120% от площади ПГО
- ✅ ПРАВИЛО 2: Максимальный размах крыла ≤ 2.0 метра
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import SCAT.AERO.main as main
import SCAT.AERO.optimizer as base_opt


@dataclass(frozen=True)
class StabilityMetrics:
    """Метрики аэродинамической стабильности."""
    # Производные по углу атаки (pitch)
    Cma: float  # pitch stiffness
    Cmq: float  # pitch damping
    
    # Производные по крену (roll)
    Clp: float  # roll damping
    Clr: float  # roll yaw-coupling
    
    # Производные по рысканью (yaw)
    Cnr: float  # yaw damping
    Cnp: float  # yaw roll-coupling
    
    # Статические коэффициенты
    Cm: float   # pitch moment
    Cl: float   # lift coefficient
    Cd: float   # drag coefficient
    
    def is_stable_roll(self, tolerance: float = 0.01) -> bool:
        """Стабилен ли крен: Clp должен быть отрицательным (демпфирование)."""
        return self.Clp < -tolerance
    
    def is_stable_yaw(self, tolerance: float = 0.01) -> bool:
        """Стабилен ли рысканье: для этой схемы Cnr ожидается отрицательным."""
        return self.Cnr < -tolerance
    
    def pitch_stability(self, relax: bool = False) -> float:
        """
        Штраф по тангажу.
        Для канардной схемы требуем "максимально близко к нейтрали".
        """
        factor = 0.5 if relax else 1.0
        neutral_error = abs(self.Cma)
        unstable_bias = max(0.0, self.Cma) * 2.0
        return (neutral_error * 18.0 + unstable_bias * 24.0) * factor
    
    def pitch_damping(self, relax: bool = False) -> float:
        """Демпфирование по тангажу: Cmq должен быть отрицательным."""
        factor = 0.5 if relax else 1.0
        return max(0.0, self.Cmq) * 30.0 * factor


@dataclass
class WingMetrics:
    """Метрики геометрии крыла."""
    area: float
    aspect_ratio: float
    span: float
    root_chord: float
    tip_chord: float
    mean_chord: float
    sweep: float
    dihedral: float


@dataclass
class DesignEvaluation:
    """Результат оценки одной конфигурации."""
    score: float
    efficiency: float
    stability: StabilityMetrics
    wing: WingMetrics
    canard: WingMetrics
    geometry_penalty: float
    constraints_violation: float
    max_wing_span_limit: float = 2.0
    
    def is_valid(
        self,
        min_wing_area: float = 0.3,
        max_geometry_penalty: float = 50.0,
        max_constraints_violation: float = 15.0,
        max_wing_span: float | None = None,
    ) -> bool:
        """
        Проверить, удовлетворяет ли конфиг минимальным требованиям.
        Включая демпфирование крена (Clp < -0.01) и рысканья (Cnr < -0.01), как в StabilityMetrics.
        """
        # 1. Минимальная площадь крыла
        if self.wing.area < min_wing_area * 0.8:
            return False
            
        # 🔧 ПРАВИЛО 1: Площадь крыла должна быть ≥ 120% от площади ПГО
        if self.canard.area > 0 and self.wing.area < self.canard.area * 1.2:
            return False
            
        # 🔧 ПРАВИЛО 2: Максимальный размах крыла ограничен базовой конфигурацией.
        span_limit = self.max_wing_span_limit if max_wing_span is None else max_wing_span
        if self.wing.span > span_limit + 1e-9:
            return False
            
        # 2. Геометрические ограничения
        if self.geometry_penalty > max_geometry_penalty:
            return False

        # 3. Статическая устойчивость по крену и рысканью (иначе весь датасет «валиден», но не демпфируется)
        if not self.stability.is_stable_roll(tolerance=0.01):
            return False
        if not self.stability.is_stable_yaw(tolerance=0.01):
            return False

        # 4. Прочие мягкие ограничения (площадь, связки и т.д.)
        if self.constraints_violation > max_constraints_violation:
            return False
        return True


def stability_chart6_payload(
    evaluation: DesignEvaluation,
    *,
    min_wing_area: float = 0.3,
    max_wing_span: float | None = None,
) -> dict:
    """
    Данные для «графика 6» как в analyze_wing.plot_dataset_analysis (axes[1, 2]):
    столбцы 0/1 по образцу — аналог категорий «Все / Валидные / Площадь / Крен / Рысканье»
    плюс проверка тангажа по Cma для полной картины статической устойчивости.
    """
    st = evaluation.stability
    span_lim = max_wing_span if max_wing_span is not None else evaluation.max_wing_span_limit
    full_valid = evaluation.is_valid(min_wing_area=min_wing_area, max_wing_span=span_lim)
    area_ok = evaluation.wing.area >= min_wing_area * 0.8
    roll_ok = st.is_stable_roll()
    yaw_ok = st.is_stable_yaw()
    pitch_ok = st.Cma < -0.01
    categories = ["Образец", "Валидные", "Площадь OK", "Крен OK", "Рысканье OK", "Тангаж OK"]
    values = [1, int(full_valid), int(area_ok), int(roll_ok), int(yaw_ok), int(pitch_ok)]
    return {
        "categories": categories,
        "values": values,
        "metrics": {
            "Clp": st.Clp,
            "Cnr": st.Cnr,
            "Cma": st.Cma,
            "Cm": st.Cm,
            "CL": st.Cl,
            "CD": st.Cd,
            "wing_area": evaluation.wing.area,
            "wing_span": evaluation.wing.span,
            "geometry_penalty": evaluation.geometry_penalty,
            "constraints_violation": evaluation.constraints_violation,
            "is_valid": full_valid,
        },
    }


def calculate_sweep_angle(sections: list[main.SurfaceSection]) -> float:
    """Рассчитать средний угол стреловидности в градусах."""
    if len(sections) < 2:
        return 0.0
    
    xs = np.array([s.x for s in sections])
    ys = np.array([s.y for s in sections])
    
    if np.std(ys) < 1e-9:
        return 0.0
    
    coeffs = np.polyfit(ys, xs, 1)
    sweep_rad = np.arctan(coeffs[0])
    return float(np.degrees(sweep_rad))


def calculate_dihedral_angle(sections: list[main.SurfaceSection]) -> float:
    """Рассчитать диэдральный угол в градусах."""
    if len(sections) < 2:
        return 0.0
    
    ys = np.array([s.y for s in sections])
    zs = np.array([s.z for s in sections])
    
    if np.std(ys) < 1e-9:
        return 0.0
    
    coeffs = np.polyfit(ys, zs, 1)
    dihedral_rad = np.arctan(coeffs[0])
    return float(np.degrees(dihedral_rad))


def extract_wing_metrics(surface: main.SurfaceConfig) -> WingMetrics:
    """Извлечь метрики геометрии из поверхности."""
    area = base_opt.surface_planform_area(surface)
    ar = base_opt.surface_aspect_ratio(surface)
    span = base_opt.surface_half_span(surface) * (2.0 if surface.symmetric else 1.0)
    mc = base_opt.surface_mean_chord(surface)
    
    root = surface.sections[0] if surface.sections else main.SurfaceSection(0, 0, 0, 0.1, 0)
    tip = surface.sections[-1] if surface.sections else main.SurfaceSection(0, 0, 0, 0.1, 0)
    
    sweep = calculate_sweep_angle(surface.sections)
    dihedral = calculate_dihedral_angle(surface.sections)
    
    return WingMetrics(
        area=area,
        aspect_ratio=ar,
        span=span,
        root_chord=root.chord,
        tip_chord=tip.chord,
        mean_chord=mc,
        sweep=sweep,
        dihedral=dihedral,
    )


def extract_stability_from_aero(aero: dict) -> StabilityMetrics:
    """Извлечь метрики стабильности из результатов AeroSandbox."""
    return StabilityMetrics(
        Cma=float(np.atleast_1d(aero.get("Cma", 0.0))[0]),
        Cmq=float(np.atleast_1d(aero.get("Cmq", 0.0))[0]),
        Clp=float(np.atleast_1d(aero.get("Clp", 0.0))[0]),
        Clr=float(np.atleast_1d(aero.get("Clr", 0.0))[0]),
        Cnr=float(np.atleast_1d(aero.get("Cnr", 0.0))[0]),
        Cnp=float(np.atleast_1d(aero.get("Cnp", 0.0))[0]),
        Cm=float(np.atleast_1d(aero.get("Cm", 0.0))[0]),
        Cl=float(np.atleast_1d(aero.get("CL", 0.0))[0]),
        Cd=float(np.atleast_1d(aero.get("CD", 0.0))[0]),
    )


# =============================================================================
# 🔧 ПОЛНОСТЬЮ ПЕРЕПИСАННАЯ ФУНКЦИЯ ОЦЕНКИ
# =============================================================================

def evaluate_wing_design(
    config: main.AircraftConfig,
    velocity: float = 50.0,
    target_cl: float = 0.55,
    min_wing_area: float = 0.3,
    max_wing_span: float | None = None,
    alpha: float = 0.0,
    relax_mode: bool = False,
    _debug_idx: int = 0,
) -> DesignEvaluation:
    """
    Оценить конфигурацию крыла.
    🔧 Исправлено: 
       1. Экстремальные значения Cma/Cmq/Cm обрезаются до реалистичных диапазонов
       2. Штрафы рассчитываются на обрезанных значениях
       3. Линейный клиппинг [-15, 30] сохраняет градиенты для ML
       4. Добавлен штраф за нарушение соотношения площадей Крыло/ПГО
       5. ✅ Добавлен штраф за размах крыла > допустимого лимита
    """
    try:
        airplane = main.build_airplane(config)
        op_point = main.asb.OperatingPoint(velocity=velocity, alpha=float(alpha))
        aero = main.asb.AeroBuildup(airplane=airplane, op_point=op_point).run_with_stability_derivatives()
        
        cl = float(np.atleast_1d(aero["CL"])[0])
        cd = float(np.atleast_1d(aero["CD"])[0])
        cm_raw = float(np.atleast_1d(aero["Cm"])[0])
        cma_raw = float(np.atleast_1d(aero.get("Cma", 0.0))[0])
        cmq_raw = float(np.atleast_1d(aero.get("Cmq", 0.0))[0])
        
        stability = extract_stability_from_aero(aero)
        wing_metrics = extract_wing_metrics(config.wing)
        canard_metrics = extract_wing_metrics(config.canard)
        span_limit = max_wing_span if max_wing_span is not None else 2.0
        
        geometry_penalty = base_opt.geometry_penalty(config)
        
        # 🔧 МЯГКИЕ ОГРАНИЧЕНИЯ
        constraints_violation = 0.0
        if wing_metrics.area < min_wing_area:
            constraints_violation += (min_wing_area - wing_metrics.area) * 2.0
        # Крен / рысканье: устойчивость при Clp < -tol и Cnr < -tol (см. StabilityMetrics)
        _tol = 0.01
        if stability.Clp >= -_tol:
            constraints_violation += max(0.0, stability.Clp + _tol) * 18.0
        if stability.Cnr >= -_tol:
            constraints_violation += max(0.0, stability.Cnr + _tol) * 18.0
            
        # 🔧 ПРАВИЛО 1: Штраф за нарушение соотношения площадей Крыло/ПГО
        if canard_metrics.area > 0 and wing_metrics.area < canard_metrics.area * 1.2:
            ratio = wing_metrics.area / max(canard_metrics.area, 1e-6)
            constraints_violation += max(0.0, (1.2 - ratio) * 10.0)
            
        # 🔧 ПРАВИЛО 2: Штраф за превышение лимита размаха
        if wing_metrics.span > span_limit:
            constraints_violation += (wing_metrics.span - span_limit) * 12.0
            
        efficiency = cl / max(cd, 1e-6)
        
        # 🔧 КРИТИЧЕСКИЙ ФИКС: обрезка экстремальных производных
        cma = float(np.clip(cma_raw, -2.0, 2.0))
        cmq = float(np.clip(cmq_raw, -2.0, 2.0))
        cm = float(np.clip(cm_raw, -1.0, 1.0))
        
        # 🔧 ШТРАФЫ НА НОРМАЛИЗОВАННЫХ ЗНАЧЕНИЯХ
        cl_penalty = 0.2 * abs(cl - target_cl)
        cm_penalty = 0.5 * abs(cm)
        
        cma_penalty = 0.0
        if cma > 0.1:
            cma_penalty = (cma - 0.1) * 0.5
            
        cmq_penalty = max(0.0, cmq) * 0.3
        
        # 🔧 Вес геометрии минимален для фазы обучения
        geom_weight = 0.02 if relax_mode else 0.05
        
        base_score = (
            efficiency * 1.0
            - cl_penalty
            - cm_penalty
            - cma_penalty
            - cmq_penalty
            - (geometry_penalty * geom_weight)
            - (constraints_violation * 0.5)
        )
        
        # 🔍 ОТЛАДКА (только первые 5 вызовов по счётчику _debug_idx — при генерации датасета передавайте номер попытки)
        if 1 <= _debug_idx <= 5:
            ca = max(canard_metrics.area, 1e-9)
            ar_note = f"{wing_metrics.area / ca:.2f}" if canard_metrics.area > 1e-6 else "n/a (no canard)"
            print(f"  [DBG#{_debug_idx}] eff={efficiency:5.2f} | span={wing_metrics.span:.2f}m | "
                  f"wing/canard_area={ar_note} | base={base_score:6.2f}")

        # ✅ ЛИНЕЙНЫЙ КЛИППИНГ
        score = float(np.clip(base_score, -15.0, 30.0))
        
        return DesignEvaluation(
            score=score,
            efficiency=efficiency,
            stability=stability,
            wing=wing_metrics,
            canard=canard_metrics,
            geometry_penalty=geometry_penalty,
            constraints_violation=constraints_violation,
            max_wing_span_limit=span_limit,
        )
    
    except Exception as e:
        print(f"Ошибка при оценке: {e}")
        return DesignEvaluation(
            score=-5.0, efficiency=0.0,
            stability=StabilityMetrics(0,0,0,0,0,0,0,0,0),
            wing=WingMetrics(0,0,0,0,0,0,0,0),
            canard=WingMetrics(0,0,0,0,0,0,0,0),
            geometry_penalty=10.0, constraints_violation=10.0,
            max_wing_span_limit=max_wing_span if max_wing_span is not None else 2.0,
        )


# =============================================================================
# 🔧 ИСПРАВЛЕННАЯ ГЕНЕРАЦИЯ ДАТАСЕТА
# =============================================================================

def generate_wing_dataset(
    base_config: main.AircraftConfig,
    sample_count: int,
    output_path: str,
    seed: int = 42,
    velocity: float = 50.0,
    target_cl: float = 0.55,
    min_wing_area: float = 0.3,
    relax_mode: bool = True,
    baseline_ratio: float = 0.7,
    only_valid: bool = True,
    max_attempts: int | None = None,
) -> None:
    """
    Сгенерировать датасет конфигураций крыла с полными метриками.

    По умолчанию ``only_valid=True``: в JSONL попадают только строки, прошедшие ``is_valid``
    (повторный сэмплинг до ``sample_count`` штук или до исчерпания ``max_attempts``).
    """
    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    output = Path(output_path)
    records: list[dict] = []
    max_wing_span_limit = extract_wing_metrics(base_config.wing).span
    baseline = base_opt.baseline_vector()
    limit = max_attempts if max_attempts is not None else max(sample_count * 150, 10_000)
    attempts = 0
    rejected = 0

    def sample_vector_legacy(accepted_idx: int) -> np.ndarray:
        if accepted_idx == 0 or rng.random() < baseline_ratio:
            noise = rng_np.normal(0, 0.12, size=len(base_opt.PARAMETER_SPECS))
            return base_opt.clamp_vector(baseline + noise)
        return base_opt.sample_random_vector(rng)

    if only_valid:
        while len(records) < sample_count:
            attempts += 1
            if attempts > limit:
                raise ValueError(
                    f"Не удалось набрать {sample_count} валидных образцов за {limit} попыток "
                    f"(принято {len(records)}). Увеличьте max_attempts, sample_count или включите relax_mode."
                )
            vector = sample_vector_legacy(len(records))
            config = base_opt.apply_design_vector(base_config, vector)
            evaluation = evaluate_wing_design(
                config,
                velocity=velocity,
                target_cl=target_cl * 0.9 if relax_mode else target_cl,
                min_wing_area=min_wing_area * 0.7 if relax_mode else min_wing_area,
                max_wing_span=max_wing_span_limit,
                relax_mode=relax_mode,
                # Номер попытки, не len(records)+1: иначе при отбраковке DBG#1 печатается тысячи раз.
                _debug_idx=attempts,
            )
            is_valid = evaluation.is_valid(min_wing_area=min_wing_area, max_wing_span=max_wing_span_limit)
            if not is_valid:
                rejected += 1
                if rejected % 400 == 0:
                    print(f"… v1: отклонено {rejected}, принято {len(records)}/{sample_count}, попытка {attempts}/{limit}")
                continue

            idx = len(records) + 1
            record = {
                "index": idx,
                "parameters": base_opt.vector_to_dict(vector),
                "score": evaluation.score,
                "efficiency": evaluation.efficiency,
                "geometry_penalty": evaluation.geometry_penalty,
                "constraints_violation": evaluation.constraints_violation,
                "is_valid": True,
                "stability": {
                    "Cma": evaluation.stability.Cma,
                    "Cmq": evaluation.stability.Cmq,
                    "Clp": evaluation.stability.Clp,
                    "Clr": evaluation.stability.Clr,
                    "Cnr": evaluation.stability.Cnr,
                    "Cnp": evaluation.stability.Cnp,
                    "Cm": evaluation.stability.Cm,
                    "Cl": evaluation.stability.Cl,
                    "Cd": evaluation.stability.Cd,
                },
                "wing": {
                    "area": evaluation.wing.area,
                    "aspect_ratio": evaluation.wing.aspect_ratio,
                    "span": evaluation.wing.span,
                    "root_chord": evaluation.wing.root_chord,
                    "tip_chord": evaluation.wing.tip_chord,
                    "sweep": evaluation.wing.sweep,
                    "dihedral": evaluation.wing.dihedral,
                },
                "canard": {
                    "area": evaluation.canard.area,
                    "aspect_ratio": evaluation.canard.aspect_ratio,
                    "span": evaluation.canard.span,
                },
            }
            records.append(record)
            print(
                f"[✓] {idx:04d}/{sample_count} "
                f"score={evaluation.score:8.3f} "
                f"eff={evaluation.efficiency:6.3f} "
                f"area={evaluation.wing.area:6.3f} span={evaluation.wing.span:5.2f}m "
                f"Clp={evaluation.stability.Clp:7.4f} "
                f"Cnr={evaluation.stability.Cnr:7.4f}"
            )
    else:
        configs = []
        for i in range(sample_count):
            if i == 0 or rng.random() < baseline_ratio:
                noise = rng_np.normal(0, 0.12, size=len(base_opt.PARAMETER_SPECS))
                candidate = baseline + noise
                configs.append(base_opt.clamp_vector(candidate))
            else:
                configs.append(base_opt.sample_random_vector(rng))

        valid_count = 0
        for index, vector in enumerate(configs, start=1):
            config = base_opt.apply_design_vector(base_config, vector)

            evaluation = evaluate_wing_design(
                config,
                velocity=velocity,
                target_cl=target_cl * 0.9 if relax_mode else target_cl,
                min_wing_area=min_wing_area * 0.7 if relax_mode else min_wing_area,
                max_wing_span=max_wing_span_limit,
                relax_mode=relax_mode,
                _debug_idx=index,
            )

            is_valid = evaluation.is_valid(min_wing_area=min_wing_area, max_wing_span=max_wing_span_limit)
            if is_valid:
                valid_count += 1

            record = {
                "index": index,
                "parameters": base_opt.vector_to_dict(vector),
                "score": evaluation.score,
                "efficiency": evaluation.efficiency,
                "geometry_penalty": evaluation.geometry_penalty,
                "constraints_violation": evaluation.constraints_violation,
                "is_valid": is_valid,
                "stability": {
                    "Cma": evaluation.stability.Cma,
                    "Cmq": evaluation.stability.Cmq,
                    "Clp": evaluation.stability.Clp,
                    "Clr": evaluation.stability.Clr,
                    "Cnr": evaluation.stability.Cnr,
                    "Cnp": evaluation.stability.Cnp,
                    "Cm": evaluation.stability.Cm,
                    "Cl": evaluation.stability.Cl,
                    "Cd": evaluation.stability.Cd,
                },
                "wing": {
                    "area": evaluation.wing.area,
                    "aspect_ratio": evaluation.wing.aspect_ratio,
                    "span": evaluation.wing.span,
                    "root_chord": evaluation.wing.root_chord,
                    "tip_chord": evaluation.wing.tip_chord,
                    "sweep": evaluation.wing.sweep,
                    "dihedral": evaluation.wing.dihedral,
                },
                "canard": {
                    "area": evaluation.canard.area,
                    "aspect_ratio": evaluation.canard.aspect_ratio,
                    "span": evaluation.canard.span,
                },
            }
            records.append(record)

            status = "✓" if is_valid else "✗"
            print(
                f"[{status}] {index:04d}/{sample_count} "
                f"score={evaluation.score:8.3f} "
                f"eff={evaluation.efficiency:6.3f} "
                f"area={evaluation.wing.area:6.3f} span={evaluation.wing.span:5.2f}m "
                f"Clp={evaluation.stability.Clp:7.4f} "
                f"Cnr={evaluation.stability.Cnr:7.4f}"
            )

    output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    print(f"\n📊 Датасет сохранён: {output}")
    if only_valid:
        print(f"✅ Только валидные строки: {len(records)} шт. (отброшено попыток: {rejected}, всего попыток: {attempts})")
    else:
        print(f"✅ Валидных конфигураций: {valid_count}/{sample_count} ({100 * valid_count / sample_count:.1f}%)")

    if not only_valid and valid_count < sample_count * 0.3 and relax_mode:
        print(f"⚠️  Мало валидных ({valid_count}). Попробуйте увеличить sample_count или проверить базовый дизайн.")


def generate_wing_dataset_v2(
    base_config: main.AircraftConfig,
    sample_count: int,
    output_path: str,
    seed: int = 42,
    velocity: float = 50.0,
    target_cl: float = 0.55,
    min_wing_area: float = 0.3,
    relax_mode: bool = True,
    baseline_ratio: float = 0.7,
    only_valid: bool = True,
    max_attempts: int | None = None,
) -> None:
    """
    Новая генерация датасета, использующая control_v2 — per-section параметры.
    Записывает записи в том же JSONL-формате, но с именами параметров по секциям.

    По умолчанию ``only_valid=True``: в файл пишутся только конфигурации с ``is_valid``,
    с повторным сэмплированием до набора ``sample_count`` строк.
    """
    import SCAT.AERO.control_v2 as control_v2

    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    output = Path(output_path)
    records: list[dict] = []
    max_wing_span_limit = extract_wing_metrics(base_config.wing).span

    specs = control_v2.build_parameter_specs_for_config(base_config)
    baseline = control_v2.baseline_vector_from_specs(specs)
    baseline_scale = 0.10 if relax_mode else 0.08
    mid_scale = 0.18 if relax_mode else 0.14
    global_scale = 0.35 if relax_mode else 0.28
    limit = max_attempts if max_attempts is not None else max(sample_count * 150, 10_000)

    def fuselage_signature(cfg: main.AircraftConfig):
        main_f = tuple((s.x, s.y, s.z, s.radius) for s in cfg.fuselage_sections)
        extra_f = tuple(tuple((s.x, s.y, s.z, s.radius) for s in chain) for chain in cfg.extra_fuselages)
        return main_f, extra_f

    base_fus = fuselage_signature(base_config)

    def sample_vector_v2(accepted_count: int) -> np.ndarray:
        """Та же фазовая логика, что раньше по индексу принятого образца."""
        if accepted_count == 0:
            return control_v2.clamp_vector(specs, baseline.copy())
        phase = accepted_count / max(sample_count - 1, 1)
        if phase < baseline_ratio:
            noise = rng_np.normal(0, baseline_scale, size=len(specs))
            candidate = baseline + noise
        elif phase < 0.9:
            noise = rng_np.normal(0, mid_scale, size=len(specs))
            candidate = baseline + noise
        else:
            if rng.random() < 0.5:
                noise = rng_np.normal(0, global_scale, size=len(specs))
                candidate = baseline + noise
            else:
                candidate = control_v2.sample_random_vector(specs, rng)
        return control_v2.clamp_vector(specs, candidate)

    best_score = float("-inf")
    best_eff = 0.0
    attempts = 0
    rejected = 0

    if only_valid:
        while len(records) < sample_count:
            attempts += 1
            if attempts > limit:
                raise ValueError(
                    f"Не удалось набрать {sample_count} валидных образцов за {limit} попыток "
                    f"(принято {len(records)}). Увеличьте max_attempts / sample_count или ослабьте relax_mode."
                )
            vector = sample_vector_v2(len(records))
            cfg = control_v2.apply_vector_to_config(base_config, specs, vector)
            if fuselage_signature(cfg) != base_fus:
                raise RuntimeError("Dataset generator v2 изменил фюзеляж, что запрещено.")
            evaluation = evaluate_wing_design(
                cfg,
                velocity=velocity,
                target_cl=target_cl * 0.9 if relax_mode else target_cl,
                min_wing_area=min_wing_area * 0.7 if relax_mode else min_wing_area,
                max_wing_span=max_wing_span_limit,
                relax_mode=relax_mode,
                _debug_idx=attempts,
            )
            best_score = max(best_score, evaluation.score)
            best_eff = max(best_eff, evaluation.efficiency)
            is_valid = evaluation.is_valid(min_wing_area=min_wing_area, max_wing_span=max_wing_span_limit)
            if not is_valid:
                rejected += 1
                if rejected % 400 == 0:
                    print(f"… v2: отклонено {rejected}, принято {len(records)}/{sample_count}, попытка {attempts}/{limit}")
                continue

            idx = len(records) + 1
            record = {
                "index": idx,
                "parameters": control_v2.vector_to_dict(specs, vector),
                "score": evaluation.score,
                "efficiency": evaluation.efficiency,
                "geometry_penalty": evaluation.geometry_penalty,
                "constraints_violation": evaluation.constraints_violation,
                "is_valid": True,
                "stability": {
                    "Cma": evaluation.stability.Cma,
                    "Cmq": evaluation.stability.Cmq,
                    "Clp": evaluation.stability.Clp,
                    "Clr": evaluation.stability.Clr,
                    "Cnr": evaluation.stability.Cnr,
                    "Cnp": evaluation.stability.Cnp,
                    "Cm": evaluation.stability.Cm,
                    "Cl": evaluation.stability.Cl,
                    "Cd": evaluation.stability.Cd,
                },
                "wing": {
                    "area": evaluation.wing.area,
                    "aspect_ratio": evaluation.wing.aspect_ratio,
                    "span": evaluation.wing.span,
                    "root_chord": evaluation.wing.root_chord,
                    "tip_chord": evaluation.wing.tip_chord,
                    "sweep": evaluation.wing.sweep,
                    "dihedral": evaluation.wing.dihedral,
                },
                "canard": {
                    "area": evaluation.canard.area,
                    "aspect_ratio": evaluation.canard.aspect_ratio,
                    "span": evaluation.canard.span,
                },
            }
            records.append(record)
            print(
                f"[✓] {idx:04d}/{sample_count} "
                f"score={evaluation.score:8.3f} "
                f"eff={evaluation.efficiency:6.3f} "
                f"area={evaluation.wing.area:6.3f} span={evaluation.wing.span:5.2f}m "
                f"Clp={evaluation.stability.Clp:7.4f} "
                f"Cnr={evaluation.stability.Cnr:7.4f}"
            )
    else:
        configs = [control_v2.clamp_vector(specs, baseline.copy())]
        while len(configs) < sample_count:
            idx = len(configs)
            phase = idx / max(sample_count - 1, 1)
            if phase < baseline_ratio:
                noise = rng_np.normal(0, baseline_scale, size=len(specs))
                candidate = baseline + noise
            elif phase < 0.9:
                noise = rng_np.normal(0, mid_scale, size=len(specs))
                candidate = baseline + noise
            else:
                if rng.random() < 0.5:
                    noise = rng_np.normal(0, global_scale, size=len(specs))
                    candidate = baseline + noise
                else:
                    candidate = control_v2.sample_random_vector(specs, rng)
            configs.append(control_v2.clamp_vector(specs, candidate))

        valid_count = 0
        for index, vector in enumerate(configs, start=1):
            cfg = control_v2.apply_vector_to_config(base_config, specs, vector)
            if fuselage_signature(cfg) != base_fus:
                raise RuntimeError("Dataset generator v2 изменил фюзеляж, что запрещено.")
            evaluation = evaluate_wing_design(
                cfg,
                velocity=velocity,
                target_cl=target_cl * 0.9 if relax_mode else target_cl,
                min_wing_area=min_wing_area * 0.7 if relax_mode else min_wing_area,
                max_wing_span=max_wing_span_limit,
                relax_mode=relax_mode,
                _debug_idx=index,
            )

            is_valid = evaluation.is_valid(min_wing_area=min_wing_area, max_wing_span=max_wing_span_limit)
            if is_valid:
                valid_count += 1
            best_score = max(best_score, evaluation.score)
            best_eff = max(best_eff, evaluation.efficiency)

            record = {
                "index": index,
                "parameters": control_v2.vector_to_dict(specs, vector),
                "score": evaluation.score,
                "efficiency": evaluation.efficiency,
                "geometry_penalty": evaluation.geometry_penalty,
                "constraints_violation": evaluation.constraints_violation,
                "is_valid": is_valid,
                "stability": {
                    "Cma": evaluation.stability.Cma,
                    "Cmq": evaluation.stability.Cmq,
                    "Clp": evaluation.stability.Clp,
                    "Clr": evaluation.stability.Clr,
                    "Cnr": evaluation.stability.Cnr,
                    "Cnp": evaluation.stability.Cnp,
                    "Cm": evaluation.stability.Cm,
                    "Cl": evaluation.stability.Cl,
                    "Cd": evaluation.stability.Cd,
                },
                "wing": {
                    "area": evaluation.wing.area,
                    "aspect_ratio": evaluation.wing.aspect_ratio,
                    "span": evaluation.wing.span,
                    "root_chord": evaluation.wing.root_chord,
                    "tip_chord": evaluation.wing.tip_chord,
                    "sweep": evaluation.wing.sweep,
                    "dihedral": evaluation.wing.dihedral,
                },
                "canard": {
                    "area": evaluation.canard.area,
                    "aspect_ratio": evaluation.canard.aspect_ratio,
                    "span": evaluation.canard.span,
                },
            }
            records.append(record)

            status = "✓" if is_valid else "✗"
            print(
                f"[{status}] {index:04d}/{sample_count} "
                f"score={evaluation.score:8.3f} "
                f"eff={evaluation.efficiency:6.3f} "
                f"area={evaluation.wing.area:6.3f} span={evaluation.wing.span:5.2f}m "
                f"Clp={evaluation.stability.Clp:7.4f} "
                f"Cnr={evaluation.stability.Cnr:7.4f}"
            )

    output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )
    print(f"\n📊 Датасет v2 сохранён: {output}")
    if only_valid:
        print(f"✅ Только валидные строки: {len(records)} шт. (отклонено попыток: {rejected}, всего попыток: {attempts})")
    else:
        print(f"✅ Валидных конфигураций: {valid_count}/{sample_count} ({100 * valid_count / sample_count:.1f}%)")
    print(f"🏁 Лучший score={best_score:.3f}, лучший efficiency(L/D)={best_eff:.3f}")


# =============================================================================
# Загрузка датасета (без изменений)
# =============================================================================

def dataset_to_arrays_extended(
    dataset_path: str,
    only_valid: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Загрузить датасет в массивы для обучения ML модели."""
    lines = Path(dataset_path).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    
    if not records:
        raise ValueError("Датасет пустой.")
    
    xs = []
    ys = []
    filtered_records = []

    # Поддержка и legacy-датасета, и нового v2:
    # формируем единый порядок признаков по первому объекту.
    if not records:
        raise ValueError("Датасет пустой.")
    first_params = records[0].get("parameters", {})
    if not isinstance(first_params, dict) or not first_params:
        raise ValueError("Некорректный формат датасета: parameters пустой.")
    feature_keys = sorted(first_params.keys())

    for record in records:
        if only_valid and not record.get("is_valid", False):
            continue

        params = record["parameters"]
        xs.append([float(params[key]) for key in feature_keys])
        ys.append(float(record["score"]))
        filtered_records.append(record)

    return np.array(xs, dtype=float), np.array(ys, dtype=float), filtered_records


# =============================================================================
# Точка входа
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Оптимизация крыла с ML и стабильностью")
    parser.add_argument("--samples", type=int, default=200, help="Количество примеров в датасете")
    parser.add_argument("--output", type=str, default="wing_dataset.jsonl", help="Путь к датасету")
    parser.add_argument("--velocity", type=float, default=50.0, help="Скорость полёта")
    parser.add_argument("--target-cl", type=float, default=0.55, help="Целевой CL")
    parser.add_argument("--min-wing-area", type=float, default=0.3, help="Минимальная площадь крыла")
    parser.add_argument("--no-relax", action="store_true", help="Отключить мягкий режим (для финальной оценки)")
    parser.add_argument("--baseline-ratio", type=float, default=0.7, help="Доля конфигураций вокруг базового дизайна")
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Писать в датасет и невалидные точки (по умолчанию — только is_valid).",
    )

    args = parser.parse_args()

    base_config = base_opt.load_config(None)
    generate_wing_dataset(
        base_config=base_config,
        sample_count=args.samples,
        output_path=args.output,
        velocity=args.velocity,
        target_cl=args.target_cl,
        min_wing_area=args.min_wing_area,
        relax_mode=not args.no_relax,
        baseline_ratio=args.baseline_ratio,
        only_valid=not args.include_invalid,
    )