from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import socketserver
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse


def import_aerosandbox():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    try:
        import aerosandbox as imported_asb

        return imported_asb
    except ImportError:
        project_root = Path(__file__).resolve().parent
        for candidate in project_root.glob(".venv/lib/python*/site-packages"):
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)
        try:
            import aerosandbox as imported_asb

            return imported_asb
        except ImportError:
            return None


asb = import_aerosandbox()
optimization_jobs: dict[str, dict] = {}
optimization_jobs_lock = threading.Lock()


@dataclass
class FuselageSection:
    x: float
    y: float
    z: float
    radius: float


@dataclass
class SurfaceSection:
    x: float
    y: float
    z: float
    chord: float
    twist: float


@dataclass
class SurfaceConfig:
    name: str
    enabled: bool = True
    symmetric: bool = True
    color: str = "#4f7cff"
    sections: list[SurfaceSection] = field(default_factory=list)


@dataclass
class AircraftConfig:
    airplane_name: str = "My Concept"
    airfoil_name: str = "naca2412"
    fuselage_name: str = "Main Fuselage"
    fuselage_symmetry: str = "XZ"
    draw_backend: str = "pyvista"
    thin_wings: bool = False
    fuselage_sections: list[FuselageSection] = field(default_factory=list)
    extra_fuselages: list[list[FuselageSection]] = field(default_factory=list)
    wing: SurfaceConfig = field(default_factory=lambda: SurfaceConfig(name="Main Wing"))
    canard: SurfaceConfig = field(default_factory=lambda: SurfaceConfig(name="Canard", enabled=True))
    htail: SurfaceConfig = field(default_factory=lambda: SurfaceConfig(name="Horizontal Tail", enabled=False))
    vtail: SurfaceConfig = field(
        default_factory=lambda: SurfaceConfig(
            name="Vertical Tail",
            enabled=False,
            symmetric=False,
            color="#28a36a",
        )
    )


def _classify_surfaces_from_wings(wings_data: list[dict]) -> tuple[SurfaceConfig, SurfaceConfig, SurfaceConfig, SurfaceConfig]:
    wing_cfg = None
    canard_cfg = SurfaceConfig(name="Canard", enabled=False)
    htail_cfg = SurfaceConfig(name="Horizontal Tail", enabled=False)
    vtail_cfg = SurfaceConfig(name="Vertical Tail", enabled=False, symmetric=False)
    fallback_surfaces = []
    htail_candidates: list[SurfaceConfig] = []

    for wing_data in wings_data:
        name = wing_data.get("name", "wing") or "wing"
        sections = [
            SurfaceSection(
                x=float(section.get("x", 0.0)),
                y=float(section.get("y", 0.0)),
                z=float(section.get("z", 0.0)),
                chord=float(section.get("chord", 0.1)),
                twist=float(section.get("twist", 0.0)),
            )
            for section in wing_data.get("sections", [])
        ]
        cfg = SurfaceConfig(
            name=name,
            enabled=True,
            symmetric=bool(wing_data.get("symmetric", True)),
            sections=sections,
        )
        fallback_surfaces.append(cfg)
        lname = name.lower()
        if "canard" in lname or "front" in lname:
            canard_cfg = cfg
        elif "main" in lname or ("wing" in lname and "tail" not in lname):
            wing_cfg = cfg
        elif "tail" in lname and ("v" in lname or "vertical" in lname):
            vtail_cfg = cfg
        elif "tail" in lname:
            htail_candidates.append(cfg)

    # Если хвост задан двумя независимыми половинами (Left/Right Tail),
    # сохраняем обе, чтобы не терять реальную геометрию из baseTim.
    if len(htail_candidates) >= 2:
        left_like = next(
            (c for c in htail_candidates if any(section.y > 0 for section in c.sections)),
            htail_candidates[0],
        )
        right_like = next(
            (c for c in htail_candidates if c is not left_like),
            htail_candidates[1],
        )
        htail_cfg = SurfaceConfig(
            name=left_like.name or "Left Tail",
            enabled=True,
            symmetric=False,
            color=left_like.color,
            sections=left_like.sections,
        )
        # Используем слот vtail как вторую независимую хвостовую плоскость.
        vtail_cfg = SurfaceConfig(
            name=right_like.name or "Right Tail",
            enabled=True,
            symmetric=False,
            color=right_like.color,
            sections=right_like.sections,
        )
    elif htail_candidates:
        htail_cfg = htail_candidates[0]

    if wing_cfg is None:
        wing_cfg = next(
            (surface for surface in fallback_surfaces if "wing" in surface.name.lower() and "tail" not in surface.name.lower()),
            fallback_surfaces[0] if fallback_surfaces else SurfaceConfig(name="Main Wing", enabled=False),
        )

    return wing_cfg, canard_cfg, htail_cfg, vtail_cfg


def _extract_basetim_data_via_subprocess(project_root: Path) -> dict | None:
    extractor = (
        "import contextlib, io, json, sys\n"
        f"sys.path.insert(0, {repr(str(project_root))})\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "  import baseTim\n"
        "plane = baseTim.plane\n"
        "data = {\n"
        "  'fuselages': [\n"
        "    [\n"
        "      {\n"
        "        'x': float(x.xyz_c[0]), 'y': float(x.xyz_c[1]), 'z': float(x.xyz_c[2]),\n"
        "        'radius': float(x.equivalent_radius() if hasattr(x, 'equivalent_radius') else getattr(x, 'radius', 0.1))\n"
        "      }\n"
        "      for x in getattr(f, 'xsecs', [])\n"
        "      if getattr(x, 'xyz_c', None) is not None\n"
        "    ]\n"
        "    for f in getattr(plane, 'fuselages', [])\n"
        "  ],\n"
        "  'wings': [\n"
        "    {\n"
        "      'name': getattr(w, 'name', 'wing') or 'wing',\n"
        "      'symmetric': bool(getattr(w, 'symmetric', True)),\n"
        "      'sections': [\n"
        "        {\n"
        "          'x': float(x.xyz_le[0]), 'y': float(x.xyz_le[1]), 'z': float(x.xyz_le[2]),\n"
        "          'chord': float(getattr(x, 'chord', 0.1)), 'twist': float(getattr(x, 'twist', 0.0))\n"
        "        }\n"
        "        for x in getattr(w, 'xsecs', [])\n"
        "      ]\n"
        "    }\n"
        "    for w in getattr(plane, 'wings', [])\n"
        "  ]\n"
        "}\n"
        "print(json.dumps(data, ensure_ascii=False))\n"
    )

    interpreters = [
        str(project_root / "tf-env" / "bin" / "python"),
        str(project_root / ".venv" / "bin" / "python"),
        str(project_root / "tf-env" / "bin" / "python3"),
        str(project_root / ".venv" / "bin" / "python3"),
        "python3",
        "python",
    ]
    for interpreter in interpreters:
        if "/" in interpreter and not Path(interpreter).exists():
            continue
        if "/" not in interpreter and shutil.which(interpreter) is None:
            continue
        try:
            completed = subprocess.run(
                [interpreter, "-c", extractor],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(project_root),
            )
            return json.loads(completed.stdout)
        except Exception:
            continue
    return None


def _build_config_from_basetim_data(data: dict) -> AircraftConfig | None:
    fuselage_chains = []
    for chain in data.get("fuselages", []):
        if not isinstance(chain, list):
            continue
        sections = [
            FuselageSection(
                x=float(section.get("x", 0.0)),
                y=float(section.get("y", 0.0)),
                z=float(section.get("z", 0.0)),
                radius=float(section.get("radius", 0.1)),
            )
            for section in chain
            if isinstance(section, dict)
        ]
        if len(sections) >= 2:
            fuselage_chains.append(sections)

    if not fuselage_chains:
        return None

    primary_index = max(
        range(len(fuselage_chains)),
        key=lambda idx: abs(fuselage_chains[idx][-1].x - fuselage_chains[idx][0].x),
    )
    fus_sections = fuselage_chains[primary_index]
    extra_fuselages = [chain for idx, chain in enumerate(fuselage_chains) if idx != primary_index]

    wing_cfg, canard_cfg, htail_cfg, vtail_cfg = _classify_surfaces_from_wings(data.get("wings", []))

    return AircraftConfig(
        fuselage_sections=fus_sections,
        extra_fuselages=extra_fuselages,
        wing=wing_cfg,
        canard=canard_cfg,
        htail=htail_cfg,
        vtail=vtail_cfg,
    )


def default_config() -> AircraftConfig:
  # Попробовать импортировать пользовательский базовый скрипт `baseTim.py`.
  project_root = Path(__file__).resolve().parent
  try:
    extracted = _extract_basetim_data_via_subprocess(project_root)
    if extracted is not None:
      config = _build_config_from_basetim_data(extracted)
      if config is not None:
        return config

  except Exception:
    # Если импорт не удался — продолжаем к дефолтной жёстко заданной конструкции
    pass

  # Фолбэк-дефолт: геометрия из baseTim.py
  return AircraftConfig(
    fuselage_sections=[
      FuselageSection(-0.32, 0.0, -0.01, 0.034),
      FuselageSection(-0.12, 0.0, 0.0, 0.052),
      FuselageSection(0.08, 0.0, 0.0, 0.052),
      FuselageSection(0.18, 0.0, 0.0, 0.045),
      FuselageSection(0.24, 0.0, 0.0, 0.012),
    ],
    extra_fuselages=[
      [
        FuselageSection(0.16, 0.16, 0.0, 0.008),
        FuselageSection(0.665, 0.16, 0.0, 0.008),
      ],
      [
        FuselageSection(0.16, -0.16, 0.0, 0.008),
        FuselageSection(0.665, -0.16, 0.0, 0.008),
      ],
    ],
    wing=SurfaceConfig(
      name="Main Wing",
      enabled=True,
      symmetric=True,
      color="#2468f2",
      sections=[
        SurfaceSection(-0.08, 0.0, 0.0, 0.255, 2.0),
        SurfaceSection(0.0, 0.2925, 0.0, 0.220, 0.0),
        SurfaceSection(0.08, 0.4875, 0.0, 0.170, -2.0),
        SurfaceSection(0.20, 0.5525, 0.035, 0.070, -4.0),
        SurfaceSection(0.24, 0.5720, 0.070, 0.055, -6.0),
      ],
    ),
    canard=SurfaceConfig(
      name="Canard",
      enabled=False,
      symmetric=True,
      color="#ff8c3b",
      sections=[],
    ),
    htail=SurfaceConfig(
      name="Left Tail",
      enabled=True,
      symmetric=False,
      color="#7e57ff",
      sections=[
        SurfaceSection(0.515, 0.16, 0.0, 0.15, 0.0),
        SurfaceSection(0.545, 0.0, 0.15, 0.10, 0.0),
      ],
    ),
    vtail=SurfaceConfig(
      name="Right Tail",
      enabled=True,
      symmetric=False,
      color="#26a269",
      sections=[
        SurfaceSection(0.515, -0.16, 0.0, 0.15, 0.0),
        SurfaceSection(0.545, 0.0, 0.15, 0.10, 0.0),
      ],
    ),
  )


def surface_to_dict(surface: SurfaceConfig) -> dict:
    data = asdict(surface)
    data["sections"] = [asdict(section) for section in surface.sections]
    return data


def config_to_dict(config: AircraftConfig) -> dict:
    data = asdict(config)
    data["fuselage_sections"] = [asdict(section) for section in config.fuselage_sections]
    data["extra_fuselages"] = [
        [asdict(section) for section in chain]
        for chain in config.extra_fuselages
    ]
    data["wing"] = surface_to_dict(config.wing)
    data["canard"] = surface_to_dict(config.canard)
    data["htail"] = surface_to_dict(config.htail)
    data["vtail"] = surface_to_dict(config.vtail)
    return data


def load_surface(data: dict, default_name: str, default_color: str, default_symmetric: bool) -> SurfaceConfig:
    return SurfaceConfig(
        name=data.get("name", default_name),
        enabled=data.get("enabled", True),
        symmetric=data.get("symmetric", default_symmetric),
        color=data.get("color", default_color),
        sections=[SurfaceSection(**section) for section in data.get("sections", [])],
    )


def config_from_dict(data: dict) -> AircraftConfig:
    return AircraftConfig(
        airplane_name=data.get("airplane_name", "My Concept"),
        airfoil_name=data.get("airfoil_name", "naca2412"),
        fuselage_name=data.get("fuselage_name", "Main Fuselage"),
        fuselage_symmetry=data.get("fuselage_symmetry", "XZ"),
        draw_backend=data.get("draw_backend", "pyvista"),
        thin_wings=data.get("thin_wings", False),
        fuselage_sections=[FuselageSection(**section) for section in data.get("fuselage_sections", [])],
        extra_fuselages=[
            [FuselageSection(**section) for section in chain]
            for chain in data.get("extra_fuselages", [])
            if isinstance(chain, list)
        ],
        wing=load_surface(data.get("wing", {}), "Main Wing", "#2468f2", True),
        canard=load_surface(data.get("canard", {}), "Canard", "#ff8c3b", True),
        htail=load_surface(data.get("htail", {}), "Horizontal Tail", "#7e57ff", True),
        vtail=load_surface(data.get("vtail", {}), "Vertical Tail", "#26a269", False),
    )


def build_airplane(config: AircraftConfig):
    if asb is None:
        raise RuntimeError("AeroSandbox не найден. Установи его в .venv или текущее окружение.")
    if len(config.fuselage_sections) < 2:
        raise ValueError("Для фюзеляжа нужно минимум 2 секции.")

    airfoil = asb.Airfoil(config.airfoil_name)
    primary_fuselage = asb.Fuselage(
        name=config.fuselage_name,
        xsecs=[
            asb.FuselageXSec(xyz_c=[section.x, section.y, section.z], radius=section.radius)
            for section in config.fuselage_sections
        ],
        symmetry=None if config.fuselage_symmetry == "none" else config.fuselage_symmetry,
    )
    fuselages = [primary_fuselage]
    for idx, chain in enumerate(config.extra_fuselages, start=1):
        if len(chain) < 2:
            continue
        fuselages.append(
            asb.Fuselage(
                name=f"{config.fuselage_name} Extra {idx}",
                xsecs=[
                    asb.FuselageXSec(xyz_c=[section.x, section.y, section.z], radius=section.radius)
                    for section in chain
                ],
                symmetry=None,
            )
        )

    wings = []
    for surface in (config.canard, config.wing, config.htail, config.vtail):
        if not surface.enabled:
            continue
        if len(surface.sections) < 2:
            raise ValueError(f"Поверхность '{surface.name}' должна содержать минимум 2 секции.")
        wings.append(
            asb.Wing(
                name=surface.name,
                symmetric=surface.symmetric,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[section.x, section.y, section.z],
                        chord=section.chord,
                        twist=section.twist,
                        airfoil=airfoil,
                    )
                    for section in surface.sections
                ],
            )
        )

    return asb.Airplane(name=config.airplane_name, wings=wings, fuselages=fuselages)


def fuselage_length(config: AircraftConfig) -> float:
    xs = [section.x for section in config.fuselage_sections]
    return max(xs) - min(xs) if xs else 0.0


def launch_render_process(config: AircraftConfig) -> Path:
    script = generate_python_script(config)
    temp_dir = Path(tempfile.gettempdir())
    script_path = temp_dir / "aerosandbox_preview_render.py"
    script_path.write_text(script, encoding="utf-8")
    subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(Path(__file__).resolve().parent),
    )
    return script_path


def generate_python_script(config: AircraftConfig) -> str:
    data = json.dumps(config_to_dict(config), ensure_ascii=False, indent=4)
    return f"""import json
import sys
from pathlib import Path

try:
    import aerosandbox as asb
except ImportError:
    project_root = Path("/Users/aleksandrvorobev/Documents/skat")
    for candidate in project_root.glob(".venv/lib/python*/site-packages"):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.append(candidate_str)
    import aerosandbox as asb

CONFIG = json.loads('''{data}''')

airfoil = asb.Airfoil(CONFIG["airfoil_name"])
primary_fuselage = asb.Fuselage(
    name=CONFIG["fuselage_name"],
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[section["x"], section["y"], section["z"]],
            radius=section["radius"],
        )
        for section in CONFIG["fuselage_sections"]
    ],
    symmetry=None if CONFIG["fuselage_symmetry"] == "none" else CONFIG["fuselage_symmetry"],
)
fuselages = [primary_fuselage]
for idx, chain in enumerate(CONFIG.get("extra_fuselages", []), start=1):
    if len(chain) < 2:
        continue
    fuselages.append(
        asb.Fuselage(
            name=f"{{CONFIG['fuselage_name']}} Extra {{idx}}",
            xsecs=[
                asb.FuselageXSec(
                    xyz_c=[section["x"], section["y"], section["z"]],
                    radius=section["radius"],
                )
                for section in chain
            ],
            symmetry=None,
        )
    )

wings = []
for surface_key in ["canard", "wing", "htail", "vtail"]:
    surface = CONFIG[surface_key]
    if not surface["enabled"]:
        continue
    wings.append(
        asb.Wing(
            name=surface["name"],
            symmetric=surface["symmetric"],
            xsecs=[
                asb.WingXSec(
                    xyz_le=[section["x"], section["y"], section["z"]],
                    chord=section["chord"],
                    twist=section["twist"],
                    airfoil=airfoil,
                )
                for section in surface["sections"]
            ],
        )
    )

airplane = asb.Airplane(
    name=CONFIG["airplane_name"],
    wings=wings,
    fuselages=fuselages,
)

airplane.draw(
    backend=CONFIG["draw_backend"],
    thin_wings=CONFIG["thin_wings"],
    use_preset_view_angle="iso",
    set_background_pane_color="white",
    show=True,
)
"""


def html_page() -> str:
    return """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AeroSandbox Aircraft Designer</title>
  <style>
    :root {
      --bg: #eef4fb;
      --card: #ffffff;
      --ink: #173047;
      --muted: #5a7286;
      --line: #c9d8e6;
      --accent: #1d6ef2;
      --accent-2: #ff8b39;
      --good: #209460;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(29,110,242,.15), transparent 28%),
        radial-gradient(circle at top right, rgba(255,139,57,.15), transparent 24%),
        linear-gradient(180deg, #f7fbff 0%, var(--bg) 100%);
      color: var(--ink);
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(780px, 1.8fr) minmax(420px, 1fr);
      gap: 18px;
      min-height: 100vh;
      padding: 18px;
    }
    .panel {
      background: rgba(255,255,255,.88);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(201,216,230,.9);
      border-radius: 20px;
      box-shadow: 0 20px 60px rgba(22,49,74,.10);
      overflow: hidden;
    }
    .header {
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .header h1, .header h2 {
      margin: 0;
      font-size: 22px;
    }
    .toolbar, .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    button, input, select, textarea {
      font: inherit;
    }
    button {
      border: 0;
      border-radius: 12px;
      padding: 10px 14px;
      cursor: pointer;
      color: white;
      background: linear-gradient(135deg, var(--accent), #4d93ff);
      box-shadow: 0 8px 18px rgba(29,110,242,.22);
    }
    button.secondary {
      background: white;
      color: var(--ink);
      border: 1px solid var(--line);
      box-shadow: none;
    }
    button.good {
      background: linear-gradient(135deg, var(--good), #3abb7d);
      box-shadow: 0 8px 18px rgba(32,148,96,.20);
    }
    .content {
      padding: 18px;
    }
    .tabs {
      display: flex;
      gap: 8px;
      padding: 0 18px 18px;
      flex-wrap: wrap;
    }
    .tab {
      background: #eef4fb;
      color: var(--muted);
      border-radius: 999px;
      padding: 8px 12px;
      border: 1px solid transparent;
      cursor: pointer;
      user-select: none;
    }
    .tab.active {
      background: white;
      color: var(--ink);
      border-color: var(--line);
      box-shadow: 0 8px 20px rgba(22,49,74,.08);
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .field {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .field.full { grid-column: 1 / -1; }
    label {
      font-size: 13px;
      color: var(--muted);
    }
    input, select, textarea {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: white;
      color: var(--ink);
    }
    .stack {
      display: grid;
      gap: 14px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--card);
      padding: 14px;
    }
    .card h3 {
      margin: 0 0 12px;
      font-size: 16px;
    }
    .surface-top {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 12px;
      border: 1px solid var(--line);
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px;
      text-align: center;
      background: rgba(255,255,255,.9);
    }
    th {
      background: #f4f8fc;
      color: var(--muted);
      font-weight: 600;
    }
    td input {
      width: 100%;
      padding: 7px 8px;
      border-radius: 8px;
    }
    .table-actions {
      display: flex;
      gap: 8px;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    .preview-wrap {
      padding: 18px;
      display: grid;
      gap: 14px;
    }
    canvas {
      width: 100%;
      height: 520px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #fbfdff 0%, #edf4fb 100%);
    }
    canvas.stability-chart {
      height: 280px;
      max-height: 42vh;
    }
    .summary {
      white-space: pre-wrap;
      line-height: 1.5;
      color: var(--muted);
      background: #f8fbff;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }
    .status {
      padding: 0 18px 18px;
      color: var(--muted);
    }
    .tip {
      font-size: 13px;
      color: var(--muted);
    }
    .legend {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      font-size: 13px;
      color: var(--muted);
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .legend-line {
      width: 22px;
      height: 0;
      border-top: 3px solid var(--accent);
    }
    .legend-line.optimized {
      border-top-color: var(--accent-2);
      border-top-style: dashed;
    }
    @media (max-width: 1220px) {
      .layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
      .grid, .surface-top { grid-template-columns: 1fr; }
      .header { align-items: flex-start; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <section class="panel">
      <div class="header">
        <div>
          <h1>Конструктор самолёта</h1>
          <div class="tip">Гибкая сборка фюзеляжа, крыла, ПГО, ГО и ВО с экспортом в AeroSandbox.</div>
        </div>
        <div class="toolbar">
          <button class="secondary" id="newProjectBtn">Новый</button>
          <button class="secondary" id="loadJsonBtn">Импорт JSON</button>
          <button class="secondary" id="saveJsonBtn">Экспорт JSON</button>
          <button class="secondary" id="exportPyBtn">Экспорт Python</button>
          <button class="secondary" id="optimizeBtn">Оптимизировать</button>
          <button class="good" id="renderBtn">3D визуализация</button>
        </div>
      </div>
      <div class="tabs" id="tabs"></div>
      <div class="content">
        <div id="tab-general" class="tab-panel active"></div>
        <div id="tab-fuselage" class="tab-panel"></div>
        <div id="tab-wing" class="tab-panel"></div>
        <div id="tab-canard" class="tab-panel"></div>
        <div id="tab-htail" class="tab-panel"></div>
        <div id="tab-vtail" class="tab-panel"></div>
      </div>
      <div class="status" id="status">Готово.</div>
      <input id="jsonFileInput" type="file" accept=".json,application/json" hidden>
    </section>

    <aside class="panel">
      <div class="header">
        <div>
          <h2>Превью</h2>
          <div class="tip">Верхняя половина: вид сверху. Нижняя: вид сбоку. Оранжевая пунктирная геометрия — оптимизированный вариант.</div>
        </div>
        <div class="actions">
          <button class="secondary" id="refreshBtn">Обновить</button>
          <button class="secondary" id="scaleFuselageBtn">Масштабировать длину</button>
          <button class="secondary" id="applyOptimizedBtn">Применить оптимум</button>
          <button class="secondary" id="clearOptimizedBtn">Скрыть оптимум</button>
        </div>
      </div>
      <div class="preview-wrap">
        <div class="card">
          <h3>Параметры оптимизации</h3>
          <div class="grid">
            <div class="field full">
              <label for="optMode">Режим оптимизации</label>
              <select id="optMode">
                <option value="algorithm">🧬 Algorithm (CEM) - быстро, универсально</option>
                <option value="ai">🤖 AI (Нейросеть) - требует модель, очень быстро</option>
                <option value="ai+algorithm">🔄 Гибридный (AI+CEM) - балансированно</option>
              </select>
              <div class="tip" style="margin-top: 6px;">
                Algorithm: генетическая оптимизация | AI: нейросеть (если обучена) | Гибридный: оба метода
              </div>
            </div>
            <div class="field">
              <label for="optIterations">Итерации</label>
              <input id="optIterations" type="number" value="10" min="1" step="1">
            </div>
            <div class="field">
              <label for="optPopulation">Популяция</label>
              <input id="optPopulation" type="number" value="20" min="4" step="1">
            </div>
            <div class="field">
              <label for="optMinWingArea">Мин. площадь крыла</label>
              <input id="optMinWingArea" type="number" value="0.3" min="0.1" step="0.05">
            </div>
            <div class="field">
              <label for="optTargetCl">Целевой CL</label>
              <input id="optTargetCl" type="number" value="0.55" step="0.01">
            </div>
            <div class="field">
              <label for="optVelocity">Скорость, м/с</label>
              <input id="optVelocity" type="number" value="50" step="1">
            </div>
          </div>
        </div>
        <div class="legend">
          <span class="legend-item"><span class="legend-line"></span>Текущая конфигурация</span>
          <span class="legend-item"><span class="legend-line optimized"></span>Оптимизированная конфигурация</span>
        </div>
        <canvas id="preview" width="900" height="560"></canvas>
        <div class="card">
          <h3>График 6 — стабильность и ограничения</h3>
          <div class="tip">
            Как в analyze_wing (график 6 датасета): столбцы 0/1 по критериям (образец, валидность ТЗ, площадь, крен Clp, рысканье Cnr, тангаж Cma).
            Расчёт: AeroBuildup, α = 0°, параметры из блока слева. Кнопка запрашивает сервер.
          </div>
          <div class="toolbar" style="margin: 10px 0;">
            <button class="good" id="stabilityBtn" type="button">Проверить стабильность</button>
          </div>
          <canvas id="stabilityChart6" class="stability-chart" width="900" height="280"></canvas>
          <div class="summary" id="stabilityMetrics" style="margin-top:10px;">Метрики появятся после проверки.</div>
        </div>
        <div class="summary" id="summary"></div>
        <div class="summary" id="optimizationSummary">Оптимизация ещё не запускалась.</div>
      </div>
    </aside>
  </div>

  <script>
    const defaultConfig = __DEFAULT_CONFIG__;
    const tabs = [
      ["tab-general", "Общее"],
      ["tab-fuselage", "Фюзеляж"],
      ["tab-wing", "Крыло"],
      ["tab-canard", "ПГО"],
      ["tab-htail", "ГО"],
      ["tab-vtail", "ВО"],
    ];

    function deepClone(value) {
      if (typeof structuredClone === "function") {
        return structuredClone(value);
      }
      return JSON.parse(JSON.stringify(value));
    }

    let state = deepClone(defaultConfig);
    let optimizedState = null;
    let optimizationResult = null;
    let stabilityChart6Data = null;
    let stabilityEvaluationText = "";

    function setStatus(text) {
      document.getElementById("status").textContent = text;
    }

    function number(value, fallback = 0) {
      const parsed = parseFloat(String(value).replace(",", "."));
      return Number.isFinite(parsed) ? parsed : fallback;
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function surfaceColumnDefs() {
      return [
        ["x", "X LE"],
        ["y", "Y LE"],
        ["z", "Z LE"],
        ["chord", "Chord"],
        ["twist", "Twist"],
      ];
    }

    function renderTabs() {
      const host = document.getElementById("tabs");
      host.innerHTML = tabs.map(([id, label], index) =>
        `<div class="tab ${index === 0 ? "active" : ""}" data-target="${id}">${label}</div>`
      ).join("");
      host.querySelectorAll(".tab").forEach(tab => {
        tab.addEventListener("click", () => {
          host.querySelectorAll(".tab").forEach(item => item.classList.remove("active"));
          document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.remove("active"));
          tab.classList.add("active");
          document.getElementById(tab.dataset.target).classList.add("active");
        });
      });
    }

    function generalPanelHtml() {
      return `
        <div class="stack">
          <div class="card">
            <h3>Общие параметры</h3>
            <div class="grid">
              ${inputField("airplane_name", "Имя самолёта", state.airplane_name)}
              ${inputField("fuselage_name", "Имя фюзеляжа", state.fuselage_name)}
              ${inputField("airfoil_name", "Профиль", state.airfoil_name)}
              ${selectField("fuselage_symmetry", "Симметрия фюзеляжа", state.fuselage_symmetry, ["none", "XZ", "XY", "YZ"])}
              ${selectField("draw_backend", "Backend", state.draw_backend, ["pyvista"])}
              ${inputField("fuselage_length", "Желаемая длина фюзеляжа", getFuselageLength(state).toFixed(4))}
              <div class="field full">
                <label><input type="checkbox" id="thin_wings" ${state.thin_wings ? "checked" : ""}> Тонкие крылья при рендере</label>
              </div>
            </div>
          </div>
          <div class="card">
            <h3>Что можно настраивать</h3>
            <div class="tip">
              Любое количество секций фюзеляжа и аэродинамических поверхностей, полная геометрия по X/Y/Z, chord и twist,
              включение и отключение отдельных поверхностей, автоматическое масштабирование длины фюзеляжа,
              экспорт JSON и генерация готового Python-кода для AeroSandbox.
            </div>
          </div>
        </div>
      `;
    }

    function inputField(id, label, value, type = "text") {
      return `
        <div class="field">
          <label for="${id}">${label}</label>
          <input id="${id}" type="${type}" value="${escapeHtml(value)}">
        </div>
      `;
    }

    function selectField(id, label, current, values) {
      const options = values.map(value =>
        `<option value="${value}" ${value === current ? "selected" : ""}>${value}</option>`
      ).join("");
      return `
        <div class="field">
          <label for="${id}">${label}</label>
          <select id="${id}">${options}</select>
        </div>
      `;
    }

    function renderGeneralPanel() {
      document.getElementById("tab-general").innerHTML = generalPanelHtml();
      ["airplane_name", "fuselage_name", "airfoil_name", "fuselage_symmetry", "draw_backend"].forEach(id => {
        document.getElementById(id).addEventListener("input", syncGeneralFields);
        document.getElementById(id).addEventListener("change", syncGeneralFields);
      });
      document.getElementById("thin_wings").addEventListener("change", syncGeneralFields);
    }

    function syncGeneralFields() {
      state.airplane_name = document.getElementById("airplane_name").value.trim() || "My Concept";
      state.fuselage_name = document.getElementById("fuselage_name").value.trim() || "Main Fuselage";
      state.airfoil_name = document.getElementById("airfoil_name").value.trim() || "naca2412";
      state.fuselage_symmetry = document.getElementById("fuselage_symmetry").value;
      state.draw_backend = document.getElementById("draw_backend").value;
      state.thin_wings = document.getElementById("thin_wings").checked;
      redraw();
    }

    function renderFuselagePanel() {
      const columns = [
        ["x", "X"],
        ["y", "Y"],
        ["z", "Z"],
        ["radius", "Radius"],
      ];
      const panel = document.getElementById("tab-fuselage");
      panel.innerHTML = `
        <div class="stack">
          <div class="card">
            <h3>Секции фюзеляжа</h3>
            <div class="tip">Порядок строк определяет продольную форму. Можно менять длину фюзеляжа через кнопку справа.</div>
            ${tableHtml("fuselage_sections", columns, state.fuselage_sections)}
            <div class="table-actions">
              <button class="secondary" data-add-row="fuselage">Добавить секцию</button>
            </div>
          </div>
        </div>
      `;
    }

    function renderSurfacePanel(surfaceKey, title) {
      const panel = document.getElementById(`tab-${surfaceKey}`);
      const surface = state[surfaceKey];
      panel.innerHTML = `
        <div class="stack">
          <div class="card">
            <h3>${title}</h3>
            <div class="surface-top">
              <div class="field">
                <label><input type="checkbox" data-surface-toggle="${surfaceKey}" ${surface.enabled ? "checked" : ""}> Включить</label>
              </div>
              <div class="field">
                <label><input type="checkbox" data-surface-symmetry="${surfaceKey}" ${surface.symmetric ? "checked" : ""}> Симметрия</label>
              </div>
              <div class="field">
                <label>Имя</label>
                <input data-surface-name="${surfaceKey}" value="${escapeHtml(surface.name)}">
              </div>
              <div class="field">
                <label>Цвет</label>
                <input data-surface-color="${surfaceKey}" value="${escapeHtml(surface.color)}">
              </div>
            </div>
            ${tableHtml(surfaceKey, surfaceColumnDefs(), surface.sections)}
            <div class="table-actions">
              <button class="secondary" data-add-row="${surfaceKey}">Добавить секцию</button>
            </div>
          </div>
        </div>
      `;
    }

    function tableHtml(key, columns, rows) {
      const head = columns.map(([, label]) => `<th>${label}</th>`).join("");
      const body = rows.map((row, index) => `
        <tr>
          ${columns.map(([name]) =>
            `<td><input data-table="${key}" data-row="${index}" data-field="${name}" value="${escapeHtml(row[name])}"></td>`
          ).join("")}
          <td><button class="secondary" data-move-up="${key}:${index}">↑</button></td>
          <td><button class="secondary" data-move-down="${key}:${index}">↓</button></td>
          <td><button class="secondary" data-delete-row="${key}:${index}">Удалить</button></td>
        </tr>
      `).join("");
      return `
        <table>
          <thead>
            <tr>${head}<th></th><th></th><th></th></tr>
          </thead>
          <tbody>${body}</tbody>
        </table>
      `;
    }

    function bindDynamicEvents() {
      document.querySelectorAll("[data-table]").forEach(input => {
        input.addEventListener("input", event => {
          const key = event.target.dataset.table;
          const row = Number(event.target.dataset.row);
          const field = event.target.dataset.field;
          const collection = key === "fuselage_sections" ? state.fuselage_sections : state[key].sections;
          collection[row][field] = number(event.target.value, collection[row][field]);
          redraw();
        });
      });

      document.querySelectorAll("[data-add-row]").forEach(button => {
        button.addEventListener("click", () => {
          const key = button.dataset.addRow;
          if (key === "fuselage") {
            const last = state.fuselage_sections.at(-1) || { x: 0, y: 0, z: 0, radius: 0.2 };
            state.fuselage_sections.push({ x: last.x + 0.5, y: last.y, z: last.z, radius: last.radius });
          } else {
            const surface = state[key];
            const last = surface.sections.at(-1) || { x: 0, y: 0, z: 0, chord: 1, twist: 0 };
            surface.sections.push({ x: last.x + 0.3, y: last.y + 0.4, z: last.z, chord: last.chord, twist: last.twist });
          }
          rerenderEditor();
          setStatus("Добавлена новая секция.");
        });
      });

      document.querySelectorAll("[data-delete-row]").forEach(button => {
        button.addEventListener("click", () => {
          const [key, rawIndex] = button.dataset.deleteRow.split(":");
          const index = Number(rawIndex);
          const collection = key === "fuselage_sections" ? state.fuselage_sections : state[key].sections;
          collection.splice(index, 1);
          rerenderEditor();
          setStatus("Секция удалена.");
        });
      });

      document.querySelectorAll("[data-move-up]").forEach(button => {
        button.addEventListener("click", () => moveRow(button.dataset.moveUp, -1));
      });
      document.querySelectorAll("[data-move-down]").forEach(button => {
        button.addEventListener("click", () => moveRow(button.dataset.moveDown, 1));
      });

      document.querySelectorAll("[data-surface-toggle]").forEach(input => {
        input.addEventListener("change", () => {
          state[input.dataset.surfaceToggle].enabled = input.checked;
          redraw();
        });
      });
      document.querySelectorAll("[data-surface-symmetry]").forEach(input => {
        input.addEventListener("change", () => {
          state[input.dataset.surfaceSymmetry].symmetric = input.checked;
          redraw();
        });
      });
      document.querySelectorAll("[data-surface-name]").forEach(input => {
        input.addEventListener("input", () => {
          state[input.dataset.surfaceName].name = input.value.trim() || "Surface";
          redraw();
        });
      });
      document.querySelectorAll("[data-surface-color]").forEach(input => {
        input.addEventListener("input", () => {
          state[input.dataset.surfaceColor].color = input.value.trim() || "#4f7cff";
          redraw();
        });
      });
    }

    function moveRow(serialized, delta) {
      const [key, rawIndex] = serialized.split(":");
      const index = Number(rawIndex);
      const collection = key === "fuselage_sections" ? state.fuselage_sections : state[key].sections;
      const target = index + delta;
      if (target < 0 || target >= collection.length) return;
      const [item] = collection.splice(index, 1);
      collection.splice(target, 0, item);
      rerenderEditor();
    }

    function rerenderEditor() {
      syncGeneralFields();
      renderFuselagePanel();
      renderSurfacePanel("wing", "Крыло");
      renderSurfacePanel("canard", "ПГО");
      renderSurfacePanel("htail", "Горизонтальное оперение");
      renderSurfacePanel("vtail", "Вертикальное оперение");
      bindDynamicEvents();
      redraw();
    }

    function getFuselageLength(config) {
      if (!config.fuselage_sections.length) return 0;
      const xs = config.fuselage_sections.map(section => number(section.x));
      return Math.max(...xs) - Math.min(...xs);
    }

    function scaleFuselageToTargetLength() {
      const target = number(document.getElementById("fuselage_length").value, 0);
      if (state.fuselage_sections.length < 2) {
        setStatus("Для масштабирования нужно минимум 2 секции фюзеляжа.");
        return;
      }
      const xs = state.fuselage_sections.map(section => number(section.x));
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const currentLength = maxX - minX;
      if (!currentLength) {
        setStatus("Невозможно масштабировать нулевую длину.");
        return;
      }
      const scale = target / currentLength;
      state.fuselage_sections = state.fuselage_sections.map(section => ({
        ...section,
        x: minX + (number(section.x) - minX) * scale,
      }));
      rerenderEditor();
      setStatus(`Фюзеляж масштабирован до ${target.toFixed(3)}.`);
    }

    function getSurfaceProfile(surface) {
      const ys = surface.sections.map(section => number(section.y));
      const zs = surface.sections.map(section => number(section.z));
      const spanY = ys.length ? Math.max(...ys) - Math.min(...ys) : 0;
      const spanZ = zs.length ? Math.max(...zs) - Math.min(...zs) : 0;
      return {
        spanY,
        spanZ,
        isVerticalLike: spanZ > spanY * 1.2,
      };
    }

    function collectProjectionBounds(config, view) {
      const xs = [];
      const projectionValues = [];
      const fuselageChains = [
        ...(Array.isArray(config.fuselage_sections) ? [config.fuselage_sections] : []),
        ...((Array.isArray(config.extra_fuselages) ? config.extra_fuselages : []).filter(Array.isArray)),
      ];

      fuselageChains.forEach(chain => {
        chain.forEach(section => {
          const x = number(section.x);
          const radius = number(section.radius);
          xs.push(x - radius, x + radius);
          if (view === "top") {
            projectionValues.push(number(section.y) - radius, number(section.y) + radius);
          } else {
            projectionValues.push(number(section.z) - radius, number(section.z) + radius);
          }
        });
      });

      ["canard", "wing", "htail", "vtail"].forEach(key => {
        const surface = config[key];
        if (!surface.enabled) return;
        const profile = getSurfaceProfile(surface);
        surface.sections.forEach(section => {
          const x = number(section.x);
          const chord = number(section.chord);
          xs.push(x, x + chord);

          if (view === "top") {
            const y = number(section.y);
            projectionValues.push(y);
            if (surface.symmetric) projectionValues.push(-y);
            if (!profile.isVerticalLike) {
              projectionValues.push(y + chord * 0.02, y - chord * 0.02);
              if (surface.symmetric) projectionValues.push(-y + chord * 0.02, -y - chord * 0.02);
            }
          } else {
            const z = number(section.z);
            projectionValues.push(z);
            if (profile.isVerticalLike) {
              projectionValues.push(z + chord * 0.12, z - chord * 0.12);
            } else {
              projectionValues.push(z + chord * 0.03, z - chord * 0.03);
            }
          }
        });
      });

      if (!xs.length || !projectionValues.length) {
        return { minX: 0, maxX: 1, minY: -1, maxY: 1 };
      }

      return {
        minX: Math.min(...xs),
        maxX: Math.max(...xs),
        minY: Math.min(...projectionValues),
        maxY: Math.max(...projectionValues),
      };
    }

    function mapPoint(valueX, valueY, bounds, width, height, padding) {
      const spanX = Math.max(bounds.maxX - bounds.minX, 1e-6);
      const spanY = Math.max(bounds.maxY - bounds.minY, 1e-6);
      const scale = Math.min((width - 2 * padding) / spanX, (height - 2 * padding) / spanY);
      const offsetX = (width - spanX * scale) / 2;
      const offsetY = (height - spanY * scale) / 2;
      return {
        x: offsetX + (valueX - bounds.minX) * scale,
        y: height - (offsetY + (valueY - bounds.minY) * scale),
      };
    }

    function drawProjectedLine(ctx, points, color, lineWidth = 2.2, dashed = false) {
      if (points.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      if (dashed) ctx.setLineDash([8, 5]);
      ctx.stroke();
      if (dashed) ctx.setLineDash([]);
      ctx.lineWidth = 1;
    }

    function sortSurfaceSections(surface) {
      const profile = getSurfaceProfile(surface);
      return [...surface.sections].sort((left, right) => {
        if (profile.isVerticalLike) {
          return number(left.z) - number(right.z);
        }
        return Math.abs(number(left.y)) - Math.abs(number(right.y));
      });
    }

    function drawSurfaceTop(ctx, surface, bounds, width, height, padding, isOverlay = false) {
      const sections = sortSurfaceSections(surface);
      if (sections.length < 2) return;

      const profile = getSurfaceProfile(surface);
      const color = isOverlay ? "#ff8c3b" : (surface.color || "#4f7cff");

      if (profile.isVerticalLike) {
        const drawCenterline = multiplier => {
          const points = sections.map(section =>
            mapPoint(
              number(section.x) + number(section.chord) * 0.25,
              multiplier * number(section.y),
              bounds,
              width,
              height,
              padding
            )
          );
          drawProjectedLine(ctx, points, color, 2.4, isOverlay);
        };
        drawCenterline(1);
        if (surface.symmetric) drawCenterline(-1);
        return;
      }

      const drawSide = multiplier => {
        for (let i = 0; i < sections.length - 1; i++) {
          const left = sections[i];
          const right = sections[i + 1];
          const p1 = mapPoint(number(left.x), multiplier * number(left.y), bounds, width, height, padding);
          const p2 = mapPoint(number(left.x) + number(left.chord), multiplier * number(left.y), bounds, width, height, padding);
          const p3 = mapPoint(number(right.x) + number(right.chord), multiplier * number(right.y), bounds, width, height, padding);
          const p4 = mapPoint(number(right.x), multiplier * number(right.y), bounds, width, height, padding);
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.lineTo(p3.x, p3.y);
          ctx.lineTo(p4.x, p4.y);
          ctx.closePath();
          ctx.fillStyle = color;
          ctx.globalAlpha = isOverlay ? 0.14 : 0.68;
          ctx.fill();
          ctx.globalAlpha = 1;
          ctx.strokeStyle = isOverlay ? color : "#173047";
          if (isOverlay) ctx.setLineDash([8, 5]);
          ctx.stroke();
          if (isOverlay) ctx.setLineDash([]);
        }
      };

      drawSide(1);
      if (surface.symmetric) drawSide(-1);
    }

    function drawSurfaceSide(ctx, surface, bounds, width, height, padding, offsetY, isOverlay = false) {
      const sections = sortSurfaceSections(surface);
      if (sections.length < 2) return;

      const color = isOverlay ? "#ff8c3b" : (surface.color || "#4f7cff");
      const profile = getSurfaceProfile(surface);
      const mapSide = (x, z) => {
        const point = mapPoint(x, z, bounds, width, height, padding);
        return { x: point.x, y: point.y + offsetY };
      };

      if (!profile.isVerticalLike) {
        const line = sections.map(section => mapSide(number(section.x) + number(section.chord) * 0.35, number(section.z)));
        drawProjectedLine(ctx, line, color, 2.2, isOverlay);
      }

      for (let i = 0; i < sections.length - 1; i++) {
        const left = sections[i];
        const right = sections[i + 1];
        const thicknessLeft = profile.isVerticalLike ? number(left.chord) * 0.10 : number(left.chord) * 0.025;
        const thicknessRight = profile.isVerticalLike ? number(right.chord) * 0.10 : number(right.chord) * 0.025;
        const p1 = mapSide(number(left.x), number(left.z) + thicknessLeft);
        const p2 = mapSide(number(left.x) + number(left.chord), number(left.z) + thicknessLeft);
        const p3 = mapSide(number(right.x) + number(right.chord), number(right.z) + thicknessRight);
        const p4 = mapSide(number(right.x), number(right.z) + thicknessRight);
        const p5 = mapSide(number(right.x), number(right.z) - thicknessRight);
        const p6 = mapSide(number(right.x) + number(right.chord), number(right.z) - thicknessRight);
        const p7 = mapSide(number(left.x) + number(left.chord), number(left.z) - thicknessLeft);
        const p8 = mapSide(number(left.x), number(left.z) - thicknessLeft);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.lineTo(p3.x, p3.y);
        ctx.lineTo(p6.x, p6.y);
        ctx.lineTo(p7.x, p7.y);
        ctx.lineTo(p8.x, p8.y);
        ctx.lineTo(p5.x, p5.y);
        ctx.lineTo(p4.x, p4.y);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.globalAlpha = isOverlay ? 0.14 : (profile.isVerticalLike ? 0.74 : 0.42);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = isOverlay ? color : "#173047";
        if (isOverlay) ctx.setLineDash([8, 5]);
        ctx.stroke();
        if (isOverlay) ctx.setLineDash([]);
      }
    }

    function getCombinedBounds() {
      const configs = [state];
      if (optimizedState) configs.push(optimizedState);
      const projectionsTop = configs.map(config => collectProjectionBounds(config, "top"));
      const projectionsSide = configs.map(config => collectProjectionBounds(config, "side"));
      return {
        top: {
          minX: Math.min(...projectionsTop.map(item => item.minX)),
          maxX: Math.max(...projectionsTop.map(item => item.maxX)),
          minY: Math.min(...projectionsTop.map(item => item.minY)),
          maxY: Math.max(...projectionsTop.map(item => item.maxY)),
        },
        side: {
          minX: Math.min(...projectionsSide.map(item => item.minX)),
          maxX: Math.max(...projectionsSide.map(item => item.maxX)),
          minY: Math.min(...projectionsSide.map(item => item.minY)),
          maxY: Math.max(...projectionsSide.map(item => item.maxY)),
        }
      };
    }

    function drawPreview() {
      const canvas = document.getElementById("preview");
      const ctx = canvas.getContext("2d");
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * ratio;
      canvas.height = rect.height * ratio;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      const halfHeight = rect.height / 2;
      const padding = 28;
      const bounds = getCombinedBounds();
      const boundsTop = bounds.top;
      const boundsSide = bounds.side;

      ctx.fillStyle = "#28455f";
      ctx.font = "600 14px Segoe UI";
      ctx.fillText("Вид сверху", 16, 24);
      ctx.fillText("Вид сбоку", 16, halfHeight + 24);
      ctx.strokeStyle = "#cad7e4";
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(0, halfHeight);
      ctx.lineTo(rect.width, halfHeight);
      ctx.stroke();
      ctx.setLineDash([]);

      drawAircraftPreview(ctx, state, rect.width, halfHeight, padding, boundsTop, boundsSide, halfHeight, false);
      if (optimizedState) {
        drawAircraftPreview(ctx, optimizedState, rect.width, halfHeight, padding, boundsTop, boundsSide, halfHeight, true);
      }
      renderSummary();
      drawStabilityChart6();
    }

    function drawStabilityChart6() {
      const canvas = document.getElementById("stabilityChart6");
      const metricsEl = document.getElementById("stabilityMetrics");
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(320, rect.width) * ratio;
      canvas.height = rect.height * ratio;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      if (!stabilityChart6Data || !stabilityChart6Data.categories || !stabilityChart6Data.values) {
        ctx.fillStyle = "#5a7286";
        ctx.font = "14px Segoe UI";
        ctx.fillText("Нет данных: нажмите «Проверить стабильность» или завершите оптимизацию.", 16, 36);
        return;
      }

      const categories = stabilityChart6Data.categories;
      const values = stabilityChart6Data.values;
      const padL = 52;
      const padR = 16;
      const padT = 40;
      const padB = 72;
      const w = rect.width - padL - padR;
      const h = rect.height - padT - padB;
      const n = categories.length;
      const gap = 8;
      const barW = Math.max(14, (w - gap * (n - 1)) / n);

      ctx.fillStyle = "#28455f";
      ctx.font = "600 15px Segoe UI";
      ctx.fillText("График 6 — статистика ограничений (0 = нет, 1 = да)", padL, 22);

      for (let i = 0; i < n; i++) {
        const v = values[i] ? 1 : 0;
        const x0 = padL + i * (barW + gap);
        const bh = v * h;
        const y0 = padT + (h - bh);
        let fill = "#c62828";
        if (i === 0) fill = "#546e7a";
        else if (v === 1) fill = "#209460";
        ctx.fillStyle = fill;
        ctx.fillRect(x0, y0, barW, bh || 2);
        ctx.strokeStyle = "#173047";
        ctx.lineWidth = 1;
        ctx.strokeRect(x0, y0, barW, bh || 2);
        ctx.fillStyle = "#173047";
        ctx.font = "bold 12px Segoe UI";
        ctx.textAlign = "center";
        ctx.fillText(String(v), x0 + barW / 2, y0 - 6);
        ctx.save();
        ctx.translate(x0 + barW / 2, padT + h + 14);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = "right";
        ctx.font = "11px Segoe UI";
        ctx.fillStyle = "#5a7286";
        ctx.fillText(categories[i], 0, 0);
        ctx.restore();
      }

      ctx.strokeStyle = "#cad7e4";
      ctx.beginPath();
      ctx.moveTo(padL, padT + h);
      ctx.lineTo(padL + w, padT + h);
      ctx.stroke();
      ctx.textAlign = "left";
      if (metricsEl && stabilityEvaluationText) {
        metricsEl.textContent = stabilityEvaluationText;
      }
    }

    async function fetchStabilityFromServer() {
      syncGeneralFields();
      setStatus("Проверка стабильности (AeroBuildup)...");
      try {
        const data = await postJson("/api/evaluate_stability", {
          config: state,
          options: getOptimizationOptions(),
        });
        stabilityChart6Data = data.chart6 || null;
        const ev = data.evaluation || {};
        stabilityEvaluationText =
          `Score: ${(ev.score ?? 0).toFixed(3)}  |  L/D: ${(ev.efficiency ?? 0).toFixed(2)}  |  ` +
          `CL: ${(ev.CL ?? 0).toFixed(4)}  CD: ${(ev.CD ?? 0).toFixed(5)}\\n` +
          `Clp: ${(ev.Clp ?? 0).toFixed(5)} (крен)  |  Cnr: ${(ev.Cnr ?? 0).toFixed(5)} (рысканье)  |  ` +
          `Cma: ${(ev.Cma ?? 0).toFixed(4)} (тангаж)  |  Валидность ТЗ: ${ev.is_valid ? "да" : "нет"}`;
        drawStabilityChart6();
        setStatus("Стабильность обновлена по текущей конфигурации.");
      } catch (error) {
        setStatus(error.message);
      }
    }

    function drawAircraftPreview(ctx, config, width, height, padding, boundsTop, boundsSide, sideOffsetY, isOverlay) {
      drawTop(ctx, config, width, height, padding, boundsTop, isOverlay);
      drawSide(ctx, config, width, height, padding, boundsSide, sideOffsetY, isOverlay);
    }

    function drawTop(ctx, config, width, height, padding, bounds, isOverlay) {
      const centerY = (bounds.minY + bounds.maxY) / 2;
      const a = mapPoint(bounds.minX, centerY, bounds, width, height, padding);
      const b = mapPoint(bounds.maxX, centerY, bounds, width, height, padding);
      ctx.strokeStyle = "#d6dfe8";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.setLineDash([]);

      const fuselageChains = [
        ...(Array.isArray(config.fuselage_sections) ? [config.fuselage_sections] : []),
        ...((Array.isArray(config.extra_fuselages) ? config.extra_fuselages : []).filter(Array.isArray)),
      ];
      fuselageChains.forEach(chain => {
        const sections = [...chain].sort((lhs, rhs) => number(lhs.x) - number(rhs.x));
        for (let i = 0; i < sections.length - 1; i++) {
          const left = sections[i];
          const right = sections[i + 1];
          const p1 = mapPoint(number(left.x), number(left.y) + number(left.radius), bounds, width, height, padding);
          const p2 = mapPoint(number(right.x), number(right.y) + number(right.radius), bounds, width, height, padding);
          const p3 = mapPoint(number(right.x), number(right.y) - number(right.radius), bounds, width, height, padding);
          const p4 = mapPoint(number(left.x), number(left.y) - number(left.radius), bounds, width, height, padding);
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.lineTo(p3.x, p3.y);
          ctx.lineTo(p4.x, p4.y);
          ctx.closePath();
          ctx.fillStyle = isOverlay ? "rgba(255, 140, 59, 0.16)" : "#d8e6f6";
          ctx.fill();
          ctx.strokeStyle = isOverlay ? "#ff8c3b" : "#416688";
          if (isOverlay) ctx.setLineDash([8, 5]);
          ctx.stroke();
          if (isOverlay) ctx.setLineDash([]);
        }
      });

      ["canard", "wing", "htail", "vtail"].forEach(key => {
        const surface = config[key];
        if (surface.enabled) drawSurfaceTop(ctx, surface, bounds, width, height, padding, isOverlay);
      });
    }

    function drawSide(ctx, config, width, height, padding, bounds, offsetY, isOverlay) {
      const mapSide = (x, z) => {
        const point = mapPoint(x, z, bounds, width, height, padding);
        return { x: point.x, y: point.y + offsetY };
      };

      const centerZ = (bounds.minY + bounds.maxY) / 2;
      const a = mapSide(bounds.minX, centerZ);
      const b = mapSide(bounds.maxX, centerZ);
      ctx.strokeStyle = "#d6dfe8";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.setLineDash([]);

      const fuselageChains = [
        ...(Array.isArray(config.fuselage_sections) ? [config.fuselage_sections] : []),
        ...((Array.isArray(config.extra_fuselages) ? config.extra_fuselages : []).filter(Array.isArray)),
      ];
      fuselageChains.forEach(chain => {
        const top = [];
        const bottom = [];
        [...chain].sort((lhs, rhs) => number(lhs.x) - number(rhs.x)).forEach(section => {
          top.push(mapSide(number(section.x), number(section.z) + number(section.radius)));
          bottom.push(mapSide(number(section.x), number(section.z) - number(section.radius)));
        });
        if (top.length >= 2) {
          ctx.beginPath();
          ctx.moveTo(top[0].x, top[0].y);
          top.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
          bottom.reverse().forEach(point => ctx.lineTo(point.x, point.y));
          ctx.closePath();
          ctx.fillStyle = isOverlay ? "rgba(255, 140, 59, 0.16)" : "#d8e6f6";
          ctx.fill();
          ctx.strokeStyle = isOverlay ? "#ff8c3b" : "#416688";
          if (isOverlay) ctx.setLineDash([8, 5]);
          ctx.stroke();
          if (isOverlay) ctx.setLineDash([]);
        }
      });

      ["canard", "wing", "htail", "vtail"].forEach(key => {
        const surface = config[key];
        if (surface.enabled) drawSurfaceSide(ctx, surface, bounds, width, height, padding, offsetY, isOverlay);
      });
    }

    function renderSummary() {
      const enabled = ["canard", "wing", "htail", "vtail"]
        .filter(key => state[key].enabled)
        .map(key => state[key].name)
        .join(", ") || "нет";
      document.getElementById("summary").textContent =
        `Самолёт: ${state.airplane_name}\\n` +
        `Профиль: ${state.airfoil_name}\\n` +
        `Длина фюзеляжа: ${getFuselageLength(state).toFixed(3)}\\n` +
        `Активные поверхности: ${enabled}\\n` +
        `Секций фюзеляжа: ${state.fuselage_sections.length}`;

      const optimizationSummary = document.getElementById("optimizationSummary");
      if (!optimizedState || !optimizationResult) {
        optimizationSummary.textContent = "Оптимизация ещё не запускалась.";
        return;
      }
      
      const evaluation = optimizationResult;
      const currentLength = getFuselageLength(state).toFixed(3);
      const optimizedLength = getFuselageLength(optimizedState).toFixed(3);
      
      // Обработка разных форматов результатов (AI/Algorithm/Hybrid)
      let summary = "Оптимизированная конфигурация\\n";
      
      if (evaluation.score !== undefined) {
        summary += `Score: ${evaluation.score.toFixed(3)}\\n`;
      }
      
      if (evaluation.efficiency !== undefined) {
        summary += `Efficiency (CL/CD): ${evaluation.efficiency.toFixed(3)}\\n`;
      } else if (evaluation.L_over_D !== undefined) {
        summary += `L/D: ${evaluation.L_over_D.toFixed(3)}\\n`;
      }
      
      summary += `CL: ${(evaluation.CL || evaluation.Cl || 0).toFixed(3)} | `;
      summary += `CD: ${(evaluation.CD || evaluation.Cd || 0).toFixed(4)}\\n`;
      summary += `Cm: ${(evaluation.Cm || 0).toFixed(3)} | `;
      summary += `Cma: ${(evaluation.Cma || evaluation.Cma || 0).toFixed(3)}\\n`;
      
      if (evaluation.Clp !== undefined) {
        summary += `Clp (roll damping): ${evaluation.Clp.toFixed(4)} ${evaluation.Clp < -0.01 ? "✓" : "✗"}\\n`;
      }
      if (evaluation.Cnr !== undefined) {
        summary += `Cnr (yaw stability): ${evaluation.Cnr.toFixed(4)} ${evaluation.Cnr < -0.01 ? "✓" : "✗"}\\n`;
      }
      if (evaluation.constraints_violation !== undefined) {
        summary += `Ограничения: ${evaluation.constraints_violation.toFixed(3)} ${evaluation.constraints_violation <= 1e-6 ? "✓" : "✗"}\\n`;
      }
      if (evaluation.is_valid !== undefined) {
        summary += `Валидность ТЗ: ${evaluation.is_valid ? "✓" : "✗"}\\n`;
      }
      
      if (evaluation.wing_area !== undefined) {
        summary += `Wing area: ${evaluation.wing_area.toFixed(3)}\\n`;
      }
      
      if (evaluation.alpha !== undefined) {
        summary += `Alpha: ${evaluation.alpha.toFixed(2)} deg\\n`;
      }
      
      summary += `Длина текущая/оптимум: ${currentLength} / ${optimizedLength}`;
      
      optimizationSummary.textContent = summary;
    }

    function redraw() {
      drawPreview();
    }

    function downloadText(filename, text, mimeType) {
      const blob = new Blob([text], { type: mimeType });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      link.click();
      URL.revokeObjectURL(link.href);
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Ошибка запроса");
      return data;
    }

    function getOptimizationOptions() {
      return {
        mode: document.getElementById("optMode").value || "algorithm",
        iterations: Math.max(1, Math.round(number(document.getElementById("optIterations").value, 10))),
        population: Math.max(4, Math.round(number(document.getElementById("optPopulation").value, 20))),
        min_wing_area: number(document.getElementById("optMinWingArea").value, 0.3),
        target_cl: number(document.getElementById("optTargetCl").value, 0.55),
        velocity: number(document.getElementById("optVelocity").value, 50),
      };
    }

    async function runOptimization() {
      syncGeneralFields();
      const mode = document.getElementById("optMode").value || "algorithm";
      const modeNames = {
        "algorithm": "🧬 Алгоритм (CEM)",
        "ai": "🤖 AI (Нейросеть)",
        "ai+algorithm": "🔄 Гибридный (AI+CEM)",
      };
      setStatus(`Оптимизация (${modeNames[mode]})...`);
      try {
        const data = await postJson("/api/optimize", {
          config: state,
          options: getOptimizationOptions(),
        });
        if (data.error) {
          setStatus(`Ошибка: ${data.error}`);
          return;
        }
        optimizedState = data.optimized_config;
        optimizationResult = data.evaluation;
        if (data.chart6) {
          stabilityChart6Data = data.chart6;
          const ev = data.evaluation || {};
          stabilityEvaluationText =
            `Оптимум — Score: ${(ev.score ?? 0).toFixed(3)}  |  L/D: ${(ev.efficiency ?? 0).toFixed(2)}  |  ` +
            `CL: ${(ev.CL ?? 0).toFixed(4)}  CD: ${(ev.CD ?? 0).toFixed(5)}\\n` +
            `Clp: ${(ev.Clp ?? 0).toFixed(5)}  |  Cnr: ${(ev.Cnr ?? 0).toFixed(5)}  |  Cma: ${(ev.Cma ?? 0).toFixed(4)}  |  ТЗ: ${ev.is_valid ? "да" : "нет"}`;
        }
        redraw();
        const modeStage = data.evaluation.mode ? `[${modeNames[data.evaluation.mode]}]` : "";
        setStatus(`✓ Оптимизация завершена ${modeStage}. Оранжевый пунктир показывает найденный вариант.`);
      } catch (error) {
        setStatus(error.message);
      }
    }

    function applyOptimizedState() {
      if (!optimizedState) {
        setStatus("Сначала нужно получить оптимизированную конфигурацию.");
        return;
      }
      state = deepClone(optimizedState);
      renderAll();
      setStatus("Оптимизированная геометрия применена и доступна для ручного редактирования.");
    }

    function clearOptimizedState() {
      optimizedState = null;
      optimizationResult = null;
      redraw();
      setStatus("Оптимизированный слой скрыт.");
    }

    function clearStabilityChart() {
      stabilityChart6Data = null;
      stabilityEvaluationText = "";
      const metricsEl = document.getElementById("stabilityMetrics");
      if (metricsEl) metricsEl.textContent = "Метрики появятся после проверки.";
      drawStabilityChart6();
    }

    function wireToolbar() {
      document.getElementById("newProjectBtn").addEventListener("click", () => {
        state = deepClone(defaultConfig);
        optimizedState = null;
        optimizationResult = null;
        clearStabilityChart();
        renderAll();
        setStatus("Создан новый проект.");
      });
      document.getElementById("refreshBtn").addEventListener("click", redraw);
      document.getElementById("stabilityBtn").addEventListener("click", fetchStabilityFromServer);
      document.getElementById("scaleFuselageBtn").addEventListener("click", scaleFuselageToTargetLength);
      document.getElementById("optimizeBtn").addEventListener("click", runOptimization);
      document.getElementById("applyOptimizedBtn").addEventListener("click", applyOptimizedState);
      document.getElementById("clearOptimizedBtn").addEventListener("click", clearOptimizedState);
      document.getElementById("saveJsonBtn").addEventListener("click", () => {
        syncGeneralFields();
        downloadText("aircraft_config.json", JSON.stringify(state, null, 2), "application/json");
        setStatus("JSON выгружен.");
      });
      document.getElementById("loadJsonBtn").addEventListener("click", () => {
        document.getElementById("jsonFileInput").click();
      });
      document.getElementById("jsonFileInput").addEventListener("change", async event => {
        const file = event.target.files[0];
        if (!file) return;
        try {
          state = JSON.parse(await file.text());
          optimizedState = null;
          optimizationResult = null;
          clearStabilityChart();
          renderAll();
          setStatus(`JSON загружен: ${file.name}`);
        } catch (error) {
          setStatus(`Ошибка загрузки JSON: ${error.message}`);
        } finally {
          event.target.value = "";
        }
      });
      document.getElementById("exportPyBtn").addEventListener("click", async () => {
        try {
          syncGeneralFields();
          const data = await postJson("/api/export_python", state);
          downloadText("generated_aircraft.py", data.script, "text/x-python");
          setStatus("Python-скрипт экспортирован.");
        } catch (error) {
          setStatus(error.message);
        }
      });
      document.getElementById("renderBtn").addEventListener("click", async () => {
        try {
          syncGeneralFields();
          setStatus("Открываю 3D визуализацию...");
          const data = await postJson("/api/render", state);
          setStatus(data.message);
        } catch (error) {
          setStatus(error.message);
        }
      });
    }

    function renderAll() {
      renderTabs();
      renderGeneralPanel();
      rerenderEditor();
      renderSummary();
    }

    window.addEventListener("resize", redraw);
    renderAll();
    wireToolbar();
  </script>
</body>
</html>
"""


def launch_render_process(config: AircraftConfig) -> Path:
    script = generate_python_script(config)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = Path(f.name)
    subprocess.Popen([sys.executable, str(script_path)])
    return script_path


class DesignerHandler(BaseHTTPRequestHandler):
    server_version = "AircraftDesigner/1.0"

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str) -> None:
        payload = body.replace("__DEFAULT_CONFIG__", json.dumps(config_to_dict(default_config()), ensure_ascii=False))
        data = payload.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_html(html_page())
            return
        if parsed.path == "/api/default_config":
            self._send_json({"config": config_to_dict(default_config())})
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        try:
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw)
        except Exception as exc:
            self._send_json({"error": f"Некорректный JSON: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            if parsed.path == "/api/export_python":
                config = config_from_dict(payload)
                self._send_json({"script": generate_python_script(config)})
                return
            if parsed.path == "/api/render":
                config = config_from_dict(payload)
                script_path = launch_render_process(config)
                self._send_json(
                    {
                        "message": "3D визуализация запущена в отдельном Python-процессе.",
                        "script_path": str(script_path),
                    }
                )
                return
            if parsed.path == "/api/evaluate_stability":
                import SCAT.AERO.wing_optimizer as wing_optimizer_mod

                config = config_from_dict(payload.get("config", {}))
                opts = payload.get("options", {})
                velocity = float(opts.get("velocity", 50.0))
                target_cl = float(opts.get("target_cl", 0.55))
                min_wing_area = float(opts.get("min_wing_area", 0.3))
                alpha = float(opts.get("alpha", 0.0))
                span = wing_optimizer_mod.extract_wing_metrics(config.wing).span
                ev = wing_optimizer_mod.evaluate_wing_design(
                    config,
                    velocity=velocity,
                    target_cl=target_cl,
                    min_wing_area=min_wing_area,
                    max_wing_span=span,
                    alpha=alpha,
                    relax_mode=False,
                )
                chart6 = wing_optimizer_mod.stability_chart6_payload(
                    ev,
                    min_wing_area=min_wing_area,
                    max_wing_span=span,
                )
                st = ev.stability
                self._send_json(
                    {
                        "chart6": chart6,
                        "evaluation": {
                            "score": float(ev.score),
                            "efficiency": float(ev.efficiency),
                            "CL": float(st.Cl),
                            "CD": float(st.Cd),
                            "Cm": float(st.Cm),
                            "Cma": float(st.Cma),
                            "Clp": float(st.Clp),
                            "Cnr": float(st.Cnr),
                            "wing_area": float(ev.wing.area),
                            "wing_span": float(ev.wing.span),
                            "geometry_penalty": float(ev.geometry_penalty),
                            "constraints_violation": float(ev.constraints_violation),
                            "is_valid": ev.is_valid(min_wing_area=min_wing_area, max_wing_span=span),
                        },
                    }
                )
                return
            if parsed.path == "/api/optimize":
                import SCAT.AERO.wing_optimization_v2 as opt_engine

                config = config_from_dict(payload.get("config", {}))
                options = payload.get("options", {})
                
                # Использовать новый оптимизатор с поддержкой разных режимов
                engine = opt_engine.WingOptimizationEngine(config)
                result = engine.optimize(
                    mode=options.get("mode", "algorithm"),
                    iterations=max(1, int(options.get("iterations", 10))),
                    population=max(4, int(options.get("population", 20))),
                    velocity=float(options.get("velocity", 50.0)),
                    target_cl=float(options.get("target_cl", 0.55)),
                    min_wing_area=float(options.get("min_wing_area", 0.3)),
                    seed=42,
                )
                
                self._send_json(result)
                return
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:
        return


def run_server(port: int = 8765, open_browser: bool = True) -> None:
    class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    selected_port = port
    httpd = None
    for candidate_port in [port, *range(port + 1, port + 20)]:
        try:
            httpd = ReusableThreadingTCPServer(("127.0.0.1", candidate_port), DesignerHandler)
            selected_port = candidate_port
            break
        except OSError as exc:
            if exc.errno != 48:
                raise

    if httpd is None:
        raise OSError(f"Не удалось найти свободный порт в диапазоне {port}-{port + 19}.")

    with httpd:
        url = f"http://127.0.0.1:{selected_port}"
        print(f"Aircraft Designer запущен: {url}")
        if selected_port != port:
            print(f"Порт {port} был занят, поэтому использован {selected_port}.")
        print("Остановить сервер: Ctrl+C")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальный GUI-конструктор самолёта на AeroSandbox.")
    parser.add_argument("--port", type=int, default=8765, help="Предпочитаемый порт для веб-интерфейса.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Не открывать браузер автоматически.",
    )
    args = parser.parse_args()
    run_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
