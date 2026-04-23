from __future__ import annotations

import argparse
import json
import os
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


def default_config() -> AircraftConfig:
    return AircraftConfig(
        fuselage_sections=[
            FuselageSection(0.0, 0.0, 0.0, 0.10),
            FuselageSection(0.5, 0.0, 0.0, 0.25),
            FuselageSection(2.0, 0.0, 0.0, 0.25),
            FuselageSection(4.0, 0.0, 0.0, 0.25),
            FuselageSection(5.5, 0.0, 0.0, 0.01),
        ],
        wing=SurfaceConfig(
            name="Swept Wing",
            enabled=True,
            symmetric=True,
            color="#2468f2",
            sections=[
                SurfaceSection(4.0, 0.0, 0.0, 1.5, 2.0),
                SurfaceSection(4.3, 2.0, 0.1, 1.0, -1.5),
                SurfaceSection(5.0, 2.5, 0.4, 0.5, -3.0),
            ],
        ),
        canard=SurfaceConfig(
            name="All-Moving Canard",
            enabled=True,
            symmetric=True,
            color="#ff8c3b",
            sections=[
                SurfaceSection(0.9, 0.1, -0.05, 0.7, 0.0),
                SurfaceSection(1.2, 0.9, 0.0, 0.5, 0.0),
                SurfaceSection(1.5, 1.4, 0.07, 0.3, 1.0),
            ],
        ),
        htail=SurfaceConfig(
            name="Horizontal Tail",
            enabled=False,
            symmetric=True,
            color="#7e57ff",
            sections=[
                SurfaceSection(4.6, 0.0, 0.15, 0.8, 0.0),
                SurfaceSection(4.9, 1.0, 0.2, 0.45, -1.0),
            ],
        ),
        vtail=SurfaceConfig(
            name="Vertical Tail",
            enabled=False,
            symmetric=False,
            color="#26a269",
            sections=[
                SurfaceSection(4.7, 0.0, 0.10, 0.95, 0.0),
                SurfaceSection(5.0, 0.0, 0.95, 0.35, 0.0),
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
    fuselage = asb.Fuselage(
        name=config.fuselage_name,
        xsecs=[
            asb.FuselageXSec(xyz_c=[section.x, section.y, section.z], radius=section.radius)
            for section in config.fuselage_sections
        ],
        symmetry=None if config.fuselage_symmetry == "none" else config.fuselage_symmetry,
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

    return asb.Airplane(name=config.airplane_name, wings=wings, fuselages=[fuselage])


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
fuselage = asb.Fuselage(
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
    fuselages=[fuselage],
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
            <div class="field">
              <label for="optIterations">Итерации</label>
              <input id="optIterations" type="number" value="10" min="1" step="1">
            </div>
            <div class="field">
              <label for="optPopulation">Популяция</label>
              <input id="optPopulation" type="number" value="20" min="4" step="1">
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

    let state = structuredClone(defaultConfig);
    let optimizedState = null;
    let optimizationResult = null;

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
          <div class="card">
            <h3>Параметры оптимизации</h3>
            <div class="grid">
              ${inputField("optIterations", "Итерации", "5", "number")}
              ${inputField("optPopulation", "Популяция", "10", "number")}
              ${inputField("optTargetCl", "Целевой CL", "0.55", "number")}
              ${inputField("optVelocity", "Скорость (м/с)", "50", "number")}
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
      ["airplane_name", "fuselage_name", "airfoil_name", "fuselage_symmetry", "draw_backend", "optIterations", "optPopulation", "optTargetCl", "optVelocity"].forEach(id => {
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

      config.fuselage_sections.forEach(section => {
        const x = number(section.x);
        const radius = number(section.radius);
        xs.push(x - radius, x + radius);
        if (view === "top") {
          projectionValues.push(number(section.y) - radius, number(section.y) + radius);
        } else {
          projectionValues.push(number(section.z) - radius, number(section.z) + radius);
        }
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

      const sections = [...config.fuselage_sections].sort((lhs, rhs) => number(lhs.x) - number(rhs.x));
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

      const top = [];
      const bottom = [];
      [...config.fuselage_sections].sort((lhs, rhs) => number(lhs.x) - number(rhs.x)).forEach(section => {
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
        `Самолёт: ${state.airplane_name}\n` +
        `Профиль: ${state.airfoil_name}\n` +
        `Длина фюзеляжа: ${getFuselageLength(state).toFixed(3)}\n` +
        `Активные поверхности: ${enabled}\n` +
        `Секций фюзеляжа: ${state.fuselage_sections.length}`;

      const optimizationSummary = document.getElementById("optimizationSummary");
      if (!optimizedState || !optimizationResult) {
        optimizationSummary.textContent = "Оптимизация ещё не запускалась.";
        return;
      }
      const wing = optimizationResult.geometry?.wing;
      const canard = optimizationResult.geometry?.canard;
      const currentLength = getFuselageLength(state).toFixed(3);
      const optimizedLength = getFuselageLength(optimizedState).toFixed(3);
      optimizationSummary.textContent =
        `Оптимизированная конфигурация\n` +
        `Score: ${optimizationResult.score.toFixed(3)}\n` +
        `L/D: ${optimizationResult.best_point.L_over_D.toFixed(3)}\n` +
        `CL: ${optimizationResult.best_point.CL.toFixed(3)} | CD: ${optimizationResult.best_point.CD.toFixed(4)}\n` +
        `Cm: ${optimizationResult.best_point.Cm.toFixed(3)} | Cma: ${optimizationResult.best_point.Cma.toFixed(3)}\n` +
        `Alpha: ${optimizationResult.best_point.alpha.toFixed(2)} deg\n` +
        (wing ? `Крыло: span=${wing.full_span.toFixed(3)}, area=${wing.area.toFixed(3)}, root=${wing.root_chord.toFixed(3)}, tip=${wing.tip_chord.toFixed(3)}\n` : "") +
        (canard ? `ПГО: span=${canard.full_span.toFixed(3)}, area=${canard.area.toFixed(3)}, root=${canard.root_chord.toFixed(3)}, tip=${canard.tip_chord.toFixed(3)}\n` : "") +
        `Длина текущая/оптимум: ${currentLength} / ${optimizedLength}`;
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
        iterations: Math.max(1, Math.round(number(document.getElementById("optIterations").value, 10))),
        population: Math.max(4, Math.round(number(document.getElementById("optPopulation").value, 20))),
        target_cl: number(document.getElementById("optTargetCl").value, 0.55),
        velocity: number(document.getElementById("optVelocity").value, 50),
      };
    }

    async function runOptimization() {
      syncGeneralFields();
      setStatus("Идёт оптимизация геометрии...");
      try {
        const data = await postJson("/api/optimize", {
          config: state,
          options: getOptimizationOptions(),
        });
        optimizedState = data.optimized_config;
        optimizationResult = data.evaluation;
        redraw();
        setStatus("Оптимизация завершена. Оранжевый пунктир показывает найденный вариант.");
      } catch (error) {
        setStatus(error.message);
      }
    }

    function applyOptimizedState() {
      if (!optimizedState) {
        setStatus("Сначала нужно получить оптимизированную конфигурацию.");
        return;
      }
      state = structuredClone(optimizedState);
      renderAll();
      setStatus("Оптимизированная геометрия применена и доступна для ручного редактирования.");
    }

    function clearOptimizedState() {
      optimizedState = null;
      optimizationResult = null;
      redraw();
      setStatus("Оптимизированный слой скрыт.");
    }

    function wireToolbar() {
      document.getElementById("newProjectBtn").addEventListener("click", () => {
        state = structuredClone(defaultConfig);
        optimizedState = null;
        optimizationResult = null;
        renderAll();
        setStatus("Создан новый проект.");
      });
      document.getElementById("refreshBtn").addEventListener("click", redraw);
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
            if parsed.path == "/api/optimize":
                import SCAT.AERO.optimizer as optimizer

                config = config_from_dict(payload.get("config", {}))
                options = payload.get("options", {})
                best_vector, evaluation = optimizer.optimize_design(
                    base_config=config,
                    iterations=max(1, int(options.get("iterations", 10))),
                    population=max(4, int(options.get("population", 20))),
                    elite_fraction=0.25,
                    seed=42,
                    velocity=float(options.get("velocity", 50.0)),
                    target_cl=float(options.get("target_cl", 0.55)),
                )
                optimized_config = optimizer.apply_design_vector(config, best_vector)
                self._send_json(
                    {
                        "optimized_config": config_to_dict(optimized_config),
                        "evaluation": evaluation,
                        "parameters": optimizer.vector_to_dict(best_vector),
                    }
                )
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
