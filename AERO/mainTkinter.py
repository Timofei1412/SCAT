from __future__ import annotations

import json
import math
import sys
import tkinter as tk
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


def import_aerosandbox():
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
            color="#2a78ff",
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
            color="#ff8a3d",
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
            color="#8561ff",
            sections=[
                SurfaceSection(4.6, 0.0, 0.15, 0.8, 0.0),
                SurfaceSection(4.9, 1.0, 0.2, 0.45, -1.0),
            ],
        ),
        vtail=SurfaceConfig(
            name="Vertical Tail",
            enabled=False,
            symmetric=False,
            color="#28a36a",
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


def load_surface(data: dict) -> SurfaceConfig:
    surface = SurfaceConfig(
        name=data.get("name", "Surface"),
        enabled=data.get("enabled", True),
        symmetric=data.get("symmetric", True),
        color=data.get("color", "#4f7cff"),
    )
    surface.sections = [SurfaceSection(**section) for section in data.get("sections", [])]
    return surface


def config_from_dict(data: dict) -> AircraftConfig:
    config = AircraftConfig(
        airplane_name=data.get("airplane_name", "My Concept"),
        airfoil_name=data.get("airfoil_name", "naca2412"),
        fuselage_name=data.get("fuselage_name", "Main Fuselage"),
        fuselage_symmetry=data.get("fuselage_symmetry", "XZ"),
        draw_backend=data.get("draw_backend", "pyvista"),
        thin_wings=data.get("thin_wings", False),
    )
    config.fuselage_sections = [FuselageSection(**section) for section in data.get("fuselage_sections", [])]
    config.wing = load_surface(data.get("wing", {}))
    config.canard = load_surface(data.get("canard", {}))
    config.htail = load_surface(data.get("htail", {}))
    config.vtail = load_surface(data.get("vtail", {}))
    return config


class SectionEditor(ttk.LabelFrame):
    def __init__(
        self,
        master: tk.Misc,
        title: str,
        columns: list[tuple[str, str]],
        defaults: dict[str, float],
        on_change,
    ) -> None:
        super().__init__(master, text=title, padding=10)
        self.columns = columns
        self.defaults = defaults
        self.on_change = on_change
        self.entries: dict[str, ttk.Entry] = {}

        self.tree = ttk.Treeview(
            self,
            columns=[name for name, _ in columns],
            show="headings",
            height=8,
            selectmode="browse",
        )
        for name, heading in columns:
            self.tree.heading(name, text=heading)
            self.tree.column(name, width=84, anchor="center")
        self.tree.bind("<<TreeviewSelect>>", self._load_selected_row)

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, columnspan=6, sticky="nsew")
        scrollbar.grid(row=0, column=6, sticky="ns")

        form = ttk.Frame(self)
        form.grid(row=1, column=0, columnspan=7, sticky="ew", pady=(10, 0))
        for idx, (name, heading) in enumerate(columns):
            ttk.Label(form, text=heading).grid(row=0, column=idx, sticky="w", padx=(0, 6))
            entry = ttk.Entry(form, width=10)
            entry.grid(row=1, column=idx, sticky="ew", padx=(0, 6))
            self.entries[name] = entry

        buttons = ttk.Frame(self)
        buttons.grid(row=2, column=0, columnspan=7, sticky="ew", pady=(10, 0))
        ttk.Button(buttons, text="Добавить", command=self.add_row).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(buttons, text="Обновить", command=self.update_row).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(buttons, text="Удалить", command=self.delete_row).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(buttons, text="Вверх", command=lambda: self.move_row(-1)).grid(row=0, column=3, padx=(0, 6))
        ttk.Button(buttons, text="Вниз", command=lambda: self.move_row(1)).grid(row=0, column=4)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._set_default_values()

    def _set_default_values(self) -> None:
        for name, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, str(self.defaults.get(name, 0.0)))

    def _load_selected_row(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        values = self.tree.item(selected[0], "values")
        for (name, _), value in zip(self.columns, values):
            self.entries[name].delete(0, tk.END)
            self.entries[name].insert(0, str(value))

    def _read_entries(self) -> list[float]:
        values = []
        for name, _ in self.columns:
            raw = self.entries[name].get().strip().replace(",", ".")
            if raw == "":
                raise ValueError(f"Поле {name} не должно быть пустым.")
            values.append(float(raw))
        return values

    def add_row(self) -> None:
        try:
            values = self._read_entries()
        except ValueError as exc:
            messagebox.showerror("Ошибка ввода", str(exc), parent=self)
            return
        item_id = self.tree.insert("", tk.END, values=[f"{value:.4f}" for value in values])
        self.tree.selection_set(item_id)
        self.tree.focus(item_id)
        self.on_change()

    def update_row(self) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Обновление", "Выбери строку для обновления.", parent=self)
            return
        try:
            values = self._read_entries()
        except ValueError as exc:
            messagebox.showerror("Ошибка ввода", str(exc), parent=self)
            return
        self.tree.item(selected[0], values=[f"{value:.4f}" for value in values])
        self.on_change()

    def delete_row(self) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        self.tree.delete(selected[0])
        self.on_change()

    def move_row(self, direction: int) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        item_id = selected[0]
        index = self.tree.index(item_id)
        target = index + direction
        if target < 0 or target >= len(self.tree.get_children()):
            return
        self.tree.move(item_id, "", target)
        self.on_change()

    def set_rows(self, rows: list[dict[str, float]]) -> None:
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            values = [f"{float(row[name]):.4f}" for name, _ in self.columns]
            self.tree.insert("", tk.END, values=values)
        self._set_default_values()

    def get_rows(self) -> list[dict[str, float]]:
        rows = []
        for item_id in self.tree.get_children():
            item = self.tree.item(item_id, "values")
            row = {}
            for (name, _), value in zip(self.columns, item):
                row[name] = float(value)
            rows.append(row)
        return rows


class SurfaceFrame(ttk.Frame):
    def __init__(self, master: tk.Misc, title: str, color: str, on_change) -> None:
        super().__init__(master, padding=10)
        self.on_change = on_change

        self.enabled_var = tk.BooleanVar(value=True)
        self.symmetric_var = tk.BooleanVar(value=True)
        self.name_var = tk.StringVar(value=title)
        self.color_var = tk.StringVar(value=color)

        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(header, text="Включить поверхность", variable=self.enabled_var, command=self.on_change).grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        ttk.Checkbutton(header, text="Симметрия по Y", variable=self.symmetric_var, command=self.on_change).grid(
            row=0, column=1, sticky="w", padx=(0, 12)
        )
        ttk.Label(header, text="Имя").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(header, textvariable=self.name_var, width=28).grid(row=2, column=0, sticky="ew", padx=(0, 12))
        ttk.Label(header, text="Цвет превью").grid(row=1, column=1, sticky="w", pady=(10, 0))
        ttk.Entry(header, textvariable=self.color_var, width=14).grid(row=2, column=1, sticky="w")

        self.editor = SectionEditor(
            self,
            title="Секции поверхности",
            columns=[
                ("x", "X LE"),
                ("y", "Y LE"),
                ("z", "Z LE"),
                ("chord", "Chord"),
                ("twist", "Twist"),
            ],
            defaults={"x": 0.0, "y": 0.0, "z": 0.0, "chord": 1.0, "twist": 0.0},
            on_change=self.on_change,
        )
        self.editor.pack(fill="both", expand=True)

        for variable in (self.name_var, self.color_var):
            variable.trace_add("write", lambda *_: self.on_change())

    def set_surface(self, surface: SurfaceConfig) -> None:
        self.enabled_var.set(surface.enabled)
        self.symmetric_var.set(surface.symmetric)
        self.name_var.set(surface.name)
        self.color_var.set(surface.color)
        self.editor.set_rows([asdict(section) for section in surface.sections])

    def get_surface(self) -> SurfaceConfig:
        rows = self.editor.get_rows()
        return SurfaceConfig(
            name=self.name_var.get().strip() or "Surface",
            enabled=self.enabled_var.get(),
            symmetric=self.symmetric_var.get(),
            color=self.color_var.get().strip() or "#4f7cff",
            sections=[SurfaceSection(**row) for row in rows],
        )


class AircraftDesignerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AeroSandbox Aircraft Designer")
        self.root.geometry("1460x860")
        self.root.minsize(1200, 760)
        self.current_file: Path | None = None

        self.config = default_config()

        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self._build_layout()
        self.load_config_into_ui(self.config)
        self.refresh_preview()

    def _build_layout(self) -> None:
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill="both", expand=True)

        self.left_panel = ttk.Frame(main, padding=10)
        self.right_panel = ttk.Frame(main, padding=10)
        main.add(self.left_panel, weight=3)
        main.add(self.right_panel, weight=2)

        self._build_toolbar()
        self._build_notebook()
        self._build_preview()

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self.left_panel)
        toolbar.pack(fill="x", pady=(0, 10))

        ttk.Button(toolbar, text="Новый проект", command=self.reset_project).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Открыть JSON", command=self.load_project).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Сохранить JSON", command=self.save_project).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Экспорт Python", command=self.export_python).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="3D визуализация", command=self.render_aircraft).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Обновить превью", command=self.refresh_preview).pack(side="left")

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.left_panel)
        self.notebook.pack(fill="both", expand=True)

        self.general_tab = ttk.Frame(self.notebook, padding=12)
        self.fuselage_tab = ttk.Frame(self.notebook, padding=12)
        self.wing_tab = SurfaceFrame(self.notebook, "Main Wing", "#2a78ff", self.refresh_preview)
        self.canard_tab = SurfaceFrame(self.notebook, "Canard", "#ff8a3d", self.refresh_preview)
        self.htail_tab = SurfaceFrame(self.notebook, "Horizontal Tail", "#8561ff", self.refresh_preview)
        self.vtail_tab = SurfaceFrame(self.notebook, "Vertical Tail", "#28a36a", self.refresh_preview)

        self.notebook.add(self.general_tab, text="Общее")
        self.notebook.add(self.fuselage_tab, text="Фюзеляж")
        self.notebook.add(self.wing_tab, text="Крыло")
        self.notebook.add(self.canard_tab, text="ПГО")
        self.notebook.add(self.htail_tab, text="ГО")
        self.notebook.add(self.vtail_tab, text="ВО")

        self._build_general_tab()
        self._build_fuselage_tab()

    def _build_general_tab(self) -> None:
        self.airplane_name_var = tk.StringVar()
        self.airfoil_var = tk.StringVar()
        self.fuselage_name_var = tk.StringVar()
        self.fuselage_symmetry_var = tk.StringVar(value="XZ")
        self.draw_backend_var = tk.StringVar(value="pyvista")
        self.thin_wings_var = tk.BooleanVar(value=False)
        self.fuselage_length_var = tk.StringVar(value="5.5")

        fields = ttk.LabelFrame(self.general_tab, text="Параметры модели", padding=10)
        fields.pack(fill="x")

        labels = [
            ("Имя самолёта", self.airplane_name_var),
            ("Имя фюзеляжа", self.fuselage_name_var),
            ("Профиль", self.airfoil_var),
            ("Симметрия фюзеляжа", self.fuselage_symmetry_var),
            ("Backend", self.draw_backend_var),
            ("Длина фюзеляжа", self.fuselage_length_var),
        ]
        for row, (label, variable) in enumerate(labels):
            ttk.Label(fields, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
            if variable is self.fuselage_symmetry_var:
                ttk.Combobox(fields, textvariable=variable, values=["none", "XZ", "XY", "YZ"], state="readonly").grid(
                    row=row, column=1, sticky="ew", pady=4
                )
            elif variable is self.draw_backend_var:
                ttk.Combobox(fields, textvariable=variable, values=["pyvista"], state="readonly").grid(
                    row=row, column=1, sticky="ew", pady=4
                )
            else:
                ttk.Entry(fields, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)

        ttk.Checkbutton(
            fields,
            text="Рисовать тонкие крылья в AeroSandbox",
            variable=self.thin_wings_var,
            command=self.refresh_preview,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

        controls = ttk.Frame(fields)
        controls.grid(row=7, column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Button(controls, text="Масштабировать фюзеляж по длине", command=self.scale_fuselage_length).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(controls, text="Применить и обновить", command=self.refresh_preview).pack(side="left")
        fields.columnconfigure(1, weight=1)

        summary = ttk.LabelFrame(self.general_tab, text="Что можно настраивать", padding=10)
        summary.pack(fill="both", expand=True, pady=(12, 0))
        ttk.Label(
            summary,
            justify="left",
            text=(
                "• Любое количество секций фюзеляжа и поверхностей.\n"
                "• Отдельные параметры для крыла, ПГО, горизонтального и вертикального оперения.\n"
                "• Длину фюзеляжа можно менять автоматически масштабированием по X.\n"
                "• Конфигурацию можно сохранить в JSON или экспортировать в чистый Python-код."
            ),
        ).pack(anchor="w")

        for variable in (
            self.airplane_name_var,
            self.airfoil_var,
            self.fuselage_name_var,
            self.fuselage_symmetry_var,
            self.draw_backend_var,
            self.fuselage_length_var,
        ):
            variable.trace_add("write", lambda *_: self.refresh_preview())

    def _build_fuselage_tab(self) -> None:
        header = ttk.Frame(self.fuselage_tab)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(
            header,
            text="Секции фюзеляжа задаются координатой центра и радиусом. Порядок строк важен.",
        ).pack(anchor="w")

        self.fuselage_editor = SectionEditor(
            self.fuselage_tab,
            title="Сечения фюзеляжа",
            columns=[
                ("x", "X"),
                ("y", "Y"),
                ("z", "Z"),
                ("radius", "Radius"),
            ],
            defaults={"x": 0.0, "y": 0.0, "z": 0.0, "radius": 0.2},
            on_change=self.refresh_preview,
        )
        self.fuselage_editor.pack(fill="both", expand=True)

    def _build_preview(self) -> None:
        top_bar = ttk.Frame(self.right_panel)
        top_bar.pack(fill="x", pady=(0, 10))
        ttk.Label(top_bar, text="Быстрое 2D превью", font=("Helvetica", 15, "bold")).pack(side="left")

        self.preview_canvas = tk.Canvas(self.right_panel, bg="#f8fbff", highlightthickness=1, highlightbackground="#c4cfdb")
        self.preview_canvas.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Готово.")
        ttk.Label(self.right_panel, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", pady=(10, 0))

    def reset_project(self) -> None:
        self.current_file = None
        self.config = default_config()
        self.load_config_into_ui(self.config)
        self.refresh_preview()
        self.status_var.set("Создан новый проект.")

    def load_config_into_ui(self, config: AircraftConfig) -> None:
        self.airplane_name_var.set(config.airplane_name)
        self.airfoil_var.set(config.airfoil_name)
        self.fuselage_name_var.set(config.fuselage_name)
        self.fuselage_symmetry_var.set(config.fuselage_symmetry)
        self.draw_backend_var.set(config.draw_backend)
        self.thin_wings_var.set(config.thin_wings)
        self.fuselage_length_var.set(f"{self.get_fuselage_length(config.fuselage_sections):.4f}")
        self.fuselage_editor.set_rows([asdict(section) for section in config.fuselage_sections])
        self.wing_tab.set_surface(config.wing)
        self.canard_tab.set_surface(config.canard)
        self.htail_tab.set_surface(config.htail)
        self.vtail_tab.set_surface(config.vtail)

    def collect_config_from_ui(self) -> AircraftConfig:
        config = AircraftConfig(
            airplane_name=self.airplane_name_var.get().strip() or "My Concept",
            airfoil_name=self.airfoil_var.get().strip() or "naca2412",
            fuselage_name=self.fuselage_name_var.get().strip() or "Main Fuselage",
            fuselage_symmetry=self.fuselage_symmetry_var.get().strip() or "none",
            draw_backend=self.draw_backend_var.get().strip() or "pyvista",
            thin_wings=self.thin_wings_var.get(),
        )
        config.fuselage_sections = [FuselageSection(**row) for row in self.fuselage_editor.get_rows()]
        config.wing = self.wing_tab.get_surface()
        config.canard = self.canard_tab.get_surface()
        config.htail = self.htail_tab.get_surface()
        config.vtail = self.vtail_tab.get_surface()
        return config

    def get_fuselage_length(self, sections: list[FuselageSection]) -> float:
        if not sections:
            return 0.0
        xs = [section.x for section in sections]
        return max(xs) - min(xs)

    def scale_fuselage_length(self) -> None:
        rows = self.fuselage_editor.get_rows()
        if len(rows) < 2:
            messagebox.showinfo("Фюзеляж", "Нужно минимум 2 секции, чтобы изменить длину.", parent=self.root)
            return
        try:
            target_length = float(self.fuselage_length_var.get().strip().replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка", "Длина фюзеляжа должна быть числом.", parent=self.root)
            return
        current_min_x = min(row["x"] for row in rows)
        current_max_x = max(row["x"] for row in rows)
        current_length = current_max_x - current_min_x
        if math.isclose(current_length, 0.0):
            messagebox.showerror("Ошибка", "Текущая длина фюзеляжа равна нулю.", parent=self.root)
            return
        scale = target_length / current_length
        for row in rows:
            row["x"] = current_min_x + (row["x"] - current_min_x) * scale
        self.fuselage_editor.set_rows(rows)
        self.refresh_preview()
        self.status_var.set(f"Фюзеляж масштабирован до длины {target_length:.3f}.")

    def load_project(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Открыть конфигурацию",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            data = json.loads(Path(file_path).read_text(encoding="utf-8"))
            self.config = config_from_dict(data)
            self.current_file = Path(file_path)
            self.load_config_into_ui(self.config)
            self.refresh_preview()
            self.status_var.set(f"Загружено: {self.current_file.name}")
        except Exception as exc:
            messagebox.showerror("Ошибка загрузки", str(exc), parent=self.root)

    def save_project(self) -> None:
        self.config = self.collect_config_from_ui()
        file_path = self.current_file
        if file_path is None:
            path = filedialog.asksaveasfilename(
                title="Сохранить конфигурацию",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            file_path = Path(path)
        try:
            file_path.write_text(json.dumps(config_to_dict(self.config), indent=2, ensure_ascii=False), encoding="utf-8")
            self.current_file = file_path
            self.status_var.set(f"Сохранено: {file_path.name}")
        except Exception as exc:
            messagebox.showerror("Ошибка сохранения", str(exc), parent=self.root)

    def export_python(self) -> None:
        config = self.collect_config_from_ui()
        file_path = filedialog.asksaveasfilename(
            title="Экспорт Python-скрипта",
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            Path(file_path).write_text(self.generate_python_script(config), encoding="utf-8")
            self.status_var.set(f"Скрипт экспортирован: {Path(file_path).name}")
        except Exception as exc:
            messagebox.showerror("Ошибка экспорта", str(exc), parent=self.root)

    def build_airplane(self, config: AircraftConfig):
        if asb is None:
            raise RuntimeError("AeroSandbox не найден. Установи его в активное окружение.")
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
            wing = asb.Wing(
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
            wings.append(wing)

        return asb.Airplane(name=config.airplane_name, wings=wings, fuselages=[fuselage])

    def render_aircraft(self) -> None:
        try:
            config = self.collect_config_from_ui()
            airplane = self.build_airplane(config)
            airplane.draw(
                backend=config.draw_backend,
                thin_wings=config.thin_wings,
                use_preset_view_angle="iso",
                set_background_pane_color="white",
                show=True,
            )
            self.status_var.set("3D-визуализация открыта.")
        except Exception as exc:
            messagebox.showerror("Ошибка визуализации", str(exc), parent=self.root)

    def refresh_preview(self) -> None:
        try:
            config = self.collect_config_from_ui()
        except Exception:
            return
        self.config = config
        self.draw_preview(config)
        self.status_var.set(self.build_summary(config))

    def build_summary(self, config: AircraftConfig) -> str:
        enabled = [surface.name for surface in (config.canard, config.wing, config.htail, config.vtail) if surface.enabled]
        fuselage_length = self.get_fuselage_length(config.fuselage_sections)
        return (
            f"Самолёт: {config.airplane_name} | Профиль: {config.airfoil_name} | "
            f"Длина фюзеляжа: {fuselage_length:.3f} | Активные поверхности: {', '.join(enabled) if enabled else 'нет'}"
        )

    def _collect_bounds(self, config: AircraftConfig) -> tuple[float, float, float, float, float, float]:
        xs, ys, zs = [], [], []
        for section in config.fuselage_sections:
            xs.extend([section.x - section.radius, section.x + section.radius])
            ys.extend([section.y - section.radius, section.y + section.radius])
            zs.extend([section.z - section.radius, section.z + section.radius])
        for surface in (config.canard, config.wing, config.htail, config.vtail):
            if not surface.enabled:
                continue
            for section in surface.sections:
                xs.extend([section.x, section.x + section.chord])
                ys.extend([section.y, -section.y if surface.symmetric else section.y])
                zs.extend([section.z, section.z + section.chord * 0.15])
        if not xs:
            xs = [0.0, 1.0]
            ys = [-1.0, 1.0]
            zs = [-1.0, 1.0]
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

    def _map_point(
        self,
        value_x: float,
        value_y: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        width: float,
        height: float,
        padding: float,
    ) -> tuple[float, float]:
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        scale = min((width - 2 * padding) / span_x, (height - 2 * padding) / span_y)
        offset_x = (width - span_x * scale) / 2
        offset_y = (height - span_y * scale) / 2
        canvas_x = offset_x + (value_x - min_x) * scale
        canvas_y = height - (offset_y + (value_y - min_y) * scale)
        return canvas_x, canvas_y

    def draw_preview(self, config: AircraftConfig) -> None:
        canvas = self.preview_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 300)
        half_height = height / 2
        padding = 30

        min_x, max_x, min_y, max_y, min_z, max_z = self._collect_bounds(config)
        split_color = "#cad6e2"
        canvas.create_text(16, 12, text="Вид сверху", anchor="nw", fill="#35506b", font=("Helvetica", 12, "bold"))
        canvas.create_text(16, half_height + 12, text="Вид сбоку", anchor="nw", fill="#35506b", font=("Helvetica", 12, "bold"))
        canvas.create_line(0, half_height, width, half_height, fill=split_color, dash=(5, 4))

        self._draw_top_projection(canvas, config, width, half_height, padding, min_x, max_x, min_y, max_y)
        self._draw_side_projection(canvas, config, width, half_height, padding, min_x, max_x, min_z, max_z, half_height)

    def _draw_top_projection(self, canvas, config, width, height, padding, min_x, max_x, min_y, max_y) -> None:
        center_y = (min_y + max_y) / 2
        x0, y0 = self._map_point(min_x, center_y, min_x, max_x, min_y, max_y, width, height, padding)
        x1, y1 = self._map_point(max_x, center_y, min_x, max_x, min_y, max_y, width, height, padding)
        canvas.create_line(x0, y0, x1, y1, fill="#d0d7df", dash=(3, 4))

        fuselage_points_top = []
        for section in sorted(config.fuselage_sections, key=lambda item: item.x):
            top = self._map_point(section.x, section.y + section.radius, min_x, max_x, min_y, max_y, width, height, padding)
            bottom = self._map_point(
                section.x, section.y - section.radius, min_x, max_x, min_y, max_y, width, height, padding
            )
            fuselage_points_top.append((top, bottom))
        for idx in range(len(fuselage_points_top) - 1):
            a_top, a_bottom = fuselage_points_top[idx]
            b_top, b_bottom = fuselage_points_top[idx + 1]
            canvas.create_polygon(
                a_top[0], a_top[1], b_top[0], b_top[1], b_bottom[0], b_bottom[1], a_bottom[0], a_bottom[1],
                fill="#cfe0f5",
                outline="#40668d",
            )

        for surface in (config.canard, config.wing, config.htail, config.vtail):
            if not surface.enabled:
                continue
            self._draw_surface_top(canvas, surface, width, height, padding, min_x, max_x, min_y, max_y)

    def _draw_surface_top(self, canvas, surface, width, height, padding, min_x, max_x, min_y, max_y) -> None:
        sections = sorted(surface.sections, key=lambda item: item.y)
        if len(sections) < 2:
            return

        def draw_side(multiplier: float) -> None:
            for left, right in zip(sections[:-1], sections[1:]):
                p1 = self._map_point(left.x, multiplier * left.y, min_x, max_x, min_y, max_y, width, height, padding)
                p2 = self._map_point(left.x + left.chord, multiplier * left.y, min_x, max_x, min_y, max_y, width, height, padding)
                p3 = self._map_point(right.x + right.chord, multiplier * right.y, min_x, max_x, min_y, max_y, width, height, padding)
                p4 = self._map_point(right.x, multiplier * right.y, min_x, max_x, min_y, max_y, width, height, padding)
                canvas.create_polygon(
                    p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1],
                    fill=surface.color,
                    stipple="gray25",
                    outline="#1f2e3c",
                )

        draw_side(1.0)
        if surface.symmetric:
            draw_side(-1.0)

    def _draw_side_projection(
        self,
        canvas,
        config,
        width,
        height,
        padding,
        min_x,
        max_x,
        min_z,
        max_z,
        vertical_offset,
    ) -> None:
        def map_side(x_value: float, z_value: float) -> tuple[float, float]:
            x, y = self._map_point(x_value, z_value, min_x, max_x, min_z, max_z, width, height, padding)
            return x, y + vertical_offset

        center_z = (min_z + max_z) / 2
        x0, y0 = map_side(min_x, center_z)
        x1, y1 = map_side(max_x, center_z)
        canvas.create_line(x0, y0, x1, y1, fill="#d0d7df", dash=(3, 4))

        fuselage_top = []
        fuselage_bottom = []
        for section in sorted(config.fuselage_sections, key=lambda item: item.x):
            fuselage_top.append(map_side(section.x, section.z + section.radius))
            fuselage_bottom.append(map_side(section.x, section.z - section.radius))
        polygon = fuselage_top + list(reversed(fuselage_bottom))
        if len(polygon) >= 4:
            flat = [coord for point in polygon for coord in point]
            canvas.create_polygon(flat, fill="#cfe0f5", outline="#40668d")

        for surface in (config.canard, config.wing, config.htail, config.vtail):
            if not surface.enabled:
                continue
            for left, right in zip(surface.sections[:-1], surface.sections[1:]):
                p1 = map_side(left.x, left.z)
                p2 = map_side(left.x + left.chord, left.z)
                p3 = map_side(right.x + right.chord, right.z)
                p4 = map_side(right.x, right.z)
                canvas.create_polygon(
                    p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1],
                    fill=surface.color,
                    stipple="gray25",
                    outline="#1f2e3c",
                )

    def generate_python_script(self, config: AircraftConfig) -> str:
        data = json.dumps(config_to_dict(config), ensure_ascii=False, indent=4)
        return f"""import json
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


def main() -> None:
    root = tk.Tk()
    app = AircraftDesignerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
