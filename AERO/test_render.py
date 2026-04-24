import json
import sys
from pathlib import Path

try:
    import aerosandbox as asb
except ImportError:
    for candidate in Path(__file__).resolve().parent.glob(".venv/lib/python*/site-packages"):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.append(candidate_str)
    import aerosandbox as asb

CONFIG = json.loads('''{
    "airplane_name": "My Concept",
    "airfoil_name": "naca2412",
    "fuselage_name": "Main Fuselage",
    "fuselage_symmetry": "XZ",
    "draw_backend": "pyvista",
    "thin_wings": false,
    "fuselage_sections": [
        {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "radius": 0.1
        },
        {
            "x": 0.5,
            "y": 0.0,
            "z": 0.0,
            "radius": 0.25
        },
        {
            "x": 2.0,
            "y": 0.0,
            "z": 0.0,
            "radius": 0.25
        },
        {
            "x": 4.0,
            "y": 0.0,
            "z": 0.0,
            "radius": 0.25
        },
        {
            "x": 5.5,
            "y": 0.0,
            "z": 0.0,
            "radius": 0.01
        }
    ],
    "wing": {
        "name": "Swept Wing",
        "enabled": true,
        "symmetric": true,
        "color": "#2468f2",
        "sections": [
            {
                "x": 4.0,
                "y": 0.0,
                "z": 0.0,
                "chord": 1.5,
                "twist": 2.0
            },
            {
                "x": 4.3,
                "y": 2.0,
                "z": 0.1,
                "chord": 1.0,
                "twist": -1.5
            },
            {
                "x": 5.0,
                "y": 2.5,
                "z": 0.4,
                "chord": 0.5,
                "twist": -3.0
            }
        ]
    },
    "canard": {
        "name": "All-Moving Canard",
        "enabled": true,
        "symmetric": true,
        "color": "#ff8c3b",
        "sections": [
            {
                "x": 0.9,
                "y": 0.1,
                "z": -0.05,
                "chord": 0.7,
                "twist": 0.0
            },
            {
                "x": 1.2,
                "y": 0.9,
                "z": 0.0,
                "chord": 0.5,
                "twist": 0.0
            },
            {
                "x": 1.5,
                "y": 1.4,
                "z": 0.07,
                "chord": 0.3,
                "twist": 1.0
            }
        ]
    },
    "htail": {
        "name": "Horizontal Tail",
        "enabled": false,
        "symmetric": true,
        "color": "#7e57ff",
        "sections": [
            {
                "x": 4.6,
                "y": 0.0,
                "z": 0.15,
                "chord": 0.8,
                "twist": 0.0
            },
            {
                "x": 4.9,
                "y": 1.0,
                "z": 0.2,
                "chord": 0.45,
                "twist": -1.0
            }
        ]
    },
    "vtail": {
        "name": "Vertical Tail",
        "enabled": false,
        "symmetric": false,
        "color": "#26a269",
        "sections": [
            {
                "x": 4.7,
                "y": 0.0,
                "z": 0.1,
                "chord": 0.95,
                "twist": 0.0
            },
            {
                "x": 5.0,
                "y": 0.0,
                "z": 0.95,
                "chord": 0.35,
                "twist": 0.0
            }
        ]
    }
}''')

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
