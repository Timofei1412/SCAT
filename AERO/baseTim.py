import aerosandbox as asb
import aerosandbox.numpy as np

# =========================================================
# AIRFOILS
# =========================================================

wing_airfoil = asb.Airfoil("naca2412")
tail_airfoil = asb.Airfoil("naca0015")

# =========================================================
# BASIC PARAMETERS
# =========================================================

span_total = 1.30
half_span = span_total / 2
tip_dihedral_z = 0.035
# хорды (логичное сужение)
# хорды (уменьшенные законцовки в 2 раза)
root_chord = 0.255
mid_chord = 0.220
pre_tip_chord = 0.170

tip_mid_chord = 0.070   # было 0.140
tip_chord = 0.055       # было 0.110

# крутка (без изменений)
root_twist = 2
mid_twist = 0
pre_tip_twist = -2
tip_mid_twist = -4
tip_twist = -6

# законцовка по размаху (чуть уменьшили размер)
y_root = 0.0
y_mid = half_span * 0.45
y_pre_tip = half_span * 0.75
y_tip_mid = half_span * 0.85   # было 0.90
y_tip = half_span * 0.88        # было 1.00

# X не трогаем
x_root = -0.08
x_mid = 0.00
x_pre_tip = 0.08
x_tip_mid = 0.20
x_tip = 0.24

# =========================================================
# MAIN WING
# =========================================================

wing = asb.Wing(
    name="Main Wing",
    symmetric=True,

    xsecs=[

        # ROOT
        asb.WingXSec(
            xyz_le=[x_root, y_root, 0.0],
            chord=root_chord,
            twist=root_twist,
            airfoil=wing_airfoil
        ),

        # MID (центр полу-крыла)
        asb.WingXSec(
            xyz_le=[x_mid, y_mid, 0.0],
            chord=mid_chord,
            twist=mid_twist,
            airfoil=wing_airfoil
        ),

        # BEFORE TIP
        asb.WingXSec(
            xyz_le=[x_pre_tip, y_pre_tip, 0.0],
            chord=pre_tip_chord,
            twist=pre_tip_twist,
            airfoil=wing_airfoil
        ),

        # TIP MID (середина законцовки)
        asb.WingXSec(
            xyz_le=[x_tip_mid, y_tip_mid, tip_dihedral_z],
            chord=tip_mid_chord,
            twist=tip_mid_twist,
            airfoil=wing_airfoil
        ),

        # TIP EDGE (край законцовки)
        asb.WingXSec(
            xyz_le=[x_tip, y_tip, 2*tip_dihedral_z],
            chord=tip_chord,
            twist=tip_twist,
            airfoil=wing_airfoil
        ),
    ]
)

# =========================================================
# CENTER FUSELAGE
# =========================================================

fuselage = asb.Fuselage(
    name="Center Fuselage",
    xsecs=[

        asb.FuselageXSec(xyz_c=[-0.32, 0, -0.01], radius=0.034),
        asb.FuselageXSec(xyz_c=[-0.12, 0, 0], radius=0.0520),
        asb.FuselageXSec(xyz_c=[0.08, 0, 0], radius=0.052),
        asb.FuselageXSec(xyz_c=[0.18, 0, 0], radius=0.045),
        asb.FuselageXSec(xyz_c=[0.24, 0, 0], radius=0.012),
    ]
)

# =========================================================
# BOOMS
# =========================================================

boom_spacing = 0.320
boom_half_y = boom_spacing / 2

boom_start_x = 0.16
boom_length = 0.505
boom_z = 0.0

left_boom = asb.Fuselage(
    name="Left Boom",
    xsecs=[
        asb.FuselageXSec(xyz_c=[boom_start_x, boom_half_y, boom_z], radius=0.008),
        asb.FuselageXSec(xyz_c=[boom_start_x + boom_length, boom_half_y, boom_z], radius=0.008),
    ]
)

right_boom = asb.Fuselage(
    name="Right Boom",
    xsecs=[
        asb.FuselageXSec(xyz_c=[boom_start_x, -boom_half_y, boom_z], radius=0.008),
        asb.FuselageXSec(xyz_c=[boom_start_x + boom_length, -boom_half_y, boom_z], radius=0.008),
    ]
)

# =========================================================
# TAIL (оставил как у тебя)
# =========================================================

tail_airfoil = asb.Airfoil("naca0015")

tail_x = boom_start_x + boom_length - 0.15
tail_tip_z = 0.15
boom_half_y = boom_spacing / 2

left_tail = asb.Wing(
    name="Left Tail",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[tail_x, boom_half_y, boom_z],
            chord=0.15,
            twist=0,
            airfoil=tail_airfoil
        ),
        asb.WingXSec(
            xyz_le=[tail_x + 0.03, 0.0, tail_tip_z],
            chord=0.10,
            twist=0,
            airfoil=tail_airfoil
        ),
    ]
)

right_tail = asb.Wing(
    name="Right Tail",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[tail_x, -boom_half_y, boom_z],
            chord=0.15,
            twist=0,
            airfoil=tail_airfoil
        ),
        asb.WingXSec(
            xyz_le=[tail_x + 0.03, 0.0, tail_tip_z],
            chord=0.10,
            twist=0,
            airfoil=tail_airfoil
        ),
    ]
)

# =========================================================
# AIRPLANE
# =========================================================

plane = asb.Airplane(
    name="Refined Wing UAV",

    wings=[wing, left_tail, right_tail],

    fuselages=[fuselage, left_boom, right_boom]
)

# =========================================================
# FLIGHT
# =========================================================

op_point = asb.OperatingPoint(
    velocity=25,
    alpha=3
)

analysis = asb.VortexLatticeMethod(
    airplane=plane,
    op_point=op_point
)

results = analysis.run()

print("\nCL:", results["CL"])
print("CD:", results["CD"])
print("L/D:", results["CL"]/results["CD"])


if __name__ == "__main__":
    # Только при прямом запуске скрипта выполняем анализ и GUI-рендер.
    print("\nRunning baseTim as script: performing analysis and render")
    print("CL:", results["CL"])
    print("CD:", results["CD"])
    print("L/D:", results["CL"]/results["CD"])

    # Открываем окно рендера только из основного процесса/thread
    plane.draw(backend="pyvista", thin_wings=False, show=True)