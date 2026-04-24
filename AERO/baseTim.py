import aerosandbox as asb
import numpy as np

# 1. Сечения фюзеляжа (используем x, y, z вместо xyz_center)
fuselage_xsecs = [
    asb.FuselageXSec([0.0, 0, 0], radius = 0.1),
    asb.FuselageXSec([0.5, 0, 0], radius =  0.25),
    asb.FuselageXSec([2.0, 0, 0], radius =  0.25),
    asb.FuselageXSec([4.0, 0, 0], radius =  0.25),
    asb.FuselageXSec([5.5, 0, 0], radius =  0.01),
]

fuselage = asb.Fuselage(
    name="Main Fuselage",
    xsecs=fuselage_xsecs,
    symmetry="XZ"  # Зеркалим по вертикали (верх/низ)
)

# 2. Крыло (твой код без изменений)
airfoil = asb.Airfoil("naca2412")  # Более "самолётный" профиль

wing = asb.Wing(
    name="Swept Wing",
    symmetric=True,
    xsecs=[
        # Корень: без стреловидности
        asb.WingXSec(xyz_le=[4+0, 0, 0], chord=1.5, twist=2.0, airfoil=airfoil),
        # Середина: стреловидность назад + небольшое опускание
        asb.WingXSec(xyz_le=[4+0.3, 2, 0.1], chord=1.0, twist=-1.5, airfoil=airfoil),
        # Законцовка: сильнее стреловидность + отрицательный поперечный V
        asb.WingXSec(xyz_le=[4+1, 2.5, 0.4], chord=0.5, twist=-3.0, airfoil=airfoil),
    ]
)

# --- 🆕 ДОБАВЛЕНО: Цельноповоротное ПГО (Канарды) ---
canard = asb.Wing(
    name="All-Moving Canard",
    symmetric=True,
    xsecs=[
        # Корень ПГО: впереди крыла (x=0.8), у борта фюзеляжа (y=0.35)
        asb.WingXSec(xyz_le=[0.9, 0.1, -0.05], chord=0.7, twist=0.0, airfoil=airfoil),
        # Середина ПГО: умеренная стреловидность
        asb.WingXSec(xyz_le=[1.2, 0.9, 0.0], chord=0.5, twist=0.0, airfoil=airfoil),
        # Концовка ПГО: компактная
        asb.WingXSec(xyz_le=[1.5, 1.4, 0.07], chord=0.3, twist=1.0, airfoil=airfoil),
    ]
)
# ----------------------------------------------------

# 3. Сборка (добавляем ПГО в список wings)
airplane = asb.Airplane(
    name="My Concept",
    wings=[canard, wing],  # 🔹 ПГО и крыло в одном списке
    fuselages=[fuselage]
)

# 4. Визуализация
airplane.draw(
    backend='pyvista', 
    thin_wings=False,          
    use_preset_view_angle='iso',
    set_background_pane_color='white',
    show=True
)