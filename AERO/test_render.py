"""
Comprehensive Aerodynamic Analysis Script
==========================================
Performs complete aerodynamic analysis including:
- 3D aircraft visualization
- VLM calculations across alpha range
- CL, CD, Cm vs alpha
- Polar diagram (CL vs CD)
- L/D vs alpha
- Forces vs velocity
- Spanwise lift distribution
- CSV export of all data
- High-resolution PNG plots
"""

import json
import sys
import csv
from pathlib import Path
from typing import Dict

try:
    import aerosandbox as asb
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    project_root = Path("/Users/aleksandrvorobev/Documents/skat")
    for candidate in project_root.glob(".venv/lib/python*/site-packages"):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.append(candidate_str)
    import aerosandbox as asb
    import numpy as np
    import matplotlib.pyplot as plt

# Setup plotting style
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True

# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CONFIG = json.loads('''{
    "airplane_name": "My Concept",
    "airfoil_name": "naca2412",
    "fuselage_name": "Main Fuselage",
    "fuselage_symmetry": "XZ",
    "draw_backend": "pyvista",
    "thin_wings": false,
    "fuselage_sections": [
        {
            "x": -0.32,
            "y": 0,
            "z": -0.01,
            "radius": 0.034
        },
        {
            "x": -0.12,
            "y": 0,
            "z": 0,
            "radius": 0.052
        },
        {
            "x": 0.08,
            "y": 0,
            "z": 0,
            "radius": 0.052
        },
        {
            "x": 0.18,
            "y": 0,
            "z": 0,
            "radius": 0.045
        },
        {
            "x": 0.24,
            "y": 0,
            "z": 0,
            "radius": 0.012
        }
    ],
    "extra_fuselages": [
        [
            {
                "x": 0.16,
                "y": 0.16,
                "z": 0,
                "radius": 0.008
            },
            {
                "x": 0.665,
                "y": 0.16,
                "z": 0,
                "radius": 0.008
            }
        ],
        [
            {
                "x": 0.16,
                "y": -0.16,
                "z": 0,
                "radius": 0.008
            },
            {
                "x": 0.665,
                "y": -0.16,
                "z": 0,
                "radius": 0.008
            }
        ]
    ],
    "wing": {
        "name": "Main Wing",
        "enabled": true,
        "symmetric": true,
        "color": "#2468f2",
        "sections": [
            {
                "x": -0.08,
                "y": 0,
                "z": 0,
                "chord": 0.255,
                "twist": 2
            },
            {
                "x": 0,
                "y": 0.2925,
                "z": 0,
                "chord": 0.22,
                "twist": 0
            },
            {
                "x": 0.08,
                "y": 0.4875,
                "z": 0,
                "chord": 0.17,
                "twist": -2
            },
            {
                "x": 0.2,
                "y": 0.5525,
                "z": 0.035,
                "chord": 0.07,
                "twist": -4
            },
            {
                "x": 0.24,
                "y": 0.572,
                "z": 0.07,
                "chord": 0.055,
                "twist": -6
            }
        ]
    },
    "canard": {
        "name": "Canard",
        "enabled": false,
        "symmetric": true,
        "color": "#ff8c3b",
        "sections": []
    },
    "htail": {
        "name": "Left Tail",
        "enabled": true,
        "symmetric": false,
        "color": "#7e57ff",
        "sections": [
            {
                "x": 0.515,
                "y": 0.16,
                "z": 0,
                "chord": 0.15,
                "twist": 0
            },
            {
                "x": 0.545,
                "y": 0,
                "z": 0.15,
                "chord": 0.1,
                "twist": 0
            }
        ]
    },
    "vtail": {
        "name": "Right Tail",
        "enabled": true,
        "symmetric": false,
        "color": "#26a269",
        "sections": [
            {
                "x": 0.515,
                "y": -0.16,
                "z": 0,
                "chord": 0.15,
                "twist": 0
            },
            {
                "x": 0.545,
                "y": 0,
                "z": 0.15,
                "chord": 0.1,
                "twist": 0
            }
        ]
    }
}''')

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
            name=f"{CONFIG['fuselage_name']} Extra {idx}",
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

def run_alpha_sweep(airplane, alphas, velocity=10.0, density=1.225):
    """Run VLM analysis across angles of attack."""
    print(f"\n{'='*60}")
    print(f"ALPHA SWEEP ANALYSIS")
    print(f"{'='*60}")
    print(f"Alpha range: {alphas[0]}° to {alphas[-1]}° ({len(alphas)} points)")
    print(f"Velocity: {velocity} m/s, Density: {density} kg/m³\n")
    
    results = {'alpha': alphas, 'CL': [], 'CD': [], 'Cm': [], 'L': [], 'D': [], 'L_D': []}
    
    for i, alpha in enumerate(alphas):
        op_point = asb.OperatingPoint(velocity=velocity, alpha=alpha, density=density)
        vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=op_point)
        aero = vlm.run()
        
        CL, CD, Cm = aero['CL'], aero['CD'], aero['Cm']
        q = 0.5 * density * velocity**2
        S = sum(w.area() for w in airplane.wings)
        L, D = CL * q * S, CD * q * S
        
        results['CL'].append(float(CL))
        results['CD'].append(float(CD))
        results['Cm'].append(float(Cm))
        results['L'].append(float(L))
        results['D'].append(float(D))
        results['L_D'].append(float(L/D if abs(D) > 1e-9 else 0))
        
        if (i+1) % 5 == 0:
            print(f"  {i+1}/{len(alphas)} completed")
    
    for k in ['CL', 'CD', 'Cm', 'L', 'D', 'L_D']:
        results[k] = np.array(results[k])
    
    print("✓ Alpha sweep completed\n")
    return results


def run_velocity_sweep(airplane, velocities, alpha=5.0, density=1.225):
    """Run VLM analysis across velocities."""
    print(f"\n{'='*60}")
    print(f"VELOCITY SWEEP ANALYSIS")
    print(f"{'='*60}")
    print(f"Velocity range: {velocities[0]} to {velocities[-1]} m/s ({len(velocities)} points)")
    print(f"Alpha: {alpha}°, Density: {density} kg/m³\n")
    
    results = {'velocity': velocities, 'CL': [], 'CD': [], 'L': [], 'D': []}
    
    for i, vel in enumerate(velocities):
        op_point = asb.OperatingPoint(velocity=vel, alpha=alpha, density=density)
        vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=op_point)
        aero = vlm.run()
        
        CL, CD = aero['CL'], aero['CD']
        q = 0.5 * density * vel**2
        S = sum(w.area() for w in airplane.wings)
        L, D = CL * q * S, CD * q * S
        
        results['CL'].append(float(CL))
        results['CD'].append(float(CD))
        results['L'].append(float(L))
        results['D'].append(float(D))
        
        if (i+1) % 3 == 0:
            print(f"  {i+1}/{len(velocities)} completed")
    
    for k in ['CL', 'CD', 'L', 'D']:
        results[k] = np.array(results[k])
    
    print("✓ Velocity sweep completed\n")
    return results


def get_spanwise_distribution(airplane, alpha=5.0, velocity=10.0, density=1.225):
    """Calculate spanwise lift distribution."""
    print(f"Calculating spanwise distribution at α={alpha}°...")
    
    op_point = asb.OperatingPoint(velocity=velocity, alpha=alpha, density=density)
    vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=op_point)
    aero = vlm.run()
    
    # Find main wing
    main_wing = next((w for w in airplane.wings if "wing" in w.name.lower() and "tail" not in w.name.lower()), None)
    
    if not main_wing:
        return None
    
    # Create spanwise stations
    n_stations = 50
    span = main_wing.span()
    y = np.linspace(-span/2, span/2, n_stations)
    
    # Elliptical lift distribution approximation
    CL_total = aero['CL']
    q = 0.5 * density * velocity**2
    S = main_wing.area()
    L_total = CL_total * q * S
    
    lift_dist = (L_total * 4 / (np.pi * span)) * np.sqrt(np.maximum(0, 1 - (2*y/span)**2))
    circulation = lift_dist / (density * velocity)
    
    print("✓ Spanwise distribution calculated\n")
    return {'y': y, 'lift': lift_dist, 'circulation': circulation}


def save_csv(data, filename):
    """Save results to CSV."""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        keys = list(data.keys())
        writer.writerow(keys)
        for i in range(len(data[keys[0]])):
            writer.writerow([data[k][i] for k in keys])
    print(f"  Saved: {filepath}")


def plot_cl_cd_cm_alpha(results):
    """Plot CL, CD, Cm vs alpha."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # CL vs alpha
    axes[0].plot(results['alpha'], results['CL'], 'b-o', linewidth=2, markersize=4)
    axes[0].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0].axvline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0].set_ylabel('Lift Coefficient, CL')
    axes[0].set_title('Lift Coefficient vs Angle of Attack')
    axes[0].grid(True, alpha=0.3)
    
    # CD vs alpha
    axes[1].plot(results['alpha'], results['CD'], 'r-s', linewidth=2, markersize=4)
    axes[1].axvline(0, color='k', linestyle='--', linewidth=0.5)
    axes[1].set_ylabel('Drag Coefficient, CD')
    axes[1].set_title('Drag Coefficient vs Angle of Attack')
    axes[1].grid(True, alpha=0.3)
    min_idx = np.argmin(results['CD'])
    axes[1].plot(results['alpha'][min_idx], results['CD'][min_idx], 'go', markersize=10, 
                 label=f"Min CD={results['CD'][min_idx]:.5f}")
    axes[1].legend()
    
    # Cm vs alpha
    axes[2].plot(results['alpha'], results['Cm'], 'g-^', linewidth=2, markersize=4)
    axes[2].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[2].axvline(0, color='k', linestyle='--', linewidth=0.5)
    axes[2].set_xlabel('Angle of Attack, α (degrees)')
    axes[2].set_ylabel('Pitching Moment Coeff, Cm')
    axes[2].set_title('Pitching Moment Coefficient vs Angle of Attack')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = RESULTS_DIR / 'cl_cd_cm_vs_alpha.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_polar(results):
    """Plot drag polar."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(results['CD'], results['CL'], c=results['alpha'], 
                        cmap='viridis', s=50, edgecolors='black', linewidth=0.5)
    ax.plot(results['CD'], results['CL'], 'k-', linewidth=1, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Angle of Attack, α (degrees)')
    
    # Mark best L/D
    max_idx = np.argmax(results['L_D'])
    ax.plot(results['CD'][max_idx], results['CL'][max_idx], 'r*', markersize=15,
            label=f"Best L/D={results['L_D'][max_idx]:.2f} at α={results['alpha'][max_idx]:.1f}°")
    
    ax.set_xlabel('Drag Coefficient, CD')
    ax.set_ylabel('Lift Coefficient, CL')
    ax.set_title('Drag Polar (CL vs CD)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    filepath = RESULTS_DIR / 'polar.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_ld_alpha(results):
    """Plot L/D vs alpha."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(results['alpha'], results['L_D'], 'm-d', linewidth=2, markersize=4)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
    
    max_idx = np.argmax(results['L_D'])
    ax.plot(results['alpha'][max_idx], results['L_D'][max_idx], 'ro', markersize=10,
            label=f"Max L/D={results['L_D'][max_idx]:.2f} at α={results['alpha'][max_idx]:.1f}°")
    
    ax.set_xlabel('Angle of Attack, α (degrees)')
    ax.set_ylabel('Lift-to-Drag Ratio, L/D')
    ax.set_title('Lift-to-Drag Ratio vs Angle of Attack')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    filepath = RESULTS_DIR / 'ld_vs_alpha.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_forces_velocity(results):
    """Plot forces vs velocity."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Lift vs velocity
    axes[0].plot(results['velocity'], results['L'], 'b-o', linewidth=2, markersize=4)
    axes[0].set_ylabel('Lift Force, L (N)')
    axes[0].set_title('Lift Force vs Velocity')
    axes[0].grid(True, alpha=0.3)
    
    # Drag vs velocity
    axes[1].plot(results['velocity'], results['D'], 'r-s', linewidth=2, markersize=4)
    axes[1].set_xlabel('Velocity (m/s)')
    axes[1].set_ylabel('Drag Force, D (N)')
    axes[1].set_title('Drag Force vs Velocity')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = RESULTS_DIR / 'forces_vs_velocity.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_coeffs_velocity(results):
    """Plot CL, CD vs velocity."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # CL vs velocity
    axes[0].plot(results['velocity'], results['CL'], 'b-o', linewidth=2, markersize=4)
    axes[0].set_ylabel('Lift Coefficient, CL')
    axes[0].set_title('Lift Coefficient vs Velocity')
    axes[0].grid(True, alpha=0.3)
    
    # CD vs velocity
    axes[1].plot(results['velocity'], results['CD'], 'r-s', linewidth=2, markersize=4)
    axes[1].set_xlabel('Velocity (m/s)')
    axes[1].set_ylabel('Drag Coefficient, CD')
    axes[1].set_title('Drag Coefficient vs Velocity')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = RESULTS_DIR / 'coeffs_vs_velocity.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_spanwise(data):
    """Plot spanwise distributions."""
    if data is None:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Lift distribution
    axes[0].plot(data['y'], data['lift'], 'b-', linewidth=2)
    axes[0].fill_between(data['y'], 0, data['lift'], alpha=0.3)
    axes[0].set_ylabel('Lift per unit span (N/m)')
    axes[0].set_title('Spanwise Lift Distribution')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0, color='k', linestyle='--', linewidth=0.5)
    
    # Circulation distribution
    axes[1].plot(data['y'], data['circulation'], 'r-', linewidth=2)
    axes[1].fill_between(data['y'], 0, data['circulation'], alpha=0.3, color='red')
    axes[1].set_xlabel('Spanwise Position, y (m)')
    axes[1].set_ylabel('Circulation, Γ (m²/s)')
    axes[1].set_title('Spanwise Circulation Distribution')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    filepath = RESULTS_DIR / 'spanwise_distribution.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def print_geometry(airplane):
    """Print geometric parameters."""
    print(f"\n{'='*60}")
    print("GEOMETRIC PARAMETERS")
    print(f"{'='*60}")
    
    main_wing = next((w for w in airplane.wings if "wing" in w.name.lower() and "tail" not in w.name.lower()), None)
    
    if main_wing:
        print(f"Main Wing:")
        print(f"  Wingspan (b):           {main_wing.span():.4f} m")
        print(f"  Wing Area (S):          {main_wing.area():.4f} m²")
        print(f"  Mean Aerodynamic Chord: {main_wing.mean_aerodynamic_chord():.4f} m")
        print(f"  Aspect Ratio (AR):      {main_wing.aspect_ratio():.4f}")
    
    total_area = sum(w.area() for w in airplane.wings)
    print(f"\nTotal Lifting Surface Area: {total_area:.4f} m²")
    print(f"{'='*60}\n")


def print_performance(alpha_results):
    """Print key performance metrics."""
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    max_ld_idx = np.argmax(alpha_results['L_D'])
    print(f"Maximum L/D:        {alpha_results['L_D'][max_ld_idx]:.2f}")
    print(f"  at alpha:         {alpha_results['alpha'][max_ld_idx]:.1f}°")
    print(f"  CL:               {alpha_results['CL'][max_ld_idx]:.4f}")
    print(f"  CD:               {alpha_results['CD'][max_ld_idx]:.5f}")
    
    min_cd_idx = np.argmin(alpha_results['CD'])
    print(f"\nMinimum CD:         {alpha_results['CD'][min_cd_idx]:.5f}")
    print(f"  at alpha:         {alpha_results['alpha'][min_cd_idx]:.1f}°")
    
    zero_idx = np.argmin(np.abs(alpha_results['alpha']))
    print(f"\nAt α=0°:")
    print(f"  CL:               {alpha_results['CL'][zero_idx]:.4f}")
    print(f"  CD:               {alpha_results['CD'][zero_idx]:.5f}")
    print(f"  Cm:               {alpha_results['Cm'][zero_idx]:.4f}")
    print(f"  L/D:              {alpha_results['L_D'][zero_idx]:.2f}")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("COMPREHENSIVE AERODYNAMIC ANALYSIS")
print("="*60 + "\n")

# Build airplane
airfoil = asb.Airfoil(CONFIG["airfoil_name"])
primary_fuselage = asb.Fuselage(
    name=CONFIG["fuselage_name"],
    xsecs=[asb.FuselageXSec(xyz_c=[s["x"], s["y"], s["z"]], radius=s["radius"])
           for s in CONFIG["fuselage_sections"]],
    symmetry=None if CONFIG["fuselage_symmetry"] == "none" else CONFIG["fuselage_symmetry"],
)

fuselages = [primary_fuselage]
for idx, chain in enumerate(CONFIG.get("extra_fuselages", []), start=1):
    if len(chain) >= 2:
        fuselages.append(asb.Fuselage(
            name=f"{CONFIG['fuselage_name']} Extra {idx}",
            xsecs=[asb.FuselageXSec(xyz_c=[s["x"], s["y"], s["z"]], radius=s["radius"]) for s in chain],
            symmetry=None,
        ))

wings = []
for surface_key in ["canard", "wing", "htail", "vtail"]:
    surface = CONFIG[surface_key]
    if surface["enabled"]:
        wings.append(asb.Wing(
            name=surface["name"],
            symmetric=surface["symmetric"],
            xsecs=[asb.WingXSec(xyz_le=[s["x"], s["y"], s["z"]], chord=s["chord"], 
                               twist=s["twist"], airfoil=airfoil) for s in surface["sections"]],
        ))

airplane = asb.Airplane(name=CONFIG["airplane_name"], wings=wings, fuselages=fuselages)

print_geometry(airplane)

# Run analyses
alphas = np.linspace(-5, 15, 21)
alpha_results = run_alpha_sweep(airplane, alphas, velocity=10.0)

velocities = np.linspace(5, 25, 11)
vel_results = run_velocity_sweep(airplane, velocities, alpha=5.0)

spanwise_data = get_spanwise_distribution(airplane, alpha=5.0, velocity=10.0)

# Generate all plots
print(f"Generating plots...")
plot_cl_cd_cm_alpha(alpha_results)
plot_polar(alpha_results)
plot_ld_alpha(alpha_results)
plot_forces_velocity(vel_results)
plot_coeffs_velocity(vel_results)
plot_spanwise(spanwise_data)

# Save data to CSV
print(f"\nSaving data to CSV...")
save_csv(alpha_results, 'alpha_sweep.csv')
save_csv(vel_results, 'velocity_sweep.csv')
if spanwise_data:
    save_csv(spanwise_data, 'spanwise_distribution.csv')

# Print performance summary
print_performance(alpha_results)

# Show 3D visualization
print("Displaying 3D visualization...")
airplane.draw(
    backend=CONFIG["draw_backend"],
    thin_wings=CONFIG["thin_wings"],
    use_preset_view_angle="iso",
    set_background_pane_color="white",
    show=True,
)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print(f"All results saved to: {RESULTS_DIR.absolute()}")
print("="*60 + "\n")
