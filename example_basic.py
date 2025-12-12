"""
Example: Basic Grid Cell Simulation
====================================

This script demonstrates a minimal example of running the grid cell simulator
with standard parameters.
"""

import numpy as np
from gc_periodic import gc_periodic

def main():
    """Run a basic grid cell simulation."""
    
    print("="*60)
    print("Grid Cell Simulation - Basic Example")
    print("="*60)
    
    # ====================================
    # SIMULATION PARAMETERS
    # ====================================
    
    # Network size
    n = 128          # 128x128 neural grid
    tau = 5          # Neural time constant (ms)
    dt = 0.5         # Integration time step (ms)
    
    # Module parameters (for Module 1 - smallest scale)
    module = 1
    scale_factor = 1.4 ** (module - 1)
    lambda_param = 13 * scale_factor
    
    beta = 3 / lambda_param**2      # Spatial scale
    alphabar = 1.05
    gamma = alphabar * beta          # Interaction range
    abar = 1.0                       # Excitatory amplitude
    wtphase = 2                      # Directional phase shift
    alpha = 1.0                      # Velocity gain
    
    # Model options
    useSpiking = False               # Use rate-based model
    
    print(f"\nNetwork Configuration:")
    print(f"  Grid size: {n}x{n} neurons")
    print(f"  Module: {module}")
    print(f"  Beta: {beta:.6f}")
    print(f"  Gamma: {gamma:.6f}")
    print(f"  Model: {'Spiking' if useSpiking else 'Rate-based'}")
    
    # ====================================
    # RUN SIMULATION
    # ====================================
    
    print("\nStarting simulation...")
    print("This will take a few minutes...\n")
    
    # Run simulation with random trajectory (no data file)
    spikes, integrated_x, integrated_y, error, pos_x, pos_y = gc_periodic(
        filename='nonexistent.npz',  # Will generate random trajectory
        n=n,
        tau=tau,
        dt=dt,
        beta=beta,
        gamma=gamma,
        abar=abar,
        wtphase=wtphase,
        alpha=alpha,
        useSpiking=useSpiking,
        module=module,
        GET_BAND=False,           # Use isotropic grid cells
        BAND_ANGLE=0,
        duration=100000           # 100,000 time steps
    )
    
    # ====================================
    # ANALYZE RESULTS
    # ====================================
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE!")
    print("="*60)
    
    # Calculate error statistics
    mean_error = np.mean(error)
    max_error = np.max(error)
    final_error = error[-1]
    
    print(f"\nPath Integration Performance:")
    print(f"  Mean error: {mean_error:.2f} cm")
    print(f"  Max error: {max_error:.2f} cm")
    print(f"  Final error: {final_error:.2f} cm")
    
    # Calculate total distance traveled
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    distances = np.sqrt(dx**2 + dy**2)
    total_distance = np.sum(distances)
    
    print(f"\nTrajectory Statistics:")
    print(f"  Total distance: {total_distance:.2f} cm")
    print(f"  Simulation duration: {len(pos_x) * dt:.1f} ms")
    print(f"  ({len(pos_x)} time steps)")
    
    print(f"\nOutput files saved to: plots/simulation/module_{module}/")
    print("="*60)
    
    return spikes, integrated_x, integrated_y, error, pos_x, pos_y


if __name__ == "__main__":
    results = main()
