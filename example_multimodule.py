"""
Example: Multi-Module Comparison
=================================

This script demonstrates running simulations across multiple grid cell modules
with different spatial scales and comparing their performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from gc_periodic import gc_periodic

def set_module_params(module_number):
    """
    Set parameters for a specific module.
    
    Parameters:
    -----------
    module_number : int
        Module ID (1-4)
    
    Returns:
    --------
    dict : Module parameters
    """
    scale_factor = 1.4 ** (module_number - 1)
    lambda_param = 13 * scale_factor
    
    return {
        'beta': 3 / lambda_param**2,
        'gamma': 1.05 * (3 / lambda_param**2),
        'abar': 1.0,
        'wtphase': 2,
        'alpha': 1 / scale_factor,
        'scale_factor': scale_factor,
        'lambda': lambda_param
    }

def main():
    """Run simulations for multiple modules and compare results."""
    
    print("="*70)
    print("Multi-Module Grid Cell Comparison")
    print("="*70)
    
    # Common parameters
    n = 128
    tau = 5
    dt = 0.5
    useSpiking = False
    duration = 50000  # Shorter for faster comparison
    
    # Modules to simulate
    modules = [1, 2, 3, 4]
    results = {}
    
    # ====================================
    # RUN ALL MODULES
    # ====================================
    
    for module_id in modules:
        print(f"\n{'='*70}")
        print(f"Running Module {module_id}...")
        print(f"{'='*70}")
        
        # Get module-specific parameters
        params = set_module_params(module_id)
        
        print(f"  Scale factor: {params['scale_factor']:.2f}")
        print(f"  Grid spacing: {params['lambda']:.1f} cm")
        print(f"  Beta: {params['beta']:.6f}")
        
        # Run simulation
        spikes, int_x, int_y, error, pos_x, pos_y = gc_periodic(
            filename='nonexistent.npz',
            n=n,
            tau=tau,
            dt=dt,
            beta=params['beta'],
            gamma=params['gamma'],
            abar=params['abar'],
            wtphase=params['wtphase'],
            alpha=params['alpha'],
            useSpiking=useSpiking,
            module=module_id,
            GET_BAND=False,
            BAND_ANGLE=0,
            duration=duration
        )
        
        # Store results
        results[module_id] = {
            'error': error,
            'integrated_x': int_x,
            'integrated_y': int_y,
            'position_x': pos_x,
            'position_y': pos_y,
            'params': params
        }
        
        print(f"  Mean error: {np.mean(error):.2f} cm")
        print(f"  Final error: {error[-1]:.2f} cm")
    
    # ====================================
    # COMPARATIVE ANALYSIS
    # ====================================
    
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*70}\n")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Error over time for all modules
    ax1 = axes[0, 0]
    for module_id in modules:
        error = results[module_id]['error']
        time_ms = np.arange(len(error)) * dt
        ax1.plot(time_ms, error, label=f'Module {module_id}', linewidth=1.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Path Integration Error (cm)')
    ax1.set_title('Error Accumulation Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final error vs. module
    ax2 = axes[0, 1]
    final_errors = [results[m]['error'][-1] for m in modules]
    mean_errors = [np.mean(results[m]['error']) for m in modules]
    ax2.bar(modules, final_errors, alpha=0.7, label='Final Error')
    ax2.plot(modules, mean_errors, 'ro-', linewidth=2, markersize=8, label='Mean Error')
    ax2.set_xlabel('Module Number')
    ax2.set_ylabel('Error (cm)')
    ax2.set_title('Error by Module')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Trajectory comparison (Module 1 vs 4)
    ax3 = axes[1, 0]
    for module_id in [1, 4]:
        pos_x = results[module_id]['position_x']
        pos_y = results[module_id]['position_y']
        ax3.plot(pos_x[:1000], pos_y[:1000], alpha=0.6, 
                label=f'Module {module_id}', linewidth=1)
    ax3.set_xlabel('X Position (cm)')
    ax3.set_ylabel('Y Position (cm)')
    ax3.set_title('Sample Trajectories (First 1000 steps)')
    ax3.legend()
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Scale factor vs. error
    ax4 = axes[1, 1]
    scale_factors = [results[m]['params']['scale_factor'] for m in modules]
    grid_spacings = [results[m]['params']['lambda'] for m in modules]
    ax4_twin = ax4.twiny()
    ax4.plot(scale_factors, mean_errors, 'bo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Scale Factor')
    ax4.set_ylabel('Mean Error (cm)')
    ax4.set_title('Error vs. Grid Scale')
    ax4.grid(True, alpha=0.3)
    ax4_twin.set_xlabel('Grid Spacing (cm)', color='red')
    ax4_twin.plot(grid_spacings, mean_errors, alpha=0)
    ax4_twin.tick_params(axis='x', labelcolor='red')
    
    plt.tight_layout()
    plt.savefig('plots/multi_module_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison plot saved to: plots/multi_module_comparison.png")
    plt.show()
    
    # Print summary table
    print("\nSummary Table:")
    print("-" * 70)
    print(f"{'Module':<8} {'Scale':<10} {'Spacing (cm)':<15} {'Mean Error':<15} {'Final Error':<15}")
    print("-" * 70)
    for module_id in modules:
        params = results[module_id]['params']
        error = results[module_id]['error']
        print(f"{module_id:<8} {params['scale_factor']:<10.2f} "
              f"{params['lambda']:<15.1f} {np.mean(error):<15.2f} "
              f"{error[-1]:<15.2f}")
    print("-" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
