"""
Grid Cell Plotting Utilities

Separates visualization logic from simulation code.
Saves sequential frames for GIF creation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_simulation_frame(
    r_new,
    sNeuron,
    sNeuronResponse,
    position_x,
    position_y,
    increment,
    iteration,
    sampling_length,
    module,
    frame_idx
):
    """
    Plot and save a single simulation frame showing neural activity and trajectory.
    
    Parameters:
    -----------
    r_new : ndarray
        Neural population activity matrix (n x n)
    sNeuron : list
        [row, col] coordinates of the tracked neuron
    sNeuronResponse : ndarray
        Array tracking when the neuron fired
    position_x : ndarray
        X-coordinates of the trajectory
    position_y : ndarray
        Y-coordinates of the trajectory
    increment : int
        Current time step in the trajectory
    iteration : int
        Current iteration number
    sampling_length : int
        Total length of the simulation
    module : int
        Module number for organizing output
    frame_idx : int
        Sequential frame number for GIF creation
        
    Returns:
    --------
    None (saves figure to disk)
    """
    
    # Prepare output directory
    output_dir = os.path.join('plots', 'simulation', f'module_{module}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left subplot: Neural population activity
    im1 = ax1.imshow(r_new, cmap='hot', vmin=0, vmax=2)
    ax1.set_title(f'Neural Population Activity\n(Step {iteration}/{sampling_length-20})')
    ax1.set_aspect('equal', adjustable='box')
    ax1.plot(sNeuron[1], sNeuron[0], 'bo', markersize=6, label='Tracked Neuron')
    ax1.legend(loc='upper right', fontsize=8)
    plt.colorbar(im1, ax=ax1)
    
    # Right subplot: Trajectory and neuron response
    tempx = sNeuronResponse[:increment] * position_x[:increment]
    tempy = sNeuronResponse[:increment] * position_y[:increment]
    tempx = tempx[tempx != 0]
    tempy = tempy[tempy != 0]
    
    ax2.plot(position_x[:increment], position_y[:increment], '-', 
            color='gray', alpha=0.5, linewidth=0.5, label='Trajectory')
    ax2.plot(position_x[increment-1], position_y[increment-1], 'ro', 
            markersize=8, label='Current position')
    if len(tempx) > 0:
        ax2.plot(tempx, tempy, 'bx', markersize=4, label='Neuron spikes')
    ax2.set_title(f'Neuron {sNeuron} Response')
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim([position_x.min(), position_x.max()])
    ax2.set_ylim([position_y.min(), position_y.max()])
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save sequential frame for GIF creation
    save_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved frame {frame_idx:06d} to {save_path}")


def plot_path_integration_debug(
    position_x,
    position_y,
    integrated_path_x_cm,
    integrated_path_y_cm,
    valid_len,
    dt,
    scale_factor,
    module,
    output_dir
):
    """
    Create and save a debug plot comparing actual trajectory with network path integration.
    
    Parameters:
    -----------
    position_x : ndarray
        Actual X coordinates
    position_y : ndarray
        Actual Y coordinates
    integrated_path_x_cm : ndarray
        Network-integrated X path (in cm)
    integrated_path_y_cm : ndarray
        Network-integrated Y path (in cm)
    valid_len : int
        Number of valid data points
    dt : float
        Time step (ms)
    scale_factor : float
        Optimized scale factor
    module : int
        Module number
    output_dir : str
        Directory to save plot
        
    Returns:
    --------
    None (saves figure to disk)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 2D Trajectory Comparison (Top View)
    axes[0].plot(position_x[:valid_len], position_y[:valid_len], 'k-', label='Actual Rat Path', alpha=0.6)
    axes[0].plot(integrated_path_x_cm[:valid_len], integrated_path_y_cm[:valid_len], 'b--', label='Network Integration', alpha=0.8)
    axes[0].set_title(f"2D Trajectory (Scale Factor: {scale_factor:.2f})")
    axes[0].set_xlabel("X (cm)")
    axes[0].set_ylabel("Y (cm)")
    axes[0].legend()
    axes[0].grid(True)

    # 2. X Position over Time
    time_axis = np.arange(valid_len) * dt * 0.001  # Convert steps to seconds
    axes[1].plot(time_axis, position_x[:valid_len], 'k-', label='Actual X')
    axes[1].plot(time_axis, integrated_path_x_cm[:valid_len], 'b--', label='Integrated X')
    axes[1].set_title("X Position vs Time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("X (cm)")
    axes[1].legend()

    # 3. Y Position over Time
    axes[2].plot(time_axis, position_y[:valid_len], 'k-', label='Actual Y')
    axes[2].plot(time_axis, integrated_path_y_cm[:valid_len], 'b--', label='Integrated Y')
    axes[2].set_title("Y Position vs Time")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Y (cm)")
    axes[2].legend()

    plt.tight_layout()
    
    # Save the debug plot
    debug_plot_path = os.path.join(output_dir, f'debug_path_integration_mod{module}.png')
    plt.savefig(debug_plot_path)
    print(f"Debug plot saved to: {debug_plot_path}")
    plt.close(fig)
