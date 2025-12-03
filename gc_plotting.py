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
