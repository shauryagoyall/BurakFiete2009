"""
Grid Cell Simulation Engine

Main simulation loops for network formation and trajectory-based dynamics.
"""

import numpy as np
import os
from gc_network import compute_directional_input, convolve_with_weights
from gc_plotting import plot_simulation_frame


def run_network_formation(n, tau, dt, venvelope, weights, type_masks, n_iterations=1000):
    """
    Run initial network formation with aperiodic boundaries and zero velocity.
    
    This phase allows the network to self-organize into a stable grid pattern
    before trajectory-based simulation begins.
    
    Parameters:
    -----------
    n : int
        Grid size
    tau : float
        Time constant (ms)
    dt : float
        Time step (ms)
    venvelope : ndarray
        Envelope function (will switch to uniform at iteration 800)
    weights : dict
        Weight matrices (FFTs)
    type_masks : dict
        Direction masks
    n_iterations : int
        Number of iterations (default 1000)
        
    Returns:
    --------
    r : ndarray
        Final population activity matrix
    """
    r = np.zeros((n, n))
    
    # Initial movement conditions (zero velocity)
    theta_v = np.pi / 5
    left = -np.sin(theta_v)
    right = np.sin(theta_v)
    up = -np.cos(theta_v)
    down = np.cos(theta_v)
    vel = 0
    
    for iteration in range(n_iterations):
        # Switch to uniform envelope at iteration 800
        if iteration == 800:
            venvelope = np.ones((n, n))
        
        # Compute feedforward input (Equation 4)
        vel_components = {
            'left': vel * left,
            'right': vel * right,
            'up': vel * up,
            'down': vel * down
        }
        rfield = compute_directional_input(venvelope, type_masks, vel_components)
        
        # Convolve with shifted weights (aperiodic boundaries)
        convolution = convolve_with_weights(r, type_masks, weights, periodic=False)
        rfield = rfield + convolution
        
        # Neural transfer function and dynamics (Equation 1)
        fr = np.maximum(0, rfield)
        r = np.minimum(10, (dt/tau) * (5*fr - r) + r)
    
    return r


def run_periodic_simulation(
    position_x, position_y, headDirection,
    n, tau, dt, alpha, useSpiking, module,
    r_init, venvelope, weights, type_masks
):
    """
    Run main simulation with periodic boundaries and actual trajectory.
    
    This is the core simulation where the network follows the animal's trajectory
    and exhibits grid cell firing patterns.
    
    Parameters:
    -----------
    position_x, position_y : ndarray
        Trajectory coordinates
    headDirection : ndarray
        Head direction at each step (radians)
    n : int
        Grid size
    tau : float
        Time constant (ms)
    dt : float
        Time step (ms)
    alpha : float
        Velocity gain parameter
    useSpiking : bool
        Whether to use spiking neuron model
    module : int
        Module number for organizing plots
    r_init : ndarray
        Initial population activity from formation phase
    venvelope : ndarray
        Envelope function (uniform for periodic simulation)
    weights : dict
        Weight matrices (FFTs)
    type_masks : dict
        Direction masks
        
    Returns:
    --------
    spikes : list
        Spike matrices at each time step (if useSpiking=True, else None)
    """
    sampling_length = len(position_x)
    r = r_init.copy()
    s = r.copy()
    
    # Track single neuron (center of grid)
    sNeuron = [n//2, n//2]
    sNeuronResponse = np.zeros(sampling_length)
    
    # Storage for spikes
    spikes = [None] * sampling_length
    
    # Prepare output directory
    output_dir = os.path.join('plots', 'simulation', f'module_{module}')
    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0
    
    increment = 1
    
    for iteration in range(sampling_length - 20):
        # Compute velocity and directional components
        theta_v = headDirection[increment]
        vel = np.sqrt(
            (position_x[increment] - position_x[increment-1])**2 +
            (position_y[increment] - position_y[increment-1])**2
        )
        
        left = -np.cos(theta_v)
        right = np.cos(theta_v)
        up = np.sin(theta_v)
        down = -np.sin(theta_v)
        
        increment += 1
        
        # Compute feedforward input (Equation 4)
        vel_components = {
            'left': alpha * vel * left,
            'right': alpha * vel * right,
            'up': alpha * vel * up,
            'down': alpha * vel * down
        }
        rfield = compute_directional_input(venvelope, type_masks, vel_components)
        
        # Convolve with shifted weights (periodic boundaries)
        convolution = convolve_with_weights(r, type_masks, weights, periodic=True)
        rfield = rfield + convolution
        
        # Neural transfer function and dynamics
        fr = np.maximum(0, rfield)
        r = np.minimum(10, (dt/tau) * (5*fr - r) + r)
        
        # Track single neuron response
        if fr[sNeuron[0], sNeuron[1]] > 0:
            sNeuronResponse[increment] = 1
        
        # Handle spiking model
        if useSpiking:
            spike = rfield * dt > np.random.rand(n, n)
            s = s + (dt/tau) * (-s + (tau/dt) * spike)
            r = s
            spikes[increment] = spike.astype(float)
            sNeuronResponse[increment] = spike[sNeuron[0], sNeuron[1]]
        
        # Plot periodically
        if iteration % 1000 == 0:
            plot_simulation_frame(
                r_new=r,
                sNeuron=sNeuron,
                sNeuronResponse=sNeuronResponse,
                position_x=position_x,
                position_y=position_y,
                increment=increment,
                iteration=iteration,
                sampling_length=sampling_length,
                module=module,
                frame_idx=frame_idx
            )
            frame_idx += 1
    
    return spikes
