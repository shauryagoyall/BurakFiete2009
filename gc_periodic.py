import numpy as np
from scipy.signal import resample
from scipy.fft import fft2, ifft2, fftshift
import os
from gc_plotting import plot_simulation_frame, plot_path_integration_debug, plot_error_over_time, plot_connectivity_matrix, plot_all_shifted_kernels
import tqdm


def get_band_kernel(angle_degrees, X_grid, Y_grid, abar, alphabar):
    """
    Generates an anisotropic interaction kernel (band cell weights) rotated by a specific angle.
    Creates a directional Mexican Hat profile that produces stripe-like patterns.
    
    Parameters:
    -----------
    angle_degrees : float
        Rotation angle in degrees for the kernel orientation
    X_grid : np.ndarray
        2D meshgrid of X coordinates
    Y_grid : np.ndarray
        2D meshgrid of Y coordinates
    abar : float
        Amplitude scaling parameter for the excitatory component
    alphabar : float
        Width parameter controlling the spatial extent of interactions
    
    Returns:
    --------
    filt : np.ndarray
        Anisotropic interaction kernel with stripe pattern perpendicular to angle_degrees
    """
    # Convert angle to radians for trigonometric calculations
    theta = np.deg2rad(angle_degrees)
    
    # Rotate the coordinate system to align with the desired band orientation
    # Only the X-component matters for the 1D band profile
    X_rot = X_grid * np.cos(theta) + Y_grid * np.sin(theta)
    
    # Create anisotropic Mexican Hat profile along the rotated axis
    # This creates excitation along the stripe direction and inhibition perpendicular to it
    # The resulting "stripes" will be perpendicular to the rotation direction
    filt = abar * np.exp(-alphabar * X_rot**2) - np.exp(-X_rot**2)
    
    return filt


def get_periodic_displacement(r_map, last_peak, n):
    """
    Tracks neural activity pattern displacement using center of mass calculation.
    Uses a circular search region around the previous peak to handle periodic boundaries.
    
    Parameters:
    -----------
    r_map : np.ndarray
        Current neural activity map (n x n grid)
    last_peak : tuple
        (row, col) indices of the peak from the previous time step
    n : int
        Grid size (number of neurons along each dimension)
    
    Returns:
    --------
    dx : float
        Displacement in X direction (columns)
    dy : float
        Displacement in Y direction (rows)
    curr_peak : tuple
        (row, col) indices of the new peak position
    """
    # Define circular search region to prevent tracking jumps to distant activity blobs
    mask_radius = 10  # pixels - adjust based on expected movement speed and blob size
    
    # Initialize center of mass calculation variables
    weighted_y = 0.0  # Y-component of weighted position sum
    weighted_x = 0.0  # X-component of weighted position sum
    total_weight = 0.0  # Sum of all activity weights
    
    # Scan circular region around last peak with periodic boundary wrapping
    for i in range(-mask_radius, mask_radius + 1):
        for j in range(-mask_radius, mask_radius + 1):
            # Apply circular mask to limit search region
            if i**2 + j**2 <= mask_radius**2:
                # Compute wrapped indices for periodic boundary conditions
                y_idx = (last_peak[0] + i) % n
                x_idx = (last_peak[1] + j) % n
                
                # Get neural activity at this grid location
                weight = r_map[y_idx, x_idx]
                
                # Accumulate weighted position for center of mass calculation
                if weight > 0:  # Only include active neurons
                    # Use relative coordinates (i, j) to avoid periodic boundary issues
                    weighted_y += i * weight
                    weighted_x += j * weight
                    total_weight += weight
    
    # 2. Calculate center of mass displacement
    if total_weight > 0:
        # Center of mass displacement relative to last_peak
        dy = weighted_y / total_weight
        dx = weighted_x / total_weight
        
        # Round to get the new peak position
        curr_peak = (
            int(np.round(last_peak[0] + dy)) % n,
            int(np.round(last_peak[1] + dx)) % n
        )
    else:
        # Fallback: if no activity in region, stay at last position
        dy = 0
        dx = 0
        curr_peak = last_peak
    
    # Note: dy, dx are already in the correct relative frame and don't need
    # periodic boundary wrapping since they are computed as relative displacements
    # within the local mask region
        
    return dx, dy, curr_peak


def optimize_path_integration(integrated_path_x, integrated_path_y, position_x, position_y, valid_len):
    """
    Performs least squares optimization to fit network position to rat position.
    Finds the best scale factor and offset for X and Y axes.
    
    Parameters:
    -----------
    integrated_path_x : np.ndarray
        Integrated path in network coordinates (X-axis)
    integrated_path_y : np.ndarray
        Integrated path in network coordinates (Y-axis)
    position_x : np.ndarray
        Actual rat position (X-axis) in cm
    position_y : np.ndarray
        Actual rat position (Y-axis) in cm
    valid_len : int
        Number of valid time steps to use for fitting
    
    Returns:
    --------
    integrated_path_x_cm : np.ndarray
        Scaled and offset integrated path (X-axis) in cm
    integrated_path_y_cm : np.ndarray
        Scaled and offset integrated path (Y-axis) in cm
    error : np.ndarray
        Euclidean error at each time step
    final_scale_factor : float
        Average scale factor across X and Y axes
    """
    # Linear regression: Rat_Position = slope * Network_Position + intercept
    # Automatically determines the optimal scale factor and spatial offset

    # Fit X-axis: Find linear relationship between network and real-world X coordinates
    slope_x, intercept_x = np.polyfit(
        integrated_path_x[:valid_len], 
        position_x[:valid_len], 
        1  # Linear fit (degree 1 polynomial)
    )
    integrated_path_x_cm = integrated_path_x * slope_x + intercept_x

    # Fit Y-axis: Independent fit allows for anisotropic grid distortions
    # (e.g., if the grid is stretched differently in X vs Y)
    slope_y, intercept_y = np.polyfit(
        integrated_path_y[:valid_len], 
        position_y[:valid_len], 
        1
    )
    integrated_path_y_cm = integrated_path_y * slope_y + intercept_y

    # Compute average scale factor across both axes
    # Use absolute values to handle axis flips (negative slopes)
    final_scale_factor = (np.abs(slope_x) + np.abs(slope_y)) / 2.0
    print(f"Optimized Scale Factor: {final_scale_factor:.4f}")
    print(f"X Offset: {intercept_x:.2f}, Y Offset: {intercept_y:.2f}")

    # Calculate Euclidean error between actual and predicted positions
    err_x = position_x[:valid_len] - integrated_path_x_cm[:valid_len]
    err_y = position_y[:valid_len] - integrated_path_y_cm[:valid_len]
    print(position_x[:], integrated_path_x_cm[:])  # Debug output
    error = np.sqrt(err_x**2 + err_y**2)
    
    return integrated_path_x_cm, integrated_path_y_cm, error, final_scale_factor

def gc_periodic(filename, n, tau, dt, beta, gamma, abar, wtphase, alpha, useSpiking, module, 
                GET_BAND=False, BAND_ANGLE=None, duration=100000):
    """
    Simulates grid cell dynamics with periodic boundary conditions and path integration.
    
    Parameters:
    -----------
    filename : str
        Path to trajectory data file (.npz format)
    n : int
        Grid size (number of neurons per dimension)
    tau : float
        Neural time constant (ms)
    dt : float
        Time step for integration (ms)
    beta : float
        Spatial scale parameter
    gamma : float
        Interaction range parameter
    abar : float
        Excitatory amplitude in interaction kernel
    wtphase : int
        Phase shift for directional weight matrices (in pixels)
    alpha : float
        Velocity gain parameter for path integration
    useSpiking : bool
        Whether to use spiking dynamics (True) or rate model (False)
    module : int
        Module identifier for output organization
    GET_BAND : bool, optional
        Use anisotropic band kernel instead of isotropic (default: False)
    BAND_ANGLE : float, optional
        Angle for band kernel orientation (required if GET_BAND=True)
    duration : int, optional
        Number of time steps for random trajectory (default: 100000)
    
    Returns:
    --------
    spikes : list
        Spike patterns at each time step (if useSpiking=True)
    integrated_path_x_cm : np.ndarray
        Network-integrated X trajectory scaled to cm
    integrated_path_y_cm : np.ndarray
        Network-integrated Y trajectory scaled to cm
    error : np.ndarray
        Path integration error over time
    position_x : np.ndarray
        Actual X trajectory
    position_y : np.ndarray
        Actual Y trajectory
    """
    # Set random seed for reproducible trajectory generation
    np.random.seed(42)
    
    # Validate band kernel parameters
    if GET_BAND:
        assert BAND_ANGLE is not None, "BAND_ANGLE must be specified when GET_BAND is True."
    
    # Setup output directory structure for plots
    output_dir = os.path.join('plots', 'simulation', f'module_{module}')
    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0  # Counter for animation frames

    #====================================
    # DATA LOADING AND TRAJECTORY SETUP
    #====================================
    FileLoad = 0  # Flag to track whether data was loaded from file
    
    if os.path.exists(filename):
        # Load experimental trajectory data from file
        data = np.load(filename)
        position_x = data['position_x']  # X coordinates in cm
        position_y = data['position_y']  # Y coordinates in cm
        FileLoad = 1
    else:
        # Generate synthetic random walk trajectory within circular arena
        enclosureRadius = 2 * 100  # Arena radius: 2 meters = 200 cm
        temp_velocity = np.random.rand() / 2  # Initial velocity (cm/ms)
        
        # Initialize position and direction arrays
        position_x = np.zeros(duration)
        position_y = np.zeros(duration)
        headDirection = np.zeros(duration)
        
        # Set initial conditions
        position_x[0] = 0  # Start at arena center
        position_y[0] = 0
        headDirection[0] = np.random.rand() * 2 * np.pi  # Random initial heading
        
        # Generate random walk trajectory step by step
        for i in range(1, duration):
            # Add random acceleration (Gaussian noise, clipped to ±0.2 cm/ms²)
            temp_rand = np.clip(np.random.normal(0, 0.05), -0.2, 0.2)
            
            # Update velocity with acceleration, constrained to 0-0.25 cm/ms
            temp_velocity = np.clip(temp_velocity + temp_rand, 0, 0.25)
            
            # Boundary enforcement: redirect heading if agent would exit arena
            leftOrRight = 1 if np.random.rand() > 0.5 else -1  # Random turn direction
            
            # Check if next position would be outside circular boundary
            while (np.sqrt((position_x[i-1] + np.cos(headDirection[i-1]) * temp_velocity)**2 +
                           (position_y[i-1] + np.sin(headDirection[i-1]) * temp_velocity)**2) > enclosureRadius):
                # Gradually rotate heading away from boundary
                headDirection[i-1] = headDirection[i-1] + leftOrRight * np.pi / 100
            
            # Update position based on current heading and velocity
            position_x[i] = position_x[i-1] + np.cos(headDirection[i-1]) * temp_velocity
            position_y[i] = position_y[i-1] + np.sin(headDirection[i-1]) * temp_velocity
            
            # Add random walk to heading direction (uniform noise ±π/20 rad)
            headDirection[i] = np.mod(
                headDirection[i-1] + (np.random.rand() - 0.5) / 5 * np.pi / 2, 
                2 * np.pi
            )
    
    sampling_length = len(position_x)
    
    if FileLoad == 1:
        # Resample loaded data to standard time step of 0.5 ms
        if dt != 0.5:
            # For very high sampling rates (dt < 0.1 ms), downsample first
            if dt < 0.1:
                downsample_factor = int(np.floor(0.5 / dt))
                position_x = position_x[::downsample_factor]
                position_y = position_y[::downsample_factor]
                dt = downsample_factor * dt
            
            # Round dt to nearest 0.1 ms for numerical stability
            dt = np.round(dt * 10) / 10
            
            # Interpolate to finer resolution
            position_x = resample(position_x, len(position_x) * int(dt * 10))
            position_y = resample(position_y, len(position_y) * int(dt * 10))
            
            # Downsample to final 0.5 ms resolution
            position_x = position_x[::5]
            position_y = position_y[::5]
            dt = 0.5
        
        sampling_length = len(position_x)
        
        # Compute head direction from position changes
        headDirection = np.zeros(sampling_length)
        for i in range(sampling_length - 1):
            # Calculate angle of movement vector using arctangent
            headDirection[i] = np.mod(
                np.arctan2(position_y[i+1] - position_y[i],
                          position_x[i+1] - position_x[i]), 
                2 * np.pi
            )
        # Last time step: maintain previous heading
        headDirection[sampling_length-1] = headDirection[sampling_length-2]
    
    #===============================
    # NEURAL NETWORK INITIALIZATION
    #===============================
    
    # Grid dimensions for FFT convolution (padded to avoid edge effects)
    big = 2 * n  # Padded grid size for accurate FFT convolution
    dim = n // 2  # Half-grid for block matrix construction
    
    # Neural population activity arrays
    r = np.zeros((n, n))       # Current firing rates
    rfield = np.zeros((n, n))  # Input field (before rectification)
    s = np.zeros((n, n))       # Synaptic current (for spiking model)
    
    # Recording arrays
    spikes = [None] * sampling_length  # Spike patterns over time
    sNeuronResponse = np.zeros(sampling_length)  # Single neuron recording
    sNeuron = [n//2, n//2]  # Coordinates of recorded neuron (center of grid)
    
    # Spatial coordinate system for weight matrices
    x = np.arange(-n/2, n/2)  # 1D spatial axis
    lx = len(x)  # Length of spatial axis
    xbar = np.sqrt(beta) * x  # Scaled coordinates for weight functions
    
    #========================================
    # SYNAPTIC WEIGHT MATRIX CONSTRUCTION
    #========================================
    
    # Compute scaled width parameter
    alphabar = gamma / beta
    
    # Create 2D spatial grid for weight matrix computation
    X, Y = np.meshgrid(xbar, xbar)
    
    # Build interaction kernel (Mexican Hat profile)
    if GET_BAND == True:
        # Anisotropic kernel for band cells (directionally selective)
        filt = get_band_kernel(BAND_ANGLE, X, Y, abar, alphabar)
    else:
        # Isotropic kernel for grid cells (rotationally symmetric)
        # Form: excitation (positive Gaussian) - inhibition (wider negative Gaussian)
        filt = abar * np.exp(-alphabar * (X**2 + Y**2)) - np.exp(-(X**2 + Y**2))
    
    # Envelope function: soft boundary conditions during pattern formation
    # Gradually reduces to 1 (no effect) after stabilization
    x_env = x[:, np.newaxis]
    venvelope = np.exp(-4 * (x_env**2 + x**2) / (n/2)**2)
    
    # Create directionally-shifted weight matrices for velocity-dependent modulation
    # Each direction (right, left, down, up) gets its own phase-shifted kernel
    frshift = np.roll(filt, wtphase, axis=1)   # Right shift
    flshift = np.roll(filt, -wtphase, axis=1)  # Left shift
    fdshift = np.roll(filt, wtphase, axis=0)   # Down shift
    fushift = np.roll(filt, -wtphase, axis=0)  # Up shift
    
    # Precompute FFT of weight matrices for efficient convolution (large grid)
    ftu = fft2(fushift, s=(big, big))  # Up direction
    ftd = fft2(fdshift, s=(big, big))  # Down direction
    ftl = fft2(flshift, s=(big, big))  # Left direction
    ftr = fft2(frshift, s=(big, big))  # Right direction
    
    # Precompute FFT for smaller grid (used during trajectory simulation)
    ftu_small = fft2(fftshift(fushift))
    ftd_small = fft2(fftshift(fdshift))
    ftl_small = fft2(fftshift(flshift))
    ftr_small = fft2(fftshift(frshift))
    
    # Generate diagnostic plots of connectivity structure
    connectivity_output_dir = os.path.join('plots', 'connectivity', f'module_{module}')
    plot_connectivity_matrix(filt, abar, alphabar, module, connectivity_output_dir)
    plot_all_shifted_kernels(filt, wtphase, module, connectivity_output_dir)
    
    # Block matrices for velocity-dependent gain modulation
    # Each matrix selects a specific 2x2 checkerboard pattern
    typeL = np.tile(np.array([[1, 0], [0, 0]]), (dim, dim))  # Left-selective neurons
    typeR = np.tile(np.array([[0, 0], [0, 1]]), (dim, dim))  # Right-selective neurons
    typeU = np.tile(np.array([[0, 1], [0, 0]]), (dim, dim))  # Up-selective neurons
    typeD = np.tile(np.array([[0, 0], [1, 0]]), (dim, dim))  # Down-selective neurons
    
    #====================================
    # INITIAL CONDITIONS AND WARM-UP
    #====================================
    
    # Initial movement direction parameters (not used during formation)
    theta_v = np.pi / 5  # Initial heading angle
    left = -np.sin(theta_v)
    right = np.sin(theta_v)
    up = -np.cos(theta_v)
    down = np.cos(theta_v)
    vel = 0  # Zero velocity during pattern formation
    
    #=========================================
    # PHASE 1: PATTERN FORMATION (Warm-up)
    #=========================================
    # Allow grid pattern to stabilize before trajectory tracking begins
    for iteration in range(1000):
        # Remove envelope constraint after 800 steps (pattern is stable)
        if iteration == 800:
            venvelope = np.ones((n, n))  # No spatial constraints
        
        # Compute input field with velocity modulation (zero during formation)
        rfield = venvelope * (
            (1 + vel * right) * typeR + 
            (1 + vel * left) * typeL +
            (1 + vel * up) * typeU + 
            (1 + vel * down) * typeD
        )
        
        # Compute recurrent synaptic input via FFT convolution
        convolution = np.real(ifft2(
            fft2(r * typeR, s=(big, big)) * ftr +
            fft2(r * typeL, s=(big, big)) * ftl +
            fft2(r * typeD, s=(big, big)) * ftd +
            fft2(r * typeU, s=(big, big)) * ftu
        ))
        
        # Extract central region (remove padding) and add to input field
        rfield = rfield + convolution[n//2:big-n//2, n//2:big-n//2]
        
        # Apply rectification (ReLU nonlinearity)
        fr = np.maximum(0, rfield)
        
        # Integrate firing rate dynamics: dr/dt = (5*f(r) - r) / tau
        r_old = r
        r_new = np.minimum(10, (dt/tau) * (5*fr - r_old) + r_old)  # Cap at 10 Hz
        r = r_new
    
    #======================================================
    # PHASE 2: TRAJECTORY TRACKING WITH PATH INTEGRATION
    #======================================================
    
    increment = 1  # Time step counter
    s = r.copy()   # Initialize synaptic variable for spiking model
    
    # ========== PATH INTEGRATION SETUP ==========
    # Track network activity displacement to decode position
    integrated_path_x = np.zeros(sampling_length)  # X trajectory from network
    integrated_path_y = np.zeros(sampling_length)  # Y trajectory from network
    curr_integ_x = 0.0  # Accumulated X displacement
    curr_integ_y = 0.0  # Accumulated Y displacement
    
    # Find initial peak position in stabilized grid pattern
    start_peak_flat = np.argmax(r)
    last_peak_pos = np.unravel_index(start_peak_flat, r.shape)
    # ============================================

    # Main simulation loop: process each time step of the trajectory
    for iteration in tqdm.tqdm(range(sampling_length - 20)):
        
        # Extract current movement parameters from trajectory
        theta_v = headDirection[increment]  # Current heading angle
        vel = np.sqrt(
            (position_x[increment] - position_x[increment-1])**2 +
            (position_y[increment] - position_y[increment-1])**2
        )  # Speed (cm/ms)
        
        # Decompose velocity into directional components
        left = -np.cos(theta_v)
        right = np.cos(theta_v)
        up = np.sin(theta_v)
        down = -np.sin(theta_v)
        
        increment += 1
        
        # Compute velocity-modulated input field
        # Velocity shifts the activity bump via asymmetric gain modulation
        rfield = venvelope * (
            (1 + alpha * vel * right) * typeR +
            (1 + alpha * vel * left) * typeL +
            (1 + alpha * vel * up) * typeU +
            (1 + alpha * vel * down) * typeD
        )
        
        # Compute recurrent connectivity using FFT convolution (no padding)
        convolution = np.real(ifft2(
            fft2(r * typeR) * ftr_small +
            fft2(r * typeL) * ftl_small +
            fft2(r * typeD) * ftd_small +
            fft2(r * typeU) * ftu_small
        ))
        
        # Combine feedforward and recurrent inputs
        rfield = rfield + convolution
        
        # Apply threshold-linear activation (ReLU)
        fr = np.maximum(0, rfield)
        
        # Update firing rates using forward Euler integration
        r_old = r
        r_new = np.minimum(10, (dt/tau) * (5*fr - r_old) + r_old)
        r = r_new
        
        # ============ PATH INTEGRATION TRACKING ============
        # Track activity bump displacement to decode spatial position
        dx, dy, new_peak_pos = get_periodic_displacement(r_new, last_peak_pos, n)
        
        # Accumulate displacement in network coordinates
        # Note: Matrix convention has row=Y (axis 0), column=X (axis 1)
        curr_integ_y -= dy  # Y displacement (rows, typically inverted)
        curr_integ_x += dx  # X displacement (columns)
        
        # Store integrated path at current time step
        integrated_path_x[increment] = curr_integ_x
        integrated_path_y[increment] = curr_integ_y
        
        # Update peak position for next iteration
        last_peak_pos = new_peak_pos
        # ===================================================
        
        # Record activity of single monitored neuron
        if fr[sNeuron[0], sNeuron[1]] > 0:
            sNeuronResponse[increment] = 1
        
        # Spiking model: stochastic spike generation
        if useSpiking:
            # Poisson spike generation: spike probability ~ firing rate * dt
            spike = rfield * dt > np.random.rand(n, n)
            
            # Update synaptic current with exponential decay
            s = s + (dt/tau) * (-s + (tau/dt) * spike)
            r = s
            
            # Record spikes
            spikes[increment] = spike.astype(float)
            sNeuronResponse[increment] = spike[sNeuron[0], sNeuron[1]]
        
        # Generate diagnostic plots periodically during simulation
        if iteration % 1000 == 0:
            plot_simulation_frame(
                r_new=r_new,
                sNeuron=sNeuron,
                sNeuronResponse=sNeuronResponse,
                position_x=position_x,
                position_y=position_y,
                increment=increment,
                iteration=iteration,
                sampling_length=sampling_length,
                module=module,
                frame_idx=frame_idx,
            )
            frame_idx += 1

    #=======================================================
    # PHASE 3: POST-PROCESSING AND ERROR ANALYSIS
    #=======================================================
    # Fit network coordinates to real-world coordinates
    valid_len = increment
    integrated_path_x_cm, integrated_path_y_cm, error, final_scale_factor = optimize_path_integration(
        integrated_path_x, integrated_path_y, position_x, position_y, valid_len
    )
    
    #==================================
    # GENERATE ANALYSIS VISUALIZATIONS
    #==================================
    plot_path_integration_debug(
        position_x=position_x,
        position_y=position_y,
        integrated_path_x_cm=integrated_path_x_cm,
        integrated_path_y_cm=integrated_path_y_cm,
        valid_len=valid_len,
        dt=dt,
        scale_factor=final_scale_factor,
        module=module,
        output_dir=output_dir
    )
    
    # Plot error over time
    plot_error_over_time(
        error=error,
        dt=dt,
        module=module,
        output_dir=output_dir
    )
    
    return spikes, integrated_path_x_cm, integrated_path_y_cm, error, position_x, position_y
