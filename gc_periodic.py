import numpy as np
from scipy.signal import resample
from scipy.fft import fft2, ifft2, fftshift
import os
from gc_plotting import plot_simulation_frame, plot_path_integration_debug 

np.random.seed(42)  # For reproducibility

def get_periodic_displacement(r_map, last_peak, n):
    """
    Finds the shift of the pattern by tracking the blob closest to the 
    last known peak location, handling periodic boundaries.
    """
    # 1. Mask the grid to only look near the last peak to avoid jumping to a different blob
    mask_radius = 10  # Search radius in pixels (tune if blobs are very fast/large)
    mask = np.zeros_like(r_map)
    
    # Create a mask that handles periodic wrapping
    for i in range(-mask_radius, mask_radius + 1):
        for j in range(-mask_radius, mask_radius + 1):
            y_idx = (last_peak[0] + i) % n
            x_idx = (last_peak[1] + j) % n
            mask[y_idx, x_idx] = 1
            
    # Find the max value only within the masked area
    masked_r = r_map * mask
    curr_peak_flat = np.argmax(masked_r)
    curr_peak = np.unravel_index(curr_peak_flat, r_map.shape)
    
    # 2. Calculate raw displacement
    dy = curr_peak[0] - last_peak[0]
    dx = curr_peak[1] - last_peak[1]
    
    # 3. Handle Periodic Boundaries (Phase Unwrapping)
    # If the jump is larger than half the grid, it actually wrapped around
    if dy > n / 2:
        dy -= n
    elif dy < -n / 2:
        dy += n
        
    if dx > n / 2:
        dx -= n
    elif dx < -n / 2:
        dx += n
        
    return dx, dy, curr_peak

def gc_periodic(filename, n, tau, dt, beta, gamma, abar, wtphase, alpha, useSpiking, module):
    """
    Grid Cell Dynamics - Periodic with Path Integration
    """
    
    # Prepare output directory for sequential plots
    output_dir = os.path.join('plots', 'simulation', f'module_{module}')
    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0

    #---------------------
    # LOAD AND CLEAN DATA
    #---------------------
    FileLoad = 0
    
    if os.path.exists(filename):
        # Load data from file (assuming .npz or similar format)
        data = np.load(filename)
        position_x = data['position_x']
        position_y = data['position_y']
        FileLoad = 1
    else:
        # If no data is loaded, use random trajectories.
        enclosureRadius = 2 * 100  # Two meters
        temp_velocity = np.random.rand() / 2
        position_x = np.zeros(100000)
        position_y = np.zeros(100000)
        headDirection = np.zeros(100000)
        position_x[0] = 0
        position_y[0] = 0
        headDirection[0] = np.random.rand() * 2 * np.pi
        
        for i in range(1, 100000):
            # max acceleration is .1 cm/ms^2
            temp_rand = np.clip(np.random.normal(0, 0.05), -0.2, 0.2)
            
            # max velocity is .5 cm/ms
            temp_velocity = np.clip(temp_velocity + temp_rand, 0, 0.25)
            
            # Don't let trajectory go outside of the boundary
            leftOrRight = 1 if np.random.rand() > 0.5 else -1
            
            while (np.sqrt((position_x[i-1] + np.cos(headDirection[i-1]) * temp_velocity)**2 +
                           (position_y[i-1] + np.sin(headDirection[i-1]) * temp_velocity)**2) > enclosureRadius):
                headDirection[i-1] = headDirection[i-1] + leftOrRight * np.pi / 100
            
            position_x[i] = position_x[i-1] + np.cos(headDirection[i-1]) * temp_velocity
            position_y[i] = position_y[i-1] + np.sin(headDirection[i-1]) * temp_velocity
            headDirection[i] = np.mod(headDirection[i-1] + (np.random.rand() - 0.5) / 5 * np.pi / 2, 2 * np.pi)
    
    sampling_length = len(position_x)
    
    if FileLoad == 1:
        # Linearly interpolate data to scale to .5 ms
        if dt != 0.5:
            if dt < 0.1:
                downsample_factor = int(np.floor(0.5 / dt))
                position_x = position_x[::downsample_factor]
                position_y = position_y[::downsample_factor]
                dt = downsample_factor * dt
            
            dt = np.round(dt * 10) / 10
            
            # Interpolate
            position_x = resample(position_x, len(position_x) * int(dt * 10))
            position_y = resample(position_y, len(position_y) * int(dt * 10))
            
            # Downsample
            position_x = position_x[::5]
            position_y = position_y[::5]
            dt = 0.5
        
        sampling_length = len(position_x)
        
        # Add in head directions
        headDirection = np.zeros(sampling_length)
        for i in range(sampling_length - 1):
            headDirection[i] = np.mod(np.arctan2(position_y[i+1] - position_y[i],
                                                 position_x[i+1] - position_x[i]), 2 * np.pi)
        headDirection[sampling_length-1] = headDirection[sampling_length-2]
    
    #----------------------
    # INITIALIZE VARIABLES
    #----------------------
    
    big = 2 * n
    dim = n // 2
    
    # initial population activity
    r = np.zeros((n, n))
    rfield = np.zeros((n, n))
    s = np.zeros((n, n))
    
    spikes = [None] * sampling_length
    sNeuronResponse = np.zeros(sampling_length)
    sNeuron = [n//2, n//2]
    
    # Envelope and Weight Matrix parameters
    x = np.arange(-n/2, n/2)
    lx = len(x)
    xbar = np.sqrt(beta) * x
    
    #------------------------------------
    # INITIALIZE SYNAPTIC WEIGHT MATRICES
    #------------------------------------
    
    alphabar = gamma / beta
    
    # The center surround, locally inhibitory weight matrix - Equation (3)
    X, Y = np.meshgrid(xbar, xbar)
    filt = abar * np.exp(-alphabar * (X**2 + Y**2)) - np.exp(-(X**2 + Y**2))
    
    # The envelope function
    x_env = x[:, np.newaxis]
    venvelope = np.exp(-4 * (x_env**2 + x**2) / (n/2)**2)
    
    # Shifted weight matrices
    frshift = np.roll(filt, wtphase, axis=1)
    flshift = np.roll(filt, -wtphase, axis=1)
    fdshift = np.roll(filt, wtphase, axis=0)
    fushift = np.roll(filt, -wtphase, axis=0)
    
    ftu = fft2(fushift, s=(big, big))
    ftd = fft2(fdshift, s=(big, big))
    ftl = fft2(flshift, s=(big, big))
    ftr = fft2(frshift, s=(big, big))
    
    ftu_small = fft2(fftshift(fushift))
    ftd_small = fft2(fftshift(fdshift))
    ftl_small = fft2(fftshift(flshift))
    ftr_small = fft2(fftshift(frshift))
    
    # Block matrices
    typeL = np.tile(np.array([[1, 0], [0, 0]]), (dim, dim))
    typeR = np.tile(np.array([[0, 0], [0, 1]]), (dim, dim))
    typeU = np.tile(np.array([[0, 1], [0, 0]]), (dim, dim))
    typeD = np.tile(np.array([[0, 0], [1, 0]]), (dim, dim))
    
    #----------------------------
    # INITIAL MOVEMENT CONDITIONS
    #----------------------------
    
    theta_v = np.pi / 5
    left = -np.sin(theta_v)
    right = np.sin(theta_v)
    up = -np.cos(theta_v)
    down = np.cos(theta_v)
    vel = 0
    
    #------------------
    # BEGIN SIMULATION 
    #------------------
    # First loop: Formation / Healing phase (stationary or small movement)
    for iteration in range(1000):
        if iteration == 800:
            venvelope = np.ones((n, n))
        
        rfield = venvelope * ((1 + vel * right) * typeR + (1 + vel * left) * typeL +
                             (1 + vel * up) * typeU + (1 + vel * down) * typeD)
        
        convolution = np.real(ifft2(
            fft2(r * typeR, s=(big, big)) * ftr +
            fft2(r * typeL, s=(big, big)) * ftl +
            fft2(r * typeD, s=(big, big)) * ftd +
            fft2(r * typeU, s=(big, big)) * ftu
        ))
        
        rfield = rfield + convolution[n//2:big-n//2, n//2:big-n//2]
        fr = np.maximum(0, rfield)
        
        r_old = r
        r_new = np.minimum(10, (dt/tau) * (5*fr - r_old) + r_old)
        r = r_new
    
    #----------------------------------------------------------
    # COMPUTE NEURAL POPULATION ACTIVITY WITH PERIODIC BOUNDARY
    #----------------------------------------------------------
    
    increment = 1
    s = r.copy()
    
    # --- PATH INTEGRATION INITIALIZATION ---
    integrated_path_x = np.zeros(sampling_length)
    integrated_path_y = np.zeros(sampling_length)
    curr_integ_x = 0.0
    curr_integ_y = 0.0
    
    # Initialize tracker at the peak of the current stable pattern
    start_peak_flat = np.argmax(r)
    last_peak_pos = np.unravel_index(start_peak_flat, r.shape)
    # ----------------------------------------

    # Trajectory Loop
    for iteration in range(sampling_length - 20):
        
        theta_v = headDirection[increment]
        vel = np.sqrt((position_x[increment] - position_x[increment-1])**2 +
                      (position_y[increment] - position_y[increment-1])**2)
        left = -np.cos(theta_v)
        right = np.cos(theta_v)
        up = np.sin(theta_v)
        down = -np.sin(theta_v)
        
        increment += 1
        
        rfield = venvelope * ((1 + alpha * vel * right) * typeR +
                             (1 + alpha * vel * left) * typeL +
                             (1 + alpha * vel * up) * typeU +
                             (1 + alpha * vel * down) * typeD)
        
        convolution = np.real(ifft2(
            fft2(r * typeR) * ftr_small +
            fft2(r * typeL) * ftl_small +
            fft2(r * typeD) * ftd_small +
            fft2(r * typeU) * ftu_small
        ))
        
        rfield = rfield + convolution
        fr = np.maximum(0, rfield)
        
        r_old = r
        r_new = np.minimum(10, (dt/tau) * (5*fr - r_old) + r_old)
        r = r_new
        
        # --- PATH INTEGRATION TRACKING ---
        # Calculate how much the pattern moved this step
        dx, dy, new_peak_pos = get_periodic_displacement(r_new, last_peak_pos, n)
        
        # Accumulate (careful with X/Y orientation relative to your grid)
        # Often in matrices, index 0 is Y (rows) and index 1 is X (cols).
        # We will store them naturally first:
        curr_integ_y -= dy  # rows
        curr_integ_x += dx  # cols
        
        integrated_path_x[increment] = curr_integ_x
        integrated_path_y[increment] = curr_integ_y
        last_peak_pos = new_peak_pos
        # ---------------------------------
        
        if fr[sNeuron[0], sNeuron[1]] > 0:
            sNeuronResponse[increment] = 1
        
        if useSpiking:
            spike = rfield * dt > np.random.rand(n, n)
            s = s + (dt/tau) * (-s + (tau/dt) * spike)
            r = s
            spikes[increment] = spike.astype(float)
            sNeuronResponse[increment] = spike[sNeuron[0], sNeuron[1]]
        
        # Plotting (Using your existing function)
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
                frame_idx=frame_idx
            )
            frame_idx += 1

    #----------------------
    # POST-PROCESSING (Least Squares Optimization)
    #----------------------
    valid_len = increment

    # We want to fit: Rat_Position = slope * Network_Position + intercept
    # This finds the best Scale (slope) and Starting Offset (intercept) automatically.

    # 1. Fit X-Axis
    slope_x, intercept_x = np.polyfit(integrated_path_x[:valid_len], position_x[:valid_len], 1)
    integrated_path_x_cm = integrated_path_x * slope_x + intercept_x

    # 2. Fit Y-Axis
    # We fit them separately to account for potential anisotropy (stretching) in the grid
    slope_y, intercept_y = np.polyfit(integrated_path_y[:valid_len], position_y[:valid_len], 1)
    integrated_path_y_cm = integrated_path_y * slope_y + intercept_y

    # 3. Calculate Scale Factor for reporting
    # The true scale factor is the average of the X and Y slopes magnitude
    # (Note: slope_y might be negative if the axis was inverted, handling the flip automatically)
    final_scale_factor = (np.abs(slope_x) + np.abs(slope_y)) / 2.0
    print(f"Optimized Scale Factor: {final_scale_factor:.4f}")
    print(f"X Offset: {intercept_x:.2f}, Y Offset: {intercept_y:.2f}")

    # 4. Calculate Error
    err_x = position_x[:valid_len] - integrated_path_x_cm[:valid_len]
    err_y = position_y[:valid_len] - integrated_path_y_cm[:valid_len]
    error = np.sqrt(err_x**2 + err_y**2)
    
    #----------------------
    # DEBUG PLOTTING
    #----------------------
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
    
    return spikes, integrated_path_x_cm, integrated_path_y_cm, error