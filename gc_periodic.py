import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.fft import fft2, ifft2, fftshift
import os


def gc_periodic(filename, n, tau, dt, beta, gamma, abar, wtphase, alpha, useSpiking):
    """
    Grid Cell Dynamics - Periodic
    
    Parameters:
    -----------
    filename : str
        Path to data file
    n : int
        Grid size
    tau : float
        Time constant
    dt : float
        Time step
    beta : float
        Beta parameter
    gamma : float
        gamma bar parameter
    abar : float
        A bar parameter
    wtphase : int
        Weight phase
    alpha : float
        Alpha parameter
    useSpiking : bool
        Whether to use spiking model
        
    Returns:
    --------
    spikes : list
        List of spike matrices
    """
    
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
        # Random head directions between 0 and 2*pi with no more than 20 degree
        # turn at each time step and trajectory based off of the previous time
        # step's head direction
        
        enclosureRadius = 2 * 100  # Two meters
        temp_velocity = np.random.rand() / 2
        position_x = np.zeros(1000000)
        position_y = np.zeros(1000000)
        headDirection = np.zeros(1000000)
        position_x[0] = 0
        position_y[0] = 0
        headDirection[0] = np.random.rand() * 2 * np.pi
        
        for i in range(1, 1000000):
            # max acceleration is .1 cm/ms^2
            temp_rand = np.clip(np.random.normal(0, 0.05), -0.2, 0.2)
            
            # max velocity is .5 cm/ms
            temp_velocity = np.clip(temp_velocity + temp_rand, 0, 0.25)
            
            # Don't let trajectory go outside of the boundary, if it would then randomly
            # rotate to the left or right
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
                # If data is too fine, then downsample to make computations faster
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
    
    # padding for convolutions
    big = 2 * n
    dim = n // 2
    
    # initial population activity
    r = np.zeros((n, n))
    rfield = np.zeros((n, n))
    s = np.zeros((n, n))
    
    # A placeholder for spiking activity
    spikes = [None] * sampling_length
    
    # A placeholder for a single neuron response
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
    
    # The envelope function that determines the global feedforward input - Equation (5)
    x_env = x[:, np.newaxis]
    venvelope = np.exp(-4 * (x_env**2 + x**2) / (n/2)**2)
    
    # We create shifted weight matrices for each preferred firing direction
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
    
    # Block matrices used for identifying all neurons of one preferred firing direction
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
    # plt.ion()  # Turn on interactive mode
    # fig, ax = plt.subplots(figsize=(5.5, 4.5))
    
    # We run the simulation for 300 ms with aperiodic boundaries and 
    # zero velocity to form the network, then we change the 
    # envelope function to uniform input and continue the 
    # simulation with periodic boundary conditions
    
    for iteration in range(1000):
        
        #----------------------------------------
        # COMPUTE NEURAL POPULATION ACTIVITY 
        #----------------------------------------
        if iteration == 800:
            venvelope = np.ones((n, n))
        
        # Break global input into its directional components - Equation (4)
        rfield = venvelope * ((1 + vel * right) * typeR + (1 + vel * left) * typeL +
                             (1 + vel * up) * typeU + (1 + vel * down) * typeD)
        
        # Convolute population activity with shifted symmetric weights.
        convolution = np.real(ifft2(
            fft2(r * typeR, s=(big, big)) * ftr +
            fft2(r * typeL, s=(big, big)) * ftl +
            fft2(r * typeD, s=(big, big)) * ftd +
            fft2(r * typeU, s=(big, big)) * ftu
        ))
        
        # Add feedforward inputs to the shifted population activity
        rfield = rfield + convolution[n//2:big-n//2, n//2:big-n//2]
        
        # Neural Transfer Function
        fr = np.maximum(0, rfield)
        
        # Neuron dynamics - Equation (1)
        r_old = r
        r_new = np.minimum(10, (dt/tau) * (5*fr - r_old) + r_old)
        r = r_new
        
        # Update the plot every 100 timesteps
        # if iteration % 100 == 0:
            # ax.clear()
            # im = ax.imshow(r_new, cmap='hot', vmin=0, vmax=2)
            # ax.set_title(f'Neural Population Activity (Iteration {iteration}/1000)')
            # if iteration == 0:
            #     plt.colorbar(im, ax=ax)
            # plt.draw()
            # plt.pause(0.001)
    
    #----------------------------------------------------------
    # COMPUTE NEURAL POPULATION ACTIVITY WITH PERIODIC BOUNDARY
    #-----------------------------------------------------------
    
    # increment is the position in the trajectory data. start at 2 to compute velocity
    increment = 1
    s = r.copy()
    
    # Create figure for trajectory tracking
    # plt.close('all')
    
    
    for iteration in range(sampling_length - 20):
        
        theta_v = headDirection[increment]
        vel = np.sqrt((position_x[increment] - position_x[increment-1])**2 +
                     (position_y[increment] - position_y[increment-1])**2)
        left = -np.cos(theta_v)
        right = np.cos(theta_v)
        up = np.sin(theta_v)
        down = -np.sin(theta_v)
        
        increment += 1
        
        # Break feedforward input into its directional components - Equation (4)
        rfield = venvelope * ((1 + alpha * vel * right) * typeR +
                             (1 + alpha * vel * left) * typeL +
                             (1 + alpha * vel * up) * typeU +
                             (1 + alpha * vel * down) * typeD)
        
        # Convolute population activity with shifted symmetric weights.
        convolution = np.real(ifft2(
            fft2(r * typeR) * ftr_small +
            fft2(r * typeL) * ftl_small +
            fft2(r * typeD) * ftd_small +
            fft2(r * typeU) * ftu_small
        ))
        
        # Add feedforward inputs to the shifted population activity
        rfield = rfield + convolution
        
        # Neural Transfer Function
        fr = np.maximum(0, rfield)
        
        # Neuron dynamics (Eq. 1)
        r_old = r
        r_new = np.minimum(10, (dt/tau) * (5*fr - r_old) + r_old)
        r = r_new
        
        # Track single neuron response
        if fr[sNeuron[0], sNeuron[1]] > 0:
            sNeuronResponse[increment] = 1
        
        if useSpiking:
            spike = rfield * dt > np.random.rand(n, n)
            
            # Neurons decay according to Equation (6)
            s = s + (dt/tau) * (-s + (tau/dt) * spike)
            r = s
            spikes[increment] = spike.astype(float)
            
            # Track a single neuron response
            sNeuronResponse[increment] = spike[sNeuron[0], sNeuron[1]]
        
        #-----------------------------------------
        # PLOTS
        #-----------------------------------------
        
        if iteration % 10000 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            
            # Plot neural population activity
            im1 = ax1.imshow(r_new, cmap='hot', vmin=0, vmax=2)
            ax1.set_title(f'Neural Population Activity\n(Step {iteration}/{sampling_length-20})')
            ax1.set_aspect('equal', adjustable='box')
            ax1.plot(sNeuron[1], sNeuron[0], 'bo', markersize=6, label='Tracked Neuron')
            
            
            # Single neuron response plot
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
            plt.show()
            plt.close(fig)
    
    # # Final plot
    # print("\nSimulation complete! Displaying final results...")
    # ax1.clear()
    # ax2.clear()
    
    # # Final neural activity
    # im1 = ax1.imshow(r_new, cmap='hot', vmin=0, vmax=2)
    # ax1.set_title('Final Neural Population Activity')
    # fig.colorbar(im1, ax=ax1)
    
    # # Final trajectory with all spikes
    # tempx = sNeuronResponse * position_x
    # tempy = sNeuronResponse * position_y
    # tempx = tempx[tempx != 0]
    # tempy = tempy[tempy != 0]
    
    # ax2.plot(position_x, position_y, '-', color='gray', alpha=0.5, linewidth=0.5, label='Full trajectory')
    # if len(tempx) > 0:
    #     ax2.plot(tempx, tempy, 'b.', markersize=2, label=f'Neuron spikes (n={len(tempx)})')
    # ax2.set_title('Complete Single Neuron Response')
    # ax2.set_aspect('equal', adjustable='box')
    # ax2.set_xlabel('Position X')
    # ax2.set_ylabel('Position Y')
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # plt.ioff()  # Turn off interactive mode
    # plt.show()
    
    return spikes


if __name__ == "__main__":
    # Example usage
    spikes = gc_periodic(
        filename='trajectory_data.npz',
        n=100,
        tau=10,
        dt=0.5,
        beta=1.0,
        alphabar=1.0,
        abar=1.5,
        wtphase=10,
        alpha=0.1,
        useSpiking=False
    )
