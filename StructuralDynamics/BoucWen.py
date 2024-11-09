"""
@author: LC
"""

##most of the code done by claude LLM

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class BoucWenModel:
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5, n=1, A=1, mass=1, damping=0.1):
        """
        Initialize Bouc-Wen model parameters
        
        Parameters:
        -----------
        alpha : float
            Ratio of post-yield to pre-yield stiffness
        beta, gamma : float
            Parameters controlling hysteresis shape
        n : float
            Parameter controlling transition smoothness
        A : float
            Parameter controlling hysteresis amplitude
        mass : float
            Mass of the system
        damping : float
            Viscous damping coefficient
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.A = A
        self.mass = mass
        self.damping = damping
        
    def system_eqs(self, state, t, forcing):
        """
        System equations for the Bouc-Wen model
        
        Parameters:
        -----------
        state : array-like
            Current state [displacement, velocity, z]
        t : float
            Current time
        forcing : callable
            External forcing function of time
        
        Returns:
        --------
        list
            Derivatives [velocity, acceleration, z_dot]
        """
        x, x_dot, z = state
        
        # Calculate z_dot (hysteretic variable derivative)
        z_dot = x_dot * (self.A - (np.abs(z)**self.n) * 
                        (self.beta * np.sign(x_dot * z) + self.gamma))
        
        # Calculate total restoring force
        restoring_force = self.alpha * x + (1 - self.alpha) * z
        
        # Calculate acceleration
        x_ddot = (forcing(t) - self.damping * x_dot - restoring_force) / self.mass
        
        return [x_dot, x_ddot, z_dot]
    
    def simulate(self, t, x0=None, forcing=None):
        """
        Simulate the Bouc-Wen system
        
        Parameters:
        -----------
        t : array-like
            Time points for simulation
        x0 : list, optional
            Initial conditions [x, x_dot, z]
        forcing : callable, optional
            External forcing function
            
        Returns:
        --------
        tuple
            (time points, displacement, velocity, hysteretic variable)
        """
        if x0 is None:
            x0 = [0, 0, 0]
        
        if forcing is None:
            forcing = lambda t: 0
            
        solution = odeint(self.system_eqs, x0, t, args=(forcing,))
        return t, solution[:, 0], solution[:, 1], solution[:, 2]

# Example usage and visualization
if __name__ == "__main__":
    # Create instance of Bouc-Wen model
    model = BoucWenModel(alpha=0.1, beta=0.5, gamma=0.5, n=1, A=1)
    
    # Time points
    t = np.linspace(0, 100, 1000)
    
    # Define sinusoidal forcing function
    amplitude = 0.5
    frequency = 0.05
    forcing = lambda t: amplitude * np.sin(2 * np.pi * frequency * t)
    forcing = lambda t: amplitude * np.sin(2 * np.pi * frequency * t)+\
                        2*amplitude * np.sin(2 * np.pi *2* frequency * t)+\
                        0.5*amplitude * np.sin(3 * np.pi *2* frequency * t)
    
    
    # t = np.linspace(0, 30, 1000)
    # forcing = lambda t: 1 * np.random.randn()
    
    # Simulate the system
    
    t, x, x_dot, z = model.simulate(t, forcing=forcing)
    restoring_force = model.alpha * x + (1 - model.alpha) * z
    x_ddot = (forcing(t) - model.damping * x_dot - restoring_force) / model.mass
    
    # Create visualization
    # plt.style.use('seaborn')
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Time history of displacement
    ax1 = fig.add_subplot(221)
    ax1.plot(t, x, 'b-', label='Displacement')
    ax1.plot(t, [forcing(ti) for ti in t], 'r--', label='Forcing')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Displacement')
    ax1.set_title('Time History')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Phase portrait
    ax2 = fig.add_subplot(222)
    ax2.plot(x, x_dot, 'g-')
    ax2.set_xlabel('Displacement')
    ax2.set_ylabel('Velocity')
    ax2.set_title('Phase Portrait')
    ax2.grid(True)
    
    # Plot 3: Hysteresis loop
    ax3 = fig.add_subplot(223)
    ax3.plot(x, z, 'b-')
    ax3.set_xlabel('Displacement')
    ax3.set_ylabel('Hysteretic Variable z')
    ax3.set_title('Hysteresis Loop')
    ax3.grid(True)
    
    # Plot 4: 3D phase space
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot(x, x_dot, z)
    ax4.set_xlabel('Displacement')
    ax4.set_ylabel('Velocity')
    ax4.set_zlabel('z')
    ax4.set_title('3D Phase Space')
    
    plt.tight_layout()
    plt.show()
        
        
        # Plotting the results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, 
                                        figsize=(10,6))
    
    # Plot the forcing
    ax1.plot(t, forcing(t), label='p(t)',color="red")
    ax1.set_title('Forcing Function')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Forcing')
    ax1.legend()
    ax1.grid(True)
    
    # Plot displacement, velocity, and acceleration
    ax2.plot(t, x, label='Displacement')
    ax2.plot(t, x_dot, label='Velocity')
    ax2.plot(t, x_ddot, label='Acceleration ')
    ax2.set_title('Displacement, Velocity, and Acceleration')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Values')
    ax2.legend()
    ax2.grid(True)
    
    # Plot the z variable
    ax3.plot(t, z, label='z(t)', color='purple')
    ax3.set_title('Memory variable (z)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('z')
    ax3.legend()
    ax3.grid(True)
    
    # Display the plots
    plt.tight_layout()
    plt.show()
    


    # Additional analysis: Parameter variation study
    alphas = [0.1, 0.3, 0.5, 0.1]
    plt.figure(figsize=(10, 6))
    
    for alpha in alphas:
        model = BoucWenModel(alpha=alpha)
        _, x, _, z = model.simulate(t, forcing=forcing)
        plt.plot(x, z, label=f'α={alpha}')
    
    plt.xlabel('Displacement')
    plt.ylabel('Hysteretic Variable z')
    plt.title('Effect of α on Hysteresis Loops')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
    restoring_force = model.alpha * x + (1 - model.alpha) * z
     
    
    # Create visualization
    # plt.style.use('seaborn')
    plt.figure(figsize=(10, 6))
    
    # plt.plot(x, z, 'b-')
    plt.plot(x, restoring_force, 'b-')
    plt.xlabel('Displacement y(t)')
    plt.ylabel(r'Restoring force $f^{nl}$ (normalized)')
    plt.title('Hysteresis Loops')
    plt.grid(True)
    
    
    plt.tight_layout()
    plt.show()
    
        
        # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with gradient color based on z
    sc = plt.scatter(x, restoring_force, c=z, cmap='viridis', marker='o', edgecolor='none', vmin=-1, vmax=1) 


    # Colorbar to show the scale of z values
    plt.colorbar(sc, label='Memory Variable z(t)')
    
    # Labels and title
    plt.xlabel('Displacement y(t)')
    plt.ylabel(r'Restoring force $f^{nl}$ (normalized)')
    plt.title('Force vs displacement - Hysteresis Loops')
    plt.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()
    
 