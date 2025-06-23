import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

'''
Lorenz Attractor Equations:

sigma: Prandtl number (ratio of momentum diffusivity to thermal diffusivity)
rho: Rayleigh number (measures temperature difference driving convection)
beta: A geometric factor
Standard values are: sigma=10, beta=8/3, rho=28
'''

def lorenz(t, state, sigma=10, beta=8/3, rho=28): # Standard values for sigma, beta, rho
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


t_range = (0, 40)

# Create an array of time points where the solution will be evaluated.
# np.linspace(start, end, num_points) generates 5000 evenly spaced points between 0 and 40.
t_eval = np.linspace(*t_range, 5000)

initial_state = [5.0, 1.0, 1.0]

# Solve the Lorenz system using the Runge-Kutta method of order 5(4), known as 'RK45'.
# - lorenz: the function defining our differential equations
# - t_range: the time interval for integration (from 0 to 40)
# - initial_state: the initial values of x, y, z
# - t_eval: the specific time points at which to store the solution
sol = solve_ivp(lorenz, t_range, initial_state, t_eval=t_eval, method='RK45')

x, y, z = sol.y

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

ax.plot(x, y, z, lw=0.7, color='purple')
ax.set_title('Lorenz Attractor - Static 3D View')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()



# DEFINE Initial Conditions and Simulate All

# Multiple close initial conditions
initial_conditions = [
    [5., 5., 5.],
    [5.1, 5.1, 5.2],
    [4.8, 5., 4.8]
]

# Longer simulation
t_span = (0, 60)
t_eval = np.linspace(*t_span, 3000)

def integrate_lorenz(initial_state, t_span, t_eval):
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')
    return sol.y

trajectories = [integrate_lorenz(init, t_span, t_eval) for init in initial_conditions]

# ANMIATION

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Increase animation size limit
plt.rcParams['animation.embed_limit'] = 50  # in MB

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')
ax.set_xlim([-25, 25])
ax.set_ylim([-35, 35])
ax.set_zlim([0, 50])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor: Animation of 3 Initial Conditions')

colors = ['blue', 'red', 'green']
lines = [ax.plot([], [], [], lw=1, color=colors[i], markersize=0.5)[0] for i in range(3)]

# Animation update function
def update(num):
    for i in range(3):
        lines[i].set_data(trajectories[i][0][:num], trajectories[i][1][:num])
        lines[i].set_3d_properties(trajectories[i][2][:num])
    return lines

# Run animation
anim = FuncAnimation(fig, update, frames=1000, interval=30, blit=True)

# Display
HTML(anim.to_jshtml())
