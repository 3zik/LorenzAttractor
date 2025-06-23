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
ax.set_title('Lorenz Attractor – Static 3D View')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()