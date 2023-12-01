import numpy as np
import matplotlib.pyplot as plt

def heun_method(f, y0, t0, tn, h):
    num_steps = int((tn - t0) / h) + 1
    t_values = np.linspace(t0, tn, num_steps)
    y_values = np.zeros(num_steps)
    y_values[0] = y0

    for i in range(1, num_steps):   
        k1 = f(t_values[i - 1], y_values[i - 1])
        k2 = f(t_values[i - 1] + h, y_values[i - 1] + h * k1)
        y_values[i] = y_values[i - 1] + 0.5 * h * (k1 + k2)

    return t_values, y_values

def milne_simpson_method(f, y0, t0, tn, h):
    num_steps = int((tn - t0) / h) + 1
    t_values = np.linspace(t0, tn, num_steps)
    y_values = np.zeros(num_steps)
    y_values[0] = y0

    for i in range(0, num_steps - 3, 2):
        k1 = f(t_values[i], y_values[i])
        k2 = f(t_values[i + 1], y_values[i + 1])
        k3 = f(t_values[i + 2], y_values[i + 2])

        y_values[i + 2] = y_values[i] + (h / 3) * (k1 + 4 * k2 + k3)
    
    return t_values, y_values


# Given ODE: y' = -a*b*t*y^2
def example_ode(t, y, a, b):
    return -a * b * t * y**2

# Initial condition: y(a) = b
a = 1
b = 1

t0 = a
tn = a + 3
h = 0.1

# Heun's Method
t_values_heun, y_values_heun = heun_method(lambda t, y: example_ode(t, y, -a, b), b, t0, tn, h)

# Milne-Simpson Method
t_values_ms, y_values_ms = milne_simpson_method(lambda t, y: example_ode(t, y, -a, b), b, t0, tn, h)

# Plot the results
plt.plot(t_values_heun, y_values_heun, label='Heun\'s Method')
plt.plot(t_values_ms, y_values_ms, label='Milne-Simpson Method')
plt.xlabel('Time')
plt.ylabel('Solution y(t)')
plt.legend()
plt.show()
