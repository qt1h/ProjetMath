import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Méthode d'Euler pour un système d'équations différentielles du premier ordre
def euler_method_second_order(theta0, omega0, g, l, h, n):
    theta_values = [theta0]
    omega_values = [omega0]
    for i in range(n):
        theta = theta_values[-1]
        omega = omega_values[-1]
        theta_new = theta + h * omega
        omega_new = omega - h * (g / l) * np.sin(theta)
        theta_values.append(theta_new)
        omega_values.append(omega_new)
    return theta_values, omega_values

# Méthode de Runge-Kutta d'ordre 4 pour un système d'équations différentielles du second ordre
def runge_kutta_method_second_order(theta0, omega0, g, l, h, n):
    theta_values = [theta0]
    omega_values = [omega0]
    for i in range(n):
        theta = theta_values[-1]
        omega = omega_values[-1]
        k1_theta = h * omega
        k1_omega = h * (-g / l * np.sin(theta))
        
        k2_theta = h * (omega + k1_omega / 2)
        k2_omega = h * (-g / l * np.sin(theta + k1_theta / 2))
        
        k3_theta = h * (omega + k2_omega / 2)
        k3_omega = h * (-g / l * np.sin(theta + k2_theta / 2))
        
        k4_theta = h * (omega + k3_omega)
        k4_omega = h * (-g / l * np.sin(theta + k3_theta))
        
        theta_new = theta + (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta) / 6
        omega_new = omega + (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega) / 6
        
        theta_values.append(theta_new)
        omega_values.append(omega_new)
    return theta_values, omega_values

# Conditions initiales
theta0 = np.pi / 4.0  # angle initial
omega0 = 0  # vitesse angulaire initiale
g = 9.81  # accélération due à la gravité (m/s^2)
l = 1  # longueur du pendule (m)
h = 0.001  # pas de temps
n = 10000  # nombre d'itérations

# Calcul des solutions à l'aide de la méthode de Runge-Kutta
theta_values, omega_values = runge_kutta_method_second_order(theta0, omega0, g, l, h, n)
# Calcul des solutions à l'aide de la méthode d'Euler
theta_values_euler, omega_values_euler = euler_method_second_order(theta0, omega0, g, l, h, n)

# Tracé de la solution de theta(t)
t_values = [i * h for i in range(n+1)]
plt.plot(t_values, theta_values)
plt.plot(t_values, theta_values_euler)
plt.xlabel('Temps (s)')
plt.ylabel('Angle (rad)')
plt.title('Solution du pendule non linéaire')
plt.grid(True)
plt.show()
