import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Fonction pour la méthode d'Euler (second ordre)
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

# Fonction pour la méthode de Runge-Kutta (second ordre)
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

# CI
theta0 = np.pi / 4.0  # angle initial
omega0 = 0  # vitesse angulaire initiale
g = 9.81  # accélération due à la gravité (m/s^2)
l = 1  # longueur pendule (m)
h = 0.01  # pas de temps
n = 1000  # nombre d'itérations

# Solution méthode d'Euler
theta_values_euler, omega_values_euler = euler_method_second_order(theta0, omega0, g, l, h, n)
# Solution méthode de Runge-Kutta
theta_values_rk, omega_values_rk = runge_kutta_method_second_order(theta0, omega0, g, l, h, n)

# Tracé de la solution de theta(t)
t_values = np.linspace(0, n*h, n+1)

# Création des figures et axes
fig_sol, ax_sol = plt.subplots(num='Solutions', figsize=(10, 5))
fig_err, ax_err = plt.subplots(num='Erreurs', figsize=(10, 5))

# Tracé initial des solutions
line_euler, = ax_sol.plot(t_values, theta_values_euler, label='Euler Method')
line_rk, = ax_sol.plot(t_values, theta_values_rk, label='Runge-Kutta Method')
ax_sol.set_xlabel('Temps (s)')
ax_sol.set_ylabel('Angle (rad)')
ax_sol.set_title('Solution du pendule non linéaire')
ax_sol.legend()
ax_sol.grid(True)

# Tracé initial des erreurs
erreur_euler = np.array(theta_values_rk) - np.array(theta_values_euler)
erreur_rk = np.zeros_like(erreur_euler)  # En absence de solution réelle, erreur_rk est supposée nulle
line_err_euler, = ax_err.plot(t_values, erreur_euler, label='Erreur Euler')
line_err_rk, = ax_err.plot(t_values, erreur_rk, label='Erreur Runge-Kutta')
ax_err.set_xlabel('Temps (s)')
ax_err.set_ylabel('Erreur (rad)')
ax_err.set_title('Erreurs des méthodes numériques')
ax_err.legend()
ax_err.grid(True)

# Création des sliders sous le graphique des solutions
axcolor = 'lightgoldenrodyellow'
fig_sol.subplots_adjust(bottom=0.25)
ax_theta0 = fig_sol.add_axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_omega0 = fig_sol.add_axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

theta0_slider = Slider(ax_theta0, 'theta0', -2*np.pi, 2*np.pi, valinit=theta0)
omega0_slider = Slider(ax_omega0, 'omega0', -10, 10, valinit=omega0)

# Fonction de mise à jour des tracés
def update(val):
    theta0 = theta0_slider.val
    omega0 = omega0_slider.val
    theta_values_euler, omega_values_euler = euler_method_second_order(theta0, omega0, g, l, h, n)
    theta_values_rk, omega_values_rk = runge_kutta_method_second_order(theta0, omega0, g, l, h, n)
    line_euler.set_ydata(theta_values_euler)
    line_rk.set_ydata(theta_values_rk)
    erreur_euler = np.array(theta_values_rk) - np.array(theta_values_euler)
    erreur_rk = np.zeros_like(erreur_euler)  # En absence de solution réelle, erreur_rk est supposée nulle
    line_err_euler.set_ydata(erreur_euler)
    line_err_rk.set_ydata(erreur_rk)
    fig_sol.canvas.draw_idle()
    fig_err.canvas.draw_idle()

# Liaison des sliders à la fonction de mise à jour
theta0_slider.on_changed(update)
omega0_slider.on_changed(update)

# Création du bouton de réinitialisation
resetax = plt.axes([0.8, 0.01, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    theta0_slider.reset()
    omega0_slider.reset()
button.on_clicked(reset)

plt.show()
