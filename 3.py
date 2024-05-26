import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Fonction dérivée pour l'ED y' + ky = 0
def f(x, y, k):
    return -k * y

# Solution analytique
def f_reel(x, y0, k):
    return y0 * np.exp(-k * x)

# Méthode d'Euler
def euler_method(x0, y0, k, h, n):
    x_values = [x0]
    y_values = [y0]
    for i in range(n):
        x = x_values[-1]
        y = y_values[-1]
        y_new = y + h * f(x, y, k)
        x_values.append(x + h)
        y_values.append(y_new)
    return x_values, y_values

# Méthode de Runge-Kutta d'ordre 4
def runge_kutta_method(x0, y0, k, h, n):
    x_values = [x0]
    y_values = [y0]
    for i in range(n):
        x = x_values[-1]
        y = y_values[-1]
        k1 = h * f(x, y, k)
        k2 = h * f(x + h/2, y + k1/2, k)
        k3 = h * f(x + h/2, y + k2/2, k)
        k4 = h * f(x + h, y + k3, k)
        y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        x_values.append(x + h)
        y_values.append(y_new)
    return x_values, y_values

# Fonction pour calculer l'erreur
def calculate_error(y_approx, y_real):
    return np.abs(np.array(y_approx) - np.array(y_real))

# CI
x0 = 0
y0 = 1
k = 0.1  # Paramètre k
h = 0.1  # Pas de temps
n = 100  # Nombre d'itérations

# Créer les figures et axes
fig_sol, ax_sol = plt.subplots(num='Solutions', figsize=(10, 5))
fig_err, ax_err = plt.subplots(num='Erreurs', figsize=(10, 5))

# Solutions des deux méthodes
euler_x, euler_y = euler_method(x0, y0, k, h, n)
rk_x, rk_y = runge_kutta_method(x0, y0, k, h, n)

# Solution réelle
x_values = np.linspace(x0, x0 + n * h, n + 1)
y_values = f_reel(x_values, y0, k)

# Calcul des erreurs
euler_error = calculate_error(euler_y, f_reel(np.array(euler_x), y0, k))
rk_error = calculate_error(rk_y, f_reel(np.array(rk_x), y0, k))

# Tracé des solutions
ax_sol.plot(euler_x, euler_y, label='Euler Method')
ax_sol.plot(rk_x, rk_y, label='Runge-Kutta Method')
ax_sol.plot(x_values, y_values, label='Solution Analytique', linestyle='dashed')
ax_sol.set_xlabel('x')
ax_sol.set_ylabel('y')
ax_sol.set_title('Solution de y\' + ky = 0')
ax_sol.legend()
ax_sol.grid(True)

# Tracé des erreurs
ax_err.plot(euler_x, euler_error, label='Erreur Euler')
ax_err.plot(rk_x, rk_error, label='Erreur Runge-Kutta')
ax_err.set_xlabel('x')
ax_err.set_ylabel('Erreur')
ax_err.set_title('Erreur des méthodes numériques')
ax_err.legend()
ax_err.grid(True)

# Ajouter les sliders
fig_sol.subplots_adjust(left=0.1, bottom=0.35)
ax_y0 = fig_sol.add_axes([0.2, 0.2, 0.65, 0.03])
y0_slider = Slider(ax=ax_y0, label='y0', valmin=-100, valmax=100, valinit=y0)

ax_k = fig_sol.add_axes([0.2, 0.15, 0.65, 0.03])
k_slider = Slider(ax=ax_k, label="k", valmin=-10, valmax=10, valinit=k)

# Fonction pour mettre à jour les graphiques
def update(val):
    k = k_slider.val
    y0 = y0_slider.val
    euler_x, euler_y = euler_method(x0, y0, k, h, n)
    rk_x, rk_y = runge_kutta_method(x0, y0, k, h, n)
    y_values = f_reel(x_values, y0, k)
    
    euler_error = calculate_error(euler_y, f_reel(np.array(euler_x), y0, k))
    rk_error = calculate_error(rk_y, f_reel(np.array(rk_x), y0, k))
    
    ax_sol.clear()
    ax_sol.plot(euler_x, euler_y, label='Euler Method')
    ax_sol.plot(rk_x, rk_y, label='Runge-Kutta Method')
    ax_sol.plot(x_values, y_values, label='Solution Analytique', linestyle='dashed')
    ax_sol.set_xlabel('x')
    ax_sol.set_ylabel('y')
    ax_sol.set_title('Solution de y\' + ky = 0')
    ax_sol.legend()
    ax_sol.grid(True)

    ax_err.clear()
    ax_err.plot(euler_x, euler_error, label='Erreur Euler')
    ax_err.plot(rk_x, rk_error, label='Erreur Runge-Kutta')
    ax_err.set_xlabel('x')
    ax_err.set_ylabel('Erreur')
    ax_err.set_title('Erreur des méthodes numériques')
    ax_err.legend()
    ax_err.grid(True)
    
    fig_sol.canvas.draw_idle()
    fig_err.canvas.draw_idle()

# Enregistrer la fonction de mise à jour avec les sliders
y0_slider.on_changed(update)
k_slider.on_changed(update)

# Ajouter un bouton pour réinitialiser les sliders
resetax = fig_sol.add_axes([0.8, 0.05, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    k_slider.reset()
    y0_slider.reset()
button.on_clicked(reset)

plt.show()
