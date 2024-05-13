import numpy as np
import matplotlib.pyplot as plt

# Fonction dérivée pour l'équation différentielle y' + ky = 0
def f(x, y, k):
    return -k * y

def f_reel(x,y0,k):
    return y0*np.exp(-k*x)

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

# Conditions initiales
x0 = 0
y0 = 1
k = 0.1  # Paramètre k
h = 0.1  # Pas de temps
n = 100  # Nombre d'itérations

# Calcul des solutions à l'aide des deux méthodes
euler_x, euler_y = euler_method(x0, y0, k, h, n)
rk_x, rk_y = runge_kutta_method(x0, y0, k, h, n)
x_values = np.linspace(x0, x0 + n * h, n+1)
y_values = f_reel(x_values, y0, k)

# Tracé des solutions
plt.plot(euler_x, euler_y, label='Euler Method')
plt.plot(x_values,y_values, label='f reel')
plt.plot(rk_x, rk_y, label='Runge-Kutta Method')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution of y\' + ky = 0 (k = {}) with y(0) = {}'.format(k, y0))
plt.legend()
plt.grid(True)
plt.show()

#afficher l'erreur selon la méthode en f(f reelle)