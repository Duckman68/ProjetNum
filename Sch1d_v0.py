import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# Paramètres de la simulation
dx = 0.01  # Pas spatial
dt = 0.001  # Pas temporel
nx = 200  # Nombre de points dans l'espace
nt = 500  # Nombre d'étapes temporelles
x_array = np.linspace(0, (nx - 1) * dx, nx)  # Grille spatiale

# Paramètres physiques (en unités normalisées)
hbar = 1  # Constante de Planck réduite (normalisée)
m = 1  # Masse de la particule
V_potential = np.zeros(nx)  # Potentiel nul (V(x) = 0)

# Paquet d'ondes gaussien
xc = 0.5  # Position centrale du paquet d'ondes
sigma = 0.05  # Largeur du paquet d'ondes
k = 10  # Nombre d'onde (lié à l'énergie)
normalisation = 1 / (math.sqrt(sigma * np.sqrt(np.pi)))  # Normalisation
wp_gauss = normalisation * np.exp(1j * k * x_array - (x_array - xc) ** 2 / (2 * sigma ** 2))

# Fonction d'onde initiale (partie réelle et imaginaire)
wp_re = np.real(wp_gauss)
wp_im = np.imag(wp_gauss)

# Densité de probabilité initiale
density = np.zeros((nt, nx))
density[0, :] = np.abs(wp_gauss) ** 2

# Discrétisation de l'équation de Schrödinger avec différences finies
def update_wavefunction(wp_re, wp_im, V_potential, dx, dt, nx):
    # Dérivées secondes en espace (différences finies centrées)
    d2wp_re = np.roll(wp_re, -1) - 2 * wp_re + np.roll(wp_re, 1)
    d2wp_im = np.roll(wp_im, -1) - 2 * wp_im + np.roll(wp_im, 1)
    
    # Mise à jour explicite de la fonction d'onde
    wp_re_new = wp_re - (dt / (2 * m)) * (d2wp_im - V_potential * wp_im)
    wp_im_new = wp_im + (dt / (2 * m)) * (d2wp_re + V_potential * wp_re)
    
    return wp_re_new, wp_im_new

# Initialisation de la figure
fig = plt.figure()
line, = plt.plot([], [])
plt.ylim(0, 1)
plt.xlim(0, 2)
plt.plot(x_array, V_potential, label="Potentiel")
plt.title("Propagation d'un paquet d'ondes")
plt.xlabel("x")
plt.ylabel("Densité de probabilité")

# Fonction d'initialisation de l'animation
def init():
    line.set_data([], [])
    return line,

# Fonction d'animation
def animate(j):
    global wp_re, wp_im
    
    # Mise à jour de la fonction d'onde à chaque pas de temps
    wp_re, wp_im = update_wavefunction(wp_re, wp_im, V_potential, dx, dt, nx)
    
    # Calcul de la densité de probabilité
    density[j, :] = np.abs(wp_re + 1j * wp_im) ** 2
    
    # Mise à jour de l'animation
    line.set_data(x_array, density[j, :])
    return line,

# Création de l'animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nt, blit=False, interval=50, repeat=False)

# Affichage de l'animation
plt.show()

