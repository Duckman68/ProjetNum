import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg
import time

start_time = time.time()

# === Initialisation de la propagation d’un paquet d’onde ===

def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(o, final_densite[j,:])
    return line,

# Paramètres physiques et numériques
dt = 1E-7
dx = 0.001
nx = int(1/dx)*2
nt = 90000
nd = int(nt/1000) + 1
s = dt / (dx**2)
xc = 0.6
sigma = 0.05
A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))

# Paramètres du potentiel et énergie du paquet
v0 = -4000
e = 5  # E/V0
E_paquet = e * abs(v0)
k = math.sqrt(2 * abs(E_paquet))

# Grille spatiale et potentiel
o = np.linspace(0, (nx - 1) * dx, nx)
V = np.zeros(nx)
V[(o >= 0.8) & (o <= 0.9)] = v0  # puits de potentiel

# Paquet d’onde initial
cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * sigma ** 2))
densite = np.zeros((nt, nx))
densite[0,:] = np.abs(cpt) ** 2
final_densite = np.zeros((nd, nx))

re = np.real(cpt)
im = np.imag(cpt)
b = np.zeros(nx)

# === Propagation du paquet d’onde dans le temps ===
it = 0
for i in range(1, nt):
    if i % 2 != 0:
        b[1:-1] = im[1:-1]
        im[1:-1] = im[1:-1] + s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        densite[i,1:-1] = re[1:-1]**2 + im[1:-1]*b[1:-1]
    else:
        re[1:-1] = re[1:-1] - s * (im[2:] + im[:-2]) + 2 * im[1:-1] * (s + V[1:-1] * dt)

    if ((i - 1) % 1000 == 0):
        final_densite[it,:] = densite[i,:]
        it += 1

# === Construction du Hamiltonien H = T + V ===
hbar = 1
m = 1
coeff = hbar**2 / (2 * m * dx**2)

# Matrice tridiagonale de l’énergie cinétique
diagonale = np.full(nx, 2 * coeff)
off_diagonale = np.full(nx - 1, -coeff)
T = np.diag(diagonale) + np.diag(off_diagonale, k=1) + np.diag(off_diagonale, k=-1)

# Matrice du potentiel
V_diag = np.diag(V)
H = T + V_diag

# === Résolution du problème stationnaire Hψ = Eψ ===
energies, vecteurs_propres = scipy.linalg.eigh(H)

# === Affichage animation du paquet d’onde ===
plot_title = "Propagation du paquet (E/V₀ = " + str(e) + ")"
fig = plt.figure()
line, = plt.plot([], [])
plt.ylim(-6, 10)
plt.xlim(0, 2)
plt.plot(o, V, label="Potentiel")
plt.title(plot_title)
plt.xlabel("x")
plt.ylabel("Densité de probabilité")
plt.legend()

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nd, blit=False, interval=100, repeat=False)
plt.show()

# === Affichage des états stationnaires pertinents ===
# ➤ On affiche uniquement les états liés (énergie < 0)
plt.figure(figsize=(10, 6))

# Affichage des états stationnaires ψ_n(x)
for i in range(len(energies)):
    if energies[i] < 0:
        psi = vecteurs_propres[:, i]
        densite_psi = np.abs(psi) ** 2
        densite_psi /= np.max(densite_psi)  # Normalisation pour affichage
        plt.plot(o, densite_psi, label=f"État {i+1} (E={energies[i]:.2f})")

# ➤ On trace aussi la densité d’énergie du paquet
plt.axhline(0, color='gray')
plt.axvline(x=xc, linestyle='--', color='gray', label="Centre paquet initial")
plt.axhline(y=abs(E_paquet)/abs(v0), linestyle=':', color='red', label=f"E paquet = {E_paquet:.2f}")

# Potentiel pour repère visuel
plt.plot(o, V / abs(v0), 'k--', label="Potentiel (échelle relative)")

plt.title("États stationnaires (ψ_n) et énergie du paquet incident")
plt.xlabel("x")
plt.ylabel("|ψ(x)|² (normalisé)")
plt.legend()
plt.grid()
plt.show()
