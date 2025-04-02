import numpy as np
import matplotlib.pyplot as plt

# Paramètres
hbar = 1.0; m = 1.0
N = 200; L = 20.0; dx = L/N; dt = 0.01

x = np.linspace(0, L, N)
psi = np.exp(-(x-5)**2/2 + 1j*2*x)  # Paquet gaussien initial

# Évolution (méthode très simplifiée)
for _ in range(100):
    psi[1:-1] += 1j*dt*(hbar/(2*m*dx**2)*(psi[2:] - 2*psi[1:-1] + psi[:-2]))
    plt.plot(x, np.abs(psi)**2)
    plt.pause(0.1)
    plt.clf()
