import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

def calcul_etats_stationnaires(V0=200, debut_puit=1.0, largeur_puit=0.2, L=2.0, N=2000, nb_etats_max=10):
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]

    # Potentiel : -V0 dans [debut_puit, debut_puit + largeur_puit], 0 ailleurs
    V = np.zeros(N)
    V[(x >= debut_puit) & (x <= debut_puit + largeur_puit)] = -V0

    # Construction du Hamiltonien tridiagonal
    diag_principale = V + 1.0 / dx**2
    diag_hors = -0.5 / dx**2 * np.ones(N - 1)

    # Calcul des valeurs propres
    energies, psi = eigh_tridiagonal(diag_principale, diag_hors, select='a')
    indices_pos = np.where(energies > 0)[0]
    indices_utiles = indices_pos[:nb_etats_max]
    energies_pos = energies[indices_utiles]
    psi_pos = psi[:, indices_utiles]

    # Normalisation
    for n in range(len(indices_utiles)):
        norme = np.sqrt(np.sum(np.abs(psi_pos[:, n])**2) * dx)
        psi_pos[:, n] /= norme

    return x, energies_pos, psi_pos, V

def tracer_etats(x, energies, psi, V, nb_etats=None):
    if nb_etats is None:
        nb_etats = len(energies)
    plt.figure(figsize=(10, 6))
    for n in range(nb_etats):
        plt.plot(x, psi[:, n] + energies[n], label=f"État {n} : E = {energies[n]:.2f} eV")
    plt.plot(x, V, 'k--', label='Potentiel V(x)')
    plt.title("États stationnaires à énergie positive (mêmes paramètres que le paquet)")
    plt.xlabel("x")
    plt.ylabel("Énergie / ψ(x)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    x, energies, psi, potentiel = calcul_etats_stationnaires()

    print("États stationnaires à énergie positive :")
    for i, e in enumerate(energies):
        print(f"  État {i} : E = {e:.4f} eV")

    tracer_etats(x, energies, psi, potentiel)
