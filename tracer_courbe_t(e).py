import numpy as np
import matplotlib.pyplot as plt

# Paramètres du puits
V0 = 10.0  # profondeur du puits
a = 2.0    # largeur du puits

# Constantes physiques (unités réduites avec ħ²/2m = 1)
def T(E):
    kappa = np.sqrt(2 * (E + V0))  # kappa = √(2(E + V0)) en unités réduites
    num = (V0 * np.sin(kappa * a))**2
    denom = 4 * E * (E + V0)
    return 1 / (1 + num / denom)

# Énergies à tester
E_vals = np.linspace(0.1, 20, 500)
T_vals = T(E_vals)

# Tracé de la courbe
plt.plot(E_vals, T_vals, label="T(E)")
plt.title("Transmission en fonction de l'énergie")
plt.xlabel("Énergie E")
plt.ylabel("T(E)")
plt.grid(True)
plt.legend()
plt.show()
