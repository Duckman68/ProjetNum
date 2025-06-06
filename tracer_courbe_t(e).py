import numpy as np
import matplotlib.pyplot as plt

# Paramètres du puits
V0 = 10.0  # profondeur du puits
a = 2.0    # largeur du puits

# Transmission : T(E) avec formule exacte
def T(E):
    k = np.sqrt(2 * E)
    q = np.sqrt(2 * (E + V0))
    fraction = (2 * k * q) / (k**2 + q**2)
    T_val = 1 / (np.cos(q * a)**2 + np.sin(q * a)**2 * fraction**2)
    return T_val

# Énergies à tester
E_vals = np.linspace(0.1, 20, 500)
T_vals = T(E_vals)

# Tracé de la courbe
plt.figure(figsize=(10, 6))
plt.plot(E_vals, T_vals, label="T(E)", color="blue")
plt.title("Transmission T(E) pour un puits entre 0 et a")
plt.xlabel("Énergie E")
plt.ylabel("T(E)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
