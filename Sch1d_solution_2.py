import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import scipy.linalg


start_time = time.time()

def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(o, final_densite[j,:]) #Crée un graphique pour chaque densite sauvegarde
    return line,

dt=1E-7
dx=0.001
nx=int(1/dx)*2
nt=90000 # En fonction du potentiel il faut modifier ce parametre car sur certaines animations la particule atteins les bords
nd=int(nt/1000)+1#nombre d image dans notre animation
n_frame = nd
s=dt/(dx**2)
xc=0.6
sigma=0.05
A=1/(math.sqrt(sigma*math.sqrt(math.pi)))
v0=-1000
e=5#Valeur du rapport E/V0
E=e*v0
k=math.sqrt(2*abs(E))


o=np.zeros(nx)
V=np.zeros(nx)

# Initialisation des tableaux
o = np.linspace(0, (nx - 1) * dx, nx)
V = np.zeros(nx)
V[(o >= 0.8) & (o<=0.9)] = v0  # Potentiel

cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * (sigma ** 2)))
densite=np.zeros((nt,nx))
densite[0,:] = np.absolute(cpt[:]) ** 2
final_densite=np.zeros((n_frame,nx))
re=np.zeros(nx)
re[:]=np.real(cpt[:])

b=np.zeros(nx)

im=np.zeros(nx)
im[:]=np.imag(cpt[:])

it=0
for i in range(1, nt):
    if i % 2 != 0:
        b[1:-1]=im[1:-1]
        im[1:-1] = im[1:-1] + s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        densite[i,1:-1] = re[1:-1]*re[1:-1] + im[1:-1]*b[1:-1]
    else:
        re[1:-1] = re[1:-1] - s * (im[2:] + im[:-2]) + 2 * im[1:-1] * (s + V[1:-1] * dt)

for i in range(1,nt):
    if((i-1)%1000==0):
        it+=1
        final_densite[it][:]=densite[i][:]

# Construction du Hamiltonien H = T + V
# T = -ħ²/2m ∂²/∂x² → discrétisé en matrice tridiagonale
hbar = 1
m = 1
coeff = hbar**2 / (2 * m * dx**2)

# Matrice cinétique (tridiagonale)
diagonale = np.full(nx, 2 * coeff)
off_diagonale = np.full(nx - 1, -coeff)
T = np.diag(diagonale) + np.diag(off_diagonale, k=1) + np.diag(off_diagonale, k=-1)

# Matrice du potentiel
V_diag = np.diag(V)
H = T + V_diag

# Résolution du problème aux valeurs propres
energies, vecteurs_propres = scipy.linalg.eigh(H)

plot_title = "Marche Ascendante avec E/Vo="+str(e)

fig = plt.figure() # initialise la figure principale
line, = plt.plot([], [])
plt.ylim(-6,10)
plt.xlim(0,2)
plt.plot(o,V,label="Potentiel")
plt.title(plot_title)
plt.xlabel("x")
plt.ylabel("Densité de probabilité de présence")
plt.legend() #Permet de faire apparaitre la legende

ani = animation.FuncAnimation(fig,animate,init_func=init, frames=nd, blit=False, interval=100, repeat=False)

plt.show()

# Affichage des premiers états stationnaires
n_etats = 3  # Nombre d'états stationnaires à afficher
plt.figure(figsize=(10, 6))
for i in range(n_etats):
    psi = vecteurs_propres[:, i]
    densite_psi = np.abs(psi)**2
    densite_psi /= np.max(densite_psi)  # Normalisation pour affichage
    plt.plot(o, densite_psi, label=f"État {i+1} (E={energies[i]:.2f})")

plt.plot(o, V / abs(v0), 'k--', label="Potentiel (échelle relative)")
plt.title("États stationnaires (densités de probabilité)")
plt.xlabel("x")
plt.ylabel("|ψ(x)|²")
plt.legend()
plt.grid()
plt.show()



