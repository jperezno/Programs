import numpy as np
import matplotlib.pyplot as plt

# ---- Error model functions ----
def hybA(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z):
    A_error=(1 - epsilon_g)**2*(2*eta*(1 - eta)*(A*C + B*D) + (A**2 + B**2)*(eta**2 + (1 - eta)**2) + (-epsilon_g**2 + 2*epsilon_g)*(2*A*B*p_z + 2*C*D*p_y + p_x*(C**2 + D**2)))
    return A_error

def hybC(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z):
    C_error=(1 - epsilon_g)**2*(2*eta*(1 - eta)*(A*C + B*D) + (C**2 + D**2)*(eta**2 + (1 - eta)**2) + (-epsilon_g**2 + 2*epsilon_g)*(p_x*(A*C + B*D) + p_y*(A*D + B*C) + p_z*(A*D + B*C)))
    return C_error

def hybB(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z):
    B_error=(1 - epsilon_g)**2*(2*A*B*(eta**2 + (1 - eta)**2) + 2*eta*(1 - eta)*(A*D + B*C) + (-epsilon_g**2 + 2*epsilon_g)*(p_x*(A*D + B*C) + p_y*(A*C + B*D) + p_z*(A*C + B*D)))
    return B_error
def hybD(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z):
    D_error=(1 - epsilon_g)**2*(2*C*D*(eta**2 + (1 - eta)**2) + 2*eta*(1 - eta)*(A*D + B*C) + (-epsilon_g**2 + 2*epsilon_g)*(2*C*D*p_x + p_y*(C**2 + D**2) + p_z*(A**2 + B**2)))
    return D_error

def acc_prob(A,B,C,D,eta):
    P=2*eta*(1 - eta)*(2*A + 2*B)*(C + D) + (eta**2 + (1 - eta)**2)*((A + B)**2 + (C + D)**2)
    return P
# ---- Parameters ----
# eta = 1 - 1e-4    # readout efficiency
eta = 0.99
# eps_g = 5e-4      # gate error
epsilon_g = 0.01
p_x, p_y, p_z = 0.25, 0.25, 0.5  # Pauli errors

# ---- Sweep initial fidelity ----
F_init = np.linspace(0,1,200)

# Store results for multiple rounds
rounds = 5   # 0,1,2,3,4
F_curves = {r: [] for r in range(rounds)}

for F0 in F_init:
    A = F0
    B = C = D = (1-F0)/3
    p = acc_prob(A, B, C, D, eta)

    # Round 0: just initial fidelity
    F_curves[0].append(A)

    # Iterate purification rounds
    for r in range(1, rounds):
        A_new = hybA(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        B_new = hybB(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        C_new = hybC(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        D_new = hybD(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        A, B, C, D = A_new, D_new, C_new, B_new  # Flip B and D
        p = acc_prob(A, B, C, D, eta)

        F_curves[r].append(A)


# ---- Plot ----
plt.figure(figsize=(8,5))
colors = ["b","orange","g","r","purple"]
for r in range(rounds):
    plt.plot(F_init, F_curves[r], label=f"{r}", linestyle="--" if r>0 else "-", color=colors[r])

plt.xlabel(r"$A$")
plt.ylabel(r"$A'$")
plt.title("Purification with error model")
plt.legend(title="Rounds")
plt.grid(True)
plt.show()