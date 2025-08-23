import math
import numpy as np
import matplotlib.pyplot as plt

#######Swaps#######
def apply_swap_fidelity(F, eta):
    term = ((4 * F - 1) / 3) ** 2
    return (1 + ((4 * eta ** 2 - 1) * term)) / 4

def simulate_swap_decay(F_start, eta, num_swaps):
    F = F_start
    swap_fidelity = [F] #List for values
    print("=== Entanglement swapping ===")
    for swaps in range(1, num_swaps + 1):
        F = apply_swap_fidelity(F, eta)
        swap_fidelity.append(F)
        print(f"Swap {swaps}: F = {F:.6f}")
    return F, swap_fidelity

######Purifying######

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

def simulate_purification(source_fid, F_target, eta, epsilon_g,n_swaps, max_iter=10):
    print("\n=== Purification ===")
    p_z = 0.5
    p_x = (1 - p_z) / 2
    p_y = (1 - p_z) / 2
    A = source_fid
    B = C = D = (1 - A) / 3
    # B = (1 - A) / 12
    # C = 4*(1 - A) / 6
    # D = (1 - A) / 12
    p = acc_prob(A, B, C, D, eta)
    iterations = 0
    p_prod = 1

    # Let's save some relevant values
    A_list = [A]
    p_list = [p]

    while A < F_target and iterations < max_iter:
        A_new = hybA(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        B_new = hybB(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        C_new = hybC(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        D_new = hybD(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        A, B, C, D = A_new, D_new, C_new, B_new  # Flip B and D

        p = acc_prob(A, B, C, D, eta)
        p_prod *= p
        iterations += 1
        A_list.append(A)
        p_list.append(p)
        print(f"Purification {iterations}: A = {A:.6f}, P = {p:.6f}")

    if A >= F_target:
        cost = 1 + math.log(((2 ** iterations) / p_prod),2**n_swaps)
        return cost, A, iterations, A_list, p_list
    else:
        return None, A, iterations, A_list, p_list # Failed to reach target

def find_optimal_swap_purify(F_0, eta, epsilon_g, max_swaps=20, max_purify_steps=10):
    print(f"\n=== Optimizing Setup for F0 = {F_0}, eta = {eta}, epsilon_g = {epsilon_g} ===")
    results = []
    for n_swaps in range(1, max_swaps + 1):
        F_s, _ = simulate_swap_decay(F_0, eta, n_swaps)
        cost, F_final, purify_steps, _, _ = simulate_purification(F_s, F_0, eta, epsilon_g, max_purify_steps)

        if F_final >= F_0 and purify_steps <= max_purify_steps:
            results.append((n_swaps, purify_steps, cost))
        else:
            break  # Stop when purification fails

    if results:
        best = results[-1]
        print("\n Optimal Found:")
        print(f"  Max Swaps = {best[0]}")
        print(f"  Required Purification Steps = {best[1]}")
        print(f"  Total Cost = {best[2]:.4f}")
    else:
        print("\n No viable setup found for current parameters.")

#Simulation

def run_full_simulation(F_0, eta, epsilon_g, num_swaps, max_purify_steps=10):
    print(f"\n=== Simulating for F0 = {F_0}, Er = {1-eta}, Eg = {epsilon_g} ===")
    # Step 1: Swap fidelity decay
    F_s, swap_fidelity = simulate_swap_decay(F_0, eta, num_swaps)

    # Step 2: Purification
    cost, F_final, n_purify, A_list, p_list = simulate_purification(F_s, F_0, eta, epsilon_g,num_swaps ,max_purify_steps)
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    # fig.suptitle(f"Swap and Purification Simulation\nF₀ = {F_0}, eta = {eta}, epsilon_g = {epsilon_g}, Swaps = {num_swaps}", fontsize=14)
    fig.suptitle(f"Successive entanglement swapping", fontsize=14)

    # Plot swap fidelity
    axs[0].plot(range(len(swap_fidelity)), swap_fidelity, marker='o', color='blue', label='Swap Fidelity')
    axs[0].axhline(F_0, color='gray', linestyle='--', label='Target Fidelity')
    axs[0].set_title("Fidelity After Swaps")
    axs[0].set_xlabel("Swap Step")
    axs[0].set_ylabel("Fidelity")
    axs[0].legend()
    axs[0].grid(True)

    # Plot purification fidelity
    axs[1].plot(range(len(A_list)), A_list, marker='o', color='green', label='Purified Fidelity A')
    axs[1].plot(range(len(p_list)), p_list, marker='x', color='red', linestyle='--', label='Acceptance Probability P')
    axs[1].axhline(F_0, color='gray', linestyle='--')
    axs[1].set_title("Purification Process")
    axs[1].set_xlabel("Purification Step")
    axs[1].set_ylabel("Value")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    print("\n=== Summary ===")
    print(f"Initial Fidelity (F_0): {F_0}")
    print(f"Fidelity after {num_swaps} swaps (F_s): {F_s}")
    print(f"Fidelity after purification: {F_final}")
    if cost is not None:
        print(f"Purification steps needed: {n_purify}")
        print(f"Effective cost (log{2**num_swaps} scale): {cost:.4f}")
    else:
        print("Fail")

# Here we modify parameters
# # ============== Superconducting ================ 1
# epsilon_g=2.5e-3
# eta=0.99400
# F_0 = 0.9879
# # # =================== SiV ======================= 2
epsilon_g=5e-4
eta=1-1e-4
F_0 = 0.9979
# # # #=================== NV ======================== 3
# epsilon_g=3.5e-4
# eta=0.99960
# F_0 = 0.9982
# # ================== TrIon ====================== 2
# epsilon_g=5e-4
# eta= 0.999999
# F_0 = 0.9979
# # ================NeutralAtoms ================== 1
# epsilon_g=2.5e-3
# eta=0.99600
# F_0 = 0.9886
####################IDEAL
# epsilon_g=0
# eta=1
# F_0 = 0.99999
####################IDEAL
# epsilon_g=0
# eta=1
# F_0 = 1
run_full_simulation(F_0, eta, epsilon_g, num_swaps=2)
# find_optimal_swap_purify(F_0, eta, epsilon_g, max_swaps=10, max_purify_steps=2)