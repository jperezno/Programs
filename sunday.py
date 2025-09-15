import math
import numpy as np
import matplotlib.pyplot as plt

####### Swaps #######
def apply_swap_fidelity(F, eta):
    term = ((4 * F - 1) / 3) ** 2
    return (1 + ((4 * eta ** 2 - 1) * term)) / 4

def simulate_swap_decay(F_start, eta, num_swaps):
    F = F_start
    swap_fidelity = [F]  # List for values
    for swaps in range(1, num_swaps + 1):
        F = apply_swap_fidelity(F, eta)
        swap_fidelity.append(F)
    return F, swap_fidelity

###### Purifying ######
def hybA(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z):
    return (1 - epsilon_g)**2*(2*eta*(1 - eta)*(A*C + B*D) + (A**2 + B**2)*(eta**2 + (1 - eta)**2) +
            (-epsilon_g**2 + 2*epsilon_g)*(2*A*B*p_z + 2*C*D*p_y + p_x*(C**2 + D**2)))

def hybC(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z):
    return (1 - epsilon_g)**2*(2*eta*(1 - eta)*(A*C + B*D) + (C**2 + D**2)*(eta**2 + (1 - eta)**2) +
            (-epsilon_g**2 + 2*epsilon_g)*(p_x*(A*C + B*D) + p_y*(A*D + B*C) + p_z*(A*D + B*C)))

def hybB(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z):
    return (1 - epsilon_g)**2*(2*A*B*(eta**2 + (1 - eta)**2) + 2*eta*(1 - eta)*(A*D + B*C) +
            (-epsilon_g**2 + 2*epsilon_g)*(p_x*(A*D + B*C) + p_y*(A*C + B*D) + p_z*(A*C + B*D)))

def hybD(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z):
    return (1 - epsilon_g)**2*(2*C*D*(eta**2 + (1 - eta)**2) + 2*eta*(1 - eta)*(A*D + B*C) +
            (-epsilon_g**2 + 2*epsilon_g)*(2*C*D*p_x + p_y*(C**2 + D**2) + p_z*(A**2 + B**2)))

def acc_prob(A,B,C,D,eta):
    return 2*eta*(1 - eta)*(2*A + 2*B)*(C + D) + (eta**2 + (1 - eta)**2)*((A + B)**2 + (C + D)**2)

def simulate_purification(source_fid, F_target, eta, epsilon_g,n_swaps, max_iter=10):
    p_z = 0.5
    p_x = (1 - p_z) / 2
    p_y = (1 - p_z) / 2
    A = source_fid
    B = C = D = (1 - A) / 3
    p = acc_prob(A, B, C, D, eta)
    iterations = 0
    p_prod = 1

    while A < F_target and iterations < max_iter:
        A_new = hybA(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        B_new = hybB(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        C_new = hybC(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        D_new = hybD(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        A, B, C, D = A_new, D_new, C_new, B_new  # Flip B and D

        p = acc_prob(A, B, C, D, eta)
        p_prod *= p
        iterations += 1

    if A >= F_target:
        cost = 1 + math.log(((2 ** iterations) / p_prod), 2**n_swaps)
        return cost, A, iterations
    else:
        return None, A, iterations  # Failed to reach target

def find_optimal_swap_purify(F_0, eta, epsilon_g, max_swaps=20, max_purify_steps=10):
    results = []
    for n_swaps in range(1, max_swaps + 1):
        F_s, _ = simulate_swap_decay(F_0, eta, n_swaps)
        cost, F_final, purify_steps = simulate_purification(F_s, F_0, eta, epsilon_g, max_purify_steps)

        if F_final >= F_0 and purify_steps <= max_purify_steps:
            results.append((n_swaps, purify_steps, cost))
        else:
            break  # Stop when purification fails

    if results:
        return results[-1]  # Best = last successful (max swaps)
    else:
        return None

def find_optimal_over_F(F_min=0.980, F_max=1.0, step=0.0005, eta=0.99600, epsilon_g=2.5e-3, max_swaps=10, max_purify_steps=2):
    F_values = np.arange(F_min, F_max + step, step)
    results = []

    for F_0 in F_values:
        best = find_optimal_swap_purify(F_0, eta, epsilon_g, max_swaps=max_swaps, max_purify_steps=max_purify_steps)
        if best:
            n_swaps, purify_steps, cost = best
            results.append((F_0, n_swaps, purify_steps, cost))
        else:
            break  # stop when program breaks

    if not results:
        print("\nNo viable setups found in sweep.")
        return None, []

    # Find the entry with maximum N_swaps
    best_entry = max(results, key=lambda x: x[1])
    print("\n=== Global Optimum Found ===")
    print(f"F_0 = {best_entry[0]:.6f}")
    print(f"Max Swaps = {best_entry[1]}")
    print(f"Purification Steps = {best_entry[2]}")
    print(f"Total Cost = {best_entry[3]:.4f}")

    return best_entry, results

####### Run sweep and plot #######
epsilon_g=3.5e-4
eta=0.99960
# F_0 = 0.99673
best_entry, all_results = find_optimal_over_F(
    F_min=0.82, F_max=1.0, step=0.001,
    eta=eta, epsilon_g=epsilon_g,
    max_swaps=10, max_purify_steps=2
)

if all_results:
    F_vals = [r[0] for r in all_results]
    N_swaps = [r[1] for r in all_results]
    purify_steps = [r[2] for r in all_results]
    costs = [r[3] for r in all_results]

    # plt.figure()
    # plt.plot(F_vals, N_swaps, marker="o")
    # plt.xlabel("F_0")
    # plt.ylabel("Max Swaps")
    # plt.title("Optimal Swaps vs F_0")
    # plt.grid(True)
    # plt.show()

    # plt.figure()
    # plt.plot(F_vals, purify_steps, marker="s", color="orange")
    # plt.xlabel("F_0")
    # plt.ylabel("Purification Steps")
    # plt.title("Purification Steps vs F_0")
    # plt.grid(True)
    # plt.show()

    plt.figure()
    plt.plot(F_vals, costs, marker="^", color="green")
    plt.xlabel("F_0")
    plt.ylabel("Cost")
    plt.title("Cost vs F_0")
    plt.grid(True)
    plt.show()
