import math
import matplotlib.pyplot as plt

# Swap fidelity decay function
def apply_swap_fidelity(F, eta):
    term = ((4 * F - 1) / 3) ** 2
    return (1 + ((4 * eta ** 2 - 1) * term)) / 4

# Simulate decay through swaps until threshold
def simulate_swap_decay(F_start, F_threshold, eta, max_swaps=100):
    swaps = 0
    F = F_start
    while F > F_threshold and swaps < max_swaps:
        F = apply_swap_fidelity(F, eta)
        swaps += 1
        print(swaps,'F=',F)
    return swaps

# Settings
# the threshold can be found from the number of steps calculation
# e.g. for 3 purifications f_thr=0.785 and for 2 purifications f_thr=0.85
F_threshold = 0.75  # lower bound
max_swaps = 100

# Initial fidelities E = 1 - F
x=0.75
F_start_vals = [round(x + 0.005 * i, 3) for i in range(int((0.99 - x) / 0.005) + 1)]
configs = [
    {'eta': 1.0, 'label': r'$\eta = 1.0$', 'color': 'blue'},
    {'eta': 0.99, 'label': r'$\eta = 0.99$', 'color': 'green'},
    {'eta': 0.97, 'label': r'$\eta = 0.97$', 'color': 'red'},
]

plt.figure(figsize=(8, 5))

for config in configs:
    eta = config['eta']
    swap_counts = []
    errors = []
    for F_start in F_start_vals:
        num_swaps = simulate_swap_decay(F_start, F_threshold, eta, max_swaps)
        E_start = 1 - F_start
        swap_counts.append(num_swaps)
        errors.append(E_start)

    plt.plot(errors, swap_counts, 'o-', color=config['color'], label=config['label'])

plt.xlabel('Initial error $E = 1 - F_T$')
plt.ylabel(f'Number of swaps until threshold $F < {F_threshold}$')
plt.title('Number of Swaps vs. Pre-Swap fidelity Number of Purifications n_p=3')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
