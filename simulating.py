# Lets try to simulate our equations
import math
import matplotlib.pyplot as plt

#Here we assume we have already done a pi/2 rotation so this are the already swapped equations.

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


# Will be 0.9, when we swap we get 0.8, we want to get to 0.9, apply hybridprotocol to 0.8 until we get 0.9
# A on dejmps>0.8 B,C,D check if we are getting 
# every time we apply DEJMPS we get m=m+1
# each time we apply dejmps we take a note of P_f
# Run simulation for a single starting fidelity

def simulate_purification(source_fid, F_target, eta,epsilon_g,max_iter=15):
    # pre_swap_fid=F_target #Our TARGET is to go back to this fidelity
    # source_fid=(1 + ((4 * (eta) ** 2 - 1) * ((4 *pre_swap_fid - 1) / 3) ** 2)) / 4
    # print (source_fid)
    # here and print n as a functuion of the target fid (swap fidelity)
    p_z = 0.5
    p_x = (1 - p_z)/2
    p_y = (1 - p_z)/2

    A = source_fid
    # B = 2*(1 - A) / 3
    # C =(1 - A) / 6
    # D =(1 - A) / 6
    B = C = D = (1 - A) / 3
    p = acc_prob(A, B, C, D, eta)
    iterations = 0
    p_prod = 1
    # lmbda=

    while A < F_target and iterations < max_iter:
        A_new = hybA(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        B_new = hybB(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        C_new = hybC(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        D_new = hybD(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        A, B, C, D = A_new, D_new, C_new, B_new  #B to D flip
        p = acc_prob(A, B, C, D, eta)
        print('this is P=',p)
        p_prod *= p
        # print('this is Ptot=',p_prod)
        iterations += 1
        
        # if epsilon_g == 0.01 and A >= 0.95:
        #     # return iterations , A
        #     return (math.log2(((2**iterations)/p_prod)+1)), A
        if A >= F_target:
            # return iterations , A
            return (math.log2(((2**iterations)/p_prod)))+1, A
    return None, A  # Failed to reach target

configs = [
    # {'epsilon_g': 5e-4, 'eta': 1.0-1e-4, 'label': r'$\epsilon_r=SiV,\ \epsilon_g=SiV$', 'color': 'blue'},
    # {'epsilon_g': 3.5e-4, 'eta': 1.0-4e-4, 'label': r'$\epsilon_r=NV,\ \epsilon_g=NV$', 'color': 'green'},
    # {'epsilon_g': 5e-4, 'eta': 1.0-1e-5, 'label': r'$\epsilon_r=TrIon,\ \epsilon_g=TrIon$', 'color': 'red'},
    {'epsilon_g': 2.5e-3, 'eta': 0.99400, 'label': r'Superconducting', 'color': 'red'},
    {'epsilon_g': 5e-4, 'eta': 0.99990, 'label': r'SiV', 'color': 'blue'},
    {'epsilon_g': 3.5e-4, 'eta': 0.99960, 'label': r'NV' ,'color': 'green'},
    {'epsilon_g': 5e-4, 'eta': 0.99999, 'label': r'TrIon', 'color': 'orange'},
    {'epsilon_g': 2.5e-3, 'eta': 0.99600, 'label': r'NeutralAtoms', 'color': 'violet'},
    ]

# Sweep over pre-swap fidelities (i.e., intended targets)
x=0.71
step=0.01
target_vals = [round(x + step * i, 9) for i in range(int((0.99999 - x) / step) + 1)]
max_iter = 100

plt.figure(figsize=(8, 5))

for config in configs:
    results = []
    eta = config['eta']
    epsilon_g = config['epsilon_g']
    print(f"\n--- Results for {config['label']} ---")
    
    for F_target in target_vals:
        source_fid = (1 + ((4 * eta ** 2 - 1) * ((4 * F_target - 1) / 3) ** 2)) / 4
        iters, final_fid = simulate_purification(source_fid, F_target, eta, epsilon_g)
        E_target = 1 - F_target
        results.append((F_target, iters if iters is not None else max_iter, final_fid))
        print(f"Pre swap F: {F_target:.6f} Post-swap F: {source_fid:.6f}, Steps: {iters if iters is not None else 'Failed'}")

    x_vals = [r[0] for r in results]
    y_vals = [r[1] for r in results]
    plt.plot(x_vals, y_vals, 'o-', color=config['color'], label=config['label'])

##log scale 
# plt.xscale('log')
plt.xlabel(r'Target (pre swap) fidelity')
plt.ylabel('$ \lambda $ ')
plt.title('Fidelity restauration after swap')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
