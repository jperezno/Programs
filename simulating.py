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

def simulate_purification(source_fid, F_target, max_iter=100):
    epsilon_g = 0
    eta = 1
    p_z = 0.5
    p_x = (1 - p_z)/2
    p_y = (1 - p_z)/2

    A = source_fid
    B = C = D = (1 - A) / 3
    p = acc_prob(A, B, C, D, eta)
    iterations = 0
    # lmbda=

    while A < F_target and iterations < max_iter:
        A_new = hybA(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        B_new = hybB(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        C_new = hybC(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        D_new = hybD(A,B,C,D,eta,epsilon_g,p_x,p_y,p_z) / p
        A, B, C, D = A_new, D_new, C_new, B_new  #B to D flip
        p = acc_prob(A, B, C, D, eta)
        iterations += 1

        if A >= F_target:
            return (math.log2((2**(iterations))/p)+1), A
    return None, A  # Failed to reach target

# Sweep over starting fidelities
start_vals = [round(x, 3) for x in list([0.5 + 0.01*i for i in range(50)])]
results = []

for val in start_vals:
    iters, final_fid = simulate_purification(val, F_target=0.98)
    results.append((val, iters, final_fid))

# Plot results
plt.figure(figsize=(8,5))
plt.plot([r[0] for r in results], [r[1] if r[1] else 0 for r in results], 'bo')
plt.xlabel(r'$F_t$')
plt.ylabel(r'$\lambda$')
plt.title('Resource estimation')
plt.legend()
plt.grid(True)
plt.tight_layout
plt.show()