#importing
import matplotlib.pyplot as np
import numpy as np


from sympy import symbols, Eq, simplify, solve, series, limit, re
import numpy as np
import matplotlib.pyplot as plt
import re as ree
#defining function
def detect_numbers(string):
    numbers = ree.findall(r'\d+', string)
    return numbers

epsilon, eta, emem = symbols('epsilon eta e_{mem}')
expression = -(((1-epsilon)**2 + (epsilon/3)**2)*(eta**2 + (1 - eta)**2) + (epsilon*(1-epsilon)/3 + (epsilon/3)**2)*2*eta*(1 - eta)) / \
    (((1-epsilon)**2 + 2*epsilon*(1 - epsilon)/3 + (5/9)*epsilon**2)*(eta**2 + (1 - eta)**2) + (epsilon*(1-epsilon)/3 + (epsilon/3)**2)*8*eta*(1 - eta)) + 2*emem +1

def purify(epsilon, eta, emem):
    expression = -(((1-epsilon)**2 + (epsilon/3)**2)*(eta**2 + (1 - eta)**2) + (epsilon*(1-epsilon)/3 + (epsilon/3)**2)*2*eta*(1 - eta)) / \
        (((1-epsilon)**2 + 2*epsilon*(1 - epsilon)/3 + (5/9)*epsilon**2)*(eta**2 + (1 - eta)**2) + (epsilon*(1-epsilon)/3 + (epsilon/3)**2)*8*eta*(1 - eta)) + 2*emem +1
    return expression

upper_limit = 1.0; lower_limit = 0.0
epsilons_0 = np.linspace(0,upper_limit, num=30)
epsilons = [1-purify(1-i, 1, 0) for i in epsilons_0]

plt.plot(epsilons_0, epsilons, label = 'purification')
plt.plot(np.linspace(0,upper_limit,num=30), np.linspace(0,upper_limit,num=30), linestyle='dashed', label = 'reference')
plt.title('Purification vs starting fidelity')
plt.xlabel('$F$')
plt.ylabel("$F^'$")
plt.legend()
plt.show()
