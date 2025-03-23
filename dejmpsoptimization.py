#This program aims to take the differnt coeficients from the DEJMPS protocol
# and get some nice numbers and tables out of it 
import math
import matplotlib.pyplot as plt
#Lets declare the coeficients 
def equ7quad(x1,x2,N):
    quad= ((x1**2)+(x2**2))/N
    return (quad)

def equ7lin(y1,y2,N):
    line= (2*y1*y2)/N
    return (line)

def normalization (x1,x2,y1,y2):
##This one under is the normalization factor for the DEJMPS protocol
##  dejmp((A + B)**2+(C + D)**2
#   norm=((x1+x2)**2+(y1+y2)**2)
##This one under is the normalization factor for the protocol without depolarizing step
    norm=((x1+y2)**2+(y1+x2)**2)
    return(norm)

def indices(A,B,C,D,N):
    Asign= equ7quad(A,D,N)
    Bsign= equ7lin(C,B,N)
    Csign= equ7quad(C,B,N)
    Dsign= equ7lin(A,D,N)
    #should the normalization factor also be a function called here?
    return(Asign,Bsign,Csign,Dsign)

#lets give some numbers to the program
#first round here we assume A remains as it is but take B=C=D
#lets assume A is just a random number close to 1
A=0.85
print('The value of A wont change as its fixed A=',A, 'and the sum of A,B,C,D cant be more than 1')
#values for B,C,D
B=0.10
C=0.025
D=0.025
check=A+B+C+D
print("sum=",check)
#Normalization factor for the first time is always gonna be the same
Nor=normalization(A,B,C,D)
print('The original fidelity is F=',Nor)

#iterator
iterations = []
A_val, B_val, C_val, D_val = [], [], [], []
i=1
iter=5
print('Table:')
#After every round we will permute B,C,D and leave A untouched
#A,B,C,D
while i <= iter:
    print("Iteration",i)
    iterations.append(i)
    A_val.append(A)
    B_val.append(B)
    C_val.append(C)
    D_val.append(D)
    #inside the brackets we modify ABCD accordingly,
    Asign,Bsign,Csign,Dsign=indices(A,B,C,D,Nor)    
    print(i,Asign,Bsign,Csign,Dsign)
    # print(f"A={Asign}, B={Bsign},C={Csign},D={Dsign}")
    #update of the values for the next round:
    A=Asign
    B=Bsign
    C=Csign
    D=Dsign
    # check=A+B+C+D
    # print("sum=",check)
    Nor=normalization(A,B,C,D)
    i=i+1

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(iterations, A_val, label='A', marker='o')
plt.plot(iterations, B_val, label='B', marker='s')
plt.plot(iterations, C_val, label='C', marker='^')
plt.plot(iterations, D_val, label='D', marker='x')

plt.xlabel('Iteration')
plt.ylabel('Fidelities')
plt.title('Purification over iterations')
plt.legend()
plt.grid(True)
plt.show()