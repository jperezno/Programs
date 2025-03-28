#This program aims to add a realistic error model to the DEJMPS purification model
#it is based on dejmpsoptimization.py 
import math
import matplotlib.pyplot as plt
#error functions

#no eta error
def no_readerror(eta):
    noetaerr=(eta**2)+((1-eta)**2)
    return(noetaerr)

def readerror(eta):
    etaerr=2*(eta)*(1-eta)
    return(etaerr)

#this is just a normalization factor written here to avoid mistakes
def egate_er(eps_g):
    prepaulinorm=(2*(eps_g) - (eps_g**2))/((1-eps_g)**2)
    return(prepaulinorm)

#finish error functions
#todo add probabilities
#Aqui en estas hay que reemplazar B con D, se me fue no se te olvide, este error fue arreglado en presentacion
################original
def error_quad(x1,x2,y1,y2,eta,N,eps_g,PauliErrors):
    err_quad=(((x1**2)+(x2**2))*(no_readerror(eta)) + (((x1*y1)+(x2*y2))*(readerror(eta))) +
    (egate_er(eps_g))*(PauliErrors))/(N/((1-eps_g)**2))
    return (err_quad)
################
#Asign= error_quad(A,B,C,D,eta,N,eps_g,phiplus)
#Csign= error_quad(C,D,A,B,eta,N,eps_g,psiplus)
def Aerror_quad(x1,x2,y1,y2,eta,N,eps_g,px,py,pz):
    err_quad=(((x1**2)+(x2**2))*(no_readerror(eta)) + (((x1*y1)+(x2*y2))*(readerror(eta))) +
    (egate_er(eps_g))*((px*((y1*y1)+(y2*y2)))+(py*((y1*y2)+(y1*y2)))+(pz*((x1*x2)+(x1*x2)))))/(N/((1-eps_g)**2))
    return (err_quad)

def Cerror_quad(x1,x2,y1,y2,eta,N,eps_g,px,py,pz):
    err_quad=(((x1**2)+(x2**2))*(no_readerror(eta)) + (((x1*y1)+(x2*y2))*(readerror(eta))) +
    (egate_er(eps_g))*((px*((x1*y1)+(x2*y2)))+(py*((x1*y2)+(x2*y1)))+(pz*((x1*y2)+(x2*y1)))))/(N/((1-eps_g)**2))
    return (err_quad)

################original
def error_line(x1,x2,y1,y2,eta,N,eps_g,PauliErrors):
    err_line= ((2*y1*y2)*(no_readerror(eta)) + (((x1*y2)+(y1*x2))*(readerror(eta))) + 
    (egate_er(eps_g))*(PauliErrors))/(N/((1-eps_g)**2))
    return (err_line)
###############
#Bsign= error_line(C,D,A,B,eta,N,eps_g,phimin)
#Dsign= error_line(A,B,C,D,eta,N,eps_g,psimin)
def Berror_line(x1,x2,y1,y2,eta,N,eps_g,px,py,pz):
    err_line= ((2*y1*y2)*(no_readerror(eta)) + (((x1*y2)+(y1*x2))*(readerror(eta))) + 
    (egate_er(eps_g))*((px*((y1*y2)+(y1*y2)))+(py*((y1*y1)+(y2*y2)))+(pz*((x1*x1)+(x2*x2)))))/(N/((1-eps_g)**2))
    return (err_line)
def Derror_line(x1,x2,y1,y2,eta,N,eps_g,px,py,pz):
    err_line= ((2*y1*y2)*(no_readerror(eta)) + (((x1*y2)+(y1*x2))*(readerror(eta))) + 
    (egate_er(eps_g))*((px*((x1*y2)+(x2*y1)))+(py*((x1*y1)+(x2*y2)))+(pz*((x1*y1)+(x2*y2)))))/(N/((1-eps_g)**2))
    return (err_line)


def normalization_error(x1,x2,y1,y2,eta):
    norm_error=(((x1+x2)**2+(y1+y2)**2)*(no_readerror(eta)) + (2*(x1+x2)*(y1+y2)*(readerror(eta))))
    return(norm_error)

##this is not yet used
# gateerrors            A  B  C  D
def pauli_err_phiplus(px,py,pz,x1,x2,y1,y2):#A
    # eps_x=((y1*y1)+(y2*y2))
    # eps_y=((y1*y2)+(y1*y2))
    # eps_z=((x1*x2)+(x1*x2))
    pauli_phi_plus=(px*((y1*y1)+(y2*y2)))+(py*((y1*y2)+(y1*y2)))+(pz*((x1*x2)+(x1*x2)))
    return (eps_x,eps_y,eps_z)

def pauli_err_phiminus(px,py,pz,x1,x2,y1,y2):#B
    # eps_x=((y1*y2)+(y1*y2))
    # eps_y=((y1*y1)+(y2*y2))
    # eps_z=((x1*x1)+(x2*x2))
    pauli_phi_min=(px*((y1*y2)+(y1*y2)))+(py*((y1*y1)+(y2*y2)))+(pz*((x1*x1)+(x2*x2)))
    return(eps_x,eps_y,eps_z)

def pauli_err_psiplus(px,py,pz,x1,x2,y1,y2):#C
    # eps_x=((x1*y1)+(x2*y2))
    # eps_y=((x1*y2)+(x2*y1))
    # eps_z=((x1*y2)+(x2*y1))
    pauli_psi_plus=(px*((x1*y1)+(x2*y2)))+(py*((x1*y2)+(x2*y1)))+(pz*((x1*y2)+(x2*y1)))
    return(eps_x,eps_y,eps_z)

def pauli_err_psiminus(px,py,pz,x1,x2,y1,y2):#D
    # eps_x=((x1*y2)+(x2*y1))
    # eps_y=((x1*y1)+(x2*y2))
    # eps_z=((x1*y1)+(x2*y2))
    pauli_psi_min=(px*((x1*y2)+(x2*y1)))+(py*((x1*y1)+(x2*y2)))+(pz*((x1*y1)+(x2*y2)))
    return(eps_x,eps_y,eps_z)

def indices(A,B,C,D,eta,N,eps_g,px,py,pz):
    Asign= Aerror_quad(A,B,C,D,eta,N,eps_g,px,py,pz)
    Bsign= Berror_line(C,D,A,B,eta,N,eps_g,px,py,pz)
    Csign= Cerror_quad(C,D,A,B,eta,N,eps_g,px,py,pz)
    Dsign= Derror_line(A,B,C,D,eta,N,eps_g,px,py,pz)
    return(Asign,Bsign,Csign,Dsign)


#lets give some numbers to the program
#eta here depends on the platform of your choice (10e-4 for SiV Centers)
# readerror is given as \epsilon_r=1-\eta
eta=(1-(10e-4))
eps_g=5e-4
# PauliErrors=0.5
px=0.8
py=px
pz=py
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
Nor=normalization_error(A,B,C,D,eta)
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
    Asign,Bsign,Csign,Dsign=indices(A,B,C,D,eta,Nor,eps_g,px,py,pz)
    print(i,Asign,Bsign,Csign,Dsign)
    # print(f"A={Asign}, B={Bsign},C={Csign},D={Dsign}")
    #update of the values for the next round:
    A=Asign
    B=Bsign
    C=Csign
    D=Dsign
    check=A+B+C+D
    print("sum=",check)
    Nor=normalization_error(A,B,C,D,eta)
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
