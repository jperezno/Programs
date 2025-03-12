#This program aims to take the differnt coeficients from the DEJMPS protocol
# and get some nice numbers and tables out of it 
import math
#Lets declare the coeficients 
def equ7quad(x1,x2,N):
    quad= ((x1**2)+(x2**2))/N
    return (quad)

def equ7lin(y1,y2,N):
    line= (2*y1*y2)/N
    return (line)

def normalization (x1,x2,y1,y2):
    norm=((x1+y2)**2+(y1+x2)**2)
    return(norm)

def indices(A,B,C,D,N):
    Asign= equ7quad(A,D,N)
    Bsign= equ7lin(C,B,N)
    Csign= equ7quad(C,B,N)
    Dsign= equ7lin(A,D,N)
    #should the normalization factor also be a function called here?
    return(Asign,Bsign,Csign,Dsign)

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
def error_quad(x1,x2,y1,y2,eta,N,eps_g):
    err_quad=(((x1**2)+(x2**2))*(no_readerror(eta)) + (((x1*y1)+(x2*y2))*(readerror(eta))) +
    (egate_er(eps_g))*(PauliErrors))/(N/((1-eps_g)**2))
    return (err_quad)

def error_line(x1,x2,y1,y2,eta,N,eps_g):
    err_line= ((2*y1*y2)*(no_readerror(eta)) + ((x1*y2)+(y1*x2)*(readerror(eta))) + 
    (egate_er(eps_g))*(PauliErrors))/(N/((1-eps_g)**2))
    return (err_line)

def normalization_error(x1,x2,y1,y2,eta):
    norm_error=(((x1+y2)**2+(y1+x2)**2)*(no_readerror(eta)) + (2*(x1+x2)*(y1+y2)*(readerror(eta))))
    return(norm_error)
###
#gateerrors            A  B  C  D
def pauli_err_phiplus(x1,x2,y1,y2):
    eps_x=((y1*y1)+(y2*y2))
    eps_y=((y1*y2)+(y1*y2))
    eps_z=((x1*x2)+(x1*x2))
    return(eps_x,eps_y,eps_z)

def pauli_err_phiminus(x1,x2,y1,y2):
    eps_x=((y1*y2)+(y1*y2))
    eps_y=((y1*y1)+(y2*y2))
    eps_z=((x1*x1)+(x2*x2))
    return(eps_x,eps_y,eps_z)

def pauli_err_psiplus(x1,x2,y1,y2):
    eps_x=((x1*y1)+(x2*y2))
    eps_y=((x1*y2)+(x2*y1))
    eps_z=((x1*y2)+(x2*y1))
    return(eps_x,eps_y,eps_z)

def pauli_err_psiminus(x1,x2,y1,y2):
    eps_x=((x1*y2)+(x2*y1))
    eps_y=((x1*y1)+(x2*y2))
    eps_z=((x1*y1)+(x2*y2))
    return(eps_x,eps_y,eps_z)

#lets give some numbers to the program
#first round here we assume A remains as it is but take B=C=D
#lets assume A is just a random number close to 1
A=0.5
print('The value of A wont change as its fixed A=',A, 'and the sum of A,B,C,D cant be more than 1')
#values for B,C,D
B=(1-A)/3
C=B
D=C
# check=A+B+C+D
# print("sum=",check)
#Normalization factor for the first time is always gonna be the same
Nor=normalization(A,B,C,D)
print('The original fidelity is F=',Nor)

#iterator
i=1
iter=5
print('Table:')
#After every round we will permute B,C,D and leave A untouched
#A,B,C,D
while i <= iter:
    print("Iteration",i)
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
