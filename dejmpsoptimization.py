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

#lets give some numbers to the program
#first round here we assume A remains as it is but take B=C=D

A_1=0.95 #lets assume A is just a random number close to 1
print('The value of A wont change as its fixed A=',A_1, 'and the sum of A,B,C,D cant be more than 1')
B=float(input('What is the value of B=C=D:'))#here it will be 0.01666
C=B
D=B
Nor=((A_1+B)**2+(C+D)**2)
print('The original fidelity is F=',Nor)
i=1
while i <= iter:
    #inside the brackets we modify ABCD accordingly, N remains unchanged
    Asign=equ7quad(A_1,B,Nor) #this one for A
    Bsign=equ7lin(C,D,Nor)
    Csign=equ7quad(C,D,Nor)
    Dsign=equ7lin(A_1,B,Nor)
    print('Table:')
    print(i,Asign,Bsign,Csign,Dsign)

    