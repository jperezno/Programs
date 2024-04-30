#Este programa encuentra una solucion a f(x)=0 dada una aproximación inicial p0 con el metodo de netwon

#funcion
import math
def func(x): #Aquí se define la funcion a resolver
    F= math.exp(x) + 2**(-x) + 2*(math.cos(x)) - 6  
    return (F)
def funcder(x): #Aquí se define la derivada de la función anterior
    Fd= math.exp(x) - (math.log(2))*2**(-x) - 2*(math.sin(x))
    return (Fd)
#programa
print('Se va a encontrar una solución dada una aproximación inical')
p0=float(input('Dame la aproximación inicial = '))
tol=float(input('Dame la tolerancia de error = '))
n0=int(input('Dame el número de iteraciones a realizar ='))
i=1

while i <= n0:
    p=p0-(func(p0)/funcder(p0))
    if abs(p-p0) < tol:
        print('Proceso exitoso. Resultado = ', p)
        break
    i=i+1
    p0=p
#print(p)
print('El metodo falló despues de ', n0, 'iteraciones')