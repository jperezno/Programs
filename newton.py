# #Este programa encuentra una solucion a f(x)=0 dada una aproximación inicial p0 con el metodo de netwon

# #funcion
import math, cmath
def func(x): #Aquí se define la funcion a resolver
    F=-0.5 + 2.348642069355*(0.250526104820122*math.exp(0.226757369614512*x**4.0312) - 1)**4*math.exp(-0.90702947845805*x**4.0312) + 1/4
    return (F)
def funcder(x): #Aquí se define la derivada de la función anterior
    Fd= 8.58761533785385*x**3.0312*(0.250526104820122*math.exp(0.226757369614512*x**4.0312) - 1)**3*math.exp(-0.90702947845805*x**4.0312)
    return (Fd)
#programa
print('Se va a encontrar una solución dada una aproximación inical')
# p0=float(input('Dame la aproximación inicial = '))
p0=1
# tol=float(input('Dame la tolerancia de error = '))
tol=1e8
# n0=int(input('Dame el número de iteraciones a realizar ='))
n0=100
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


# import math, cmath

# # Función f(x)
# def func(D):
#     return -0.5+2.348642069355*(0.250526104820122*cmath.exp(0.226757369614512*D**4.0312) - 1)**4*cmath.exp(-0.90702947845805*D**4.0312) + 1/4

# # Derivada numérica centrada
# def deriv(f, x, h=1e-8):
#     return (f(x + h) - f(x - h)) / (2*h)

# # Método de Newton–Raphson
# def newton_raphson(f, x0, tol=1e-8, max_iter=50):
#     x = x0
#     for i in range(1, max_iter+1):
#         fx = f(x)
#         dfx = deriv(f, x)
#         if dfx == 0:
#             print("Derivada cero. No se puede continuar.")
#             return None
#         x_new = x - fx / dfx
#         print(f"Iter {i:2d}: x = {x:15.10f}, f(x) = {fx:15.10f}")
#         if abs(x_new - x) < tol:
#             print("\nAproximación de la raíz =", x_new)
#             return x_new
#         x = x_new
#     print("\nNo se alcanzó la tolerancia en el número máximo de iteraciones.")
#     return x

# # Ejemplo de uso
# x0 = 10
# tol = 1e-8
# max_iter = 100

# root = newton_raphson(func, x0, tol, max_iter)
