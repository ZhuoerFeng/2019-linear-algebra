import numpy as np 
from sympy import Matrix
a = np.array([[0, 0, 0, 0, 0, 11], [0, 0, 0, 0, 2, 0], [0, 0, 0, 1, 0, 0], [0, 0, 7, 0, 0, 0], 
              [0, 5, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0]])
m = Matrix(a)
P, J = m.jordan_form()
print(m)

print(P)
print(J)