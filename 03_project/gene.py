import numpy as np 
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)
A = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 2, 0, 0, 0],
              [0, 0, 0, 2, 0, 0],
              [0, 0, 0, 0, 3, 0],
              [0, 0, 0, 0, 0, 4]])

B = np.array([[1, 1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 2, 1, 0, 0],
              [0, 0, 0, 2, 0, 0],
              [0, 0, 0, 0, 3, 0],
              [0, 0, 0, 0, 0, 4]])

P = np.array([[1,  0,  -2, 0, 0, 0],
              [0,  1,  0,  0, 0, 0],
              [0,  2,  1,  0, 0, 0],
              [-3, -3, 0,  1, 0, 0],
              [2,  1,  0,  0, 1, 0],
              [1,  5,  -4, 0, 0, 1]])
test = np.array([[1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [1, 0, 0, 0, 1, 1]])
P = np.dot(test, P)
P = np.dot(P, np.linalg.inv(test))
a = np.dot(P,A)
a = np.dot(a, np.linalg.inv(P))
b = np.dot(P,B)
b = np.dot(a, np.linalg.inv(P))
print(a)
print(b)
print(np.linalg.eigvals(A))
print(np.linalg.eigvals(B))
print(np.linalg.eigvals(P))
print(np.linalg.eigvals(a))
print(np.linalg.eigvals(b))
