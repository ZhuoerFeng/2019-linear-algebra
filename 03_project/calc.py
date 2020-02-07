import numpy as np 

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
A = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 2, 0, 0, 0],
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

B = np.array([[1,  0,  -2, 0, 0, 0],
              [0,  1,  0,  0, 0, 0],
              [0,  2,  1,  0, 0, 0],
              [-3, -3, 0,  1, 0, 0],
              [2,  1,  0,  0, 1, 0],
              [1,  5,  -4, 0, 0, 1]])
B = a
q, r = np.linalg.qr(B)
print(q)
print(r)
s = np.matmul(r, q)
print(s)

# q, r = np.linalg.qr(s)
# print(q)
# print(r)
# s = np.matmul(r, q)
# print(s)

for i in range(1, 3):
    q, r = np.linalg.qr(s)
    print(q)
    print(r)
    s = np.matmul(r, q)
    print(s)

print(np.linalg.eigvals(B))
