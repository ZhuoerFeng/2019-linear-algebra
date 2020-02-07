import numpy as np 

A = np.array([[1, 0, 0, 0, 0, 0, 0, 0], 
              [3, 3, 0, 0, 0, 0, 0, 0],
              [0, 1, 10, 0, 0, 0, 0, 0],
              [0, 0, 1, 2, 0, 0, 0, 0],
              [0, 0, 0, 1, 1.1, 0, 0, 0],
              [0, 0, 0, 0, 1, 1.2, 0, 0],
              [0, 0, 0, 0, 0, 1, 1.3, 0],
              [0, 0, 0, 0, 0, 0, 2, 1.5]])


np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
req = np.zeros((8, 8))
q, r = np.linalg.qr(A)
record = A
s = np.matmul(r, q)
k = 1

for i in range(1, 10):
    req = q
    k = k + 1
    record = s
    q, r = np.linalg.qr(s)
    s = np.matmul(r, q)
    print(q)
    print(s)
    # print(record)
    # print(s)
    # print(s - record)

print(s)

