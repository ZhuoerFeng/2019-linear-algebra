import numpy as np 
a = np.array([[1, 0, -2, 31/9.0], [0, 1, -1.5, 7/6.0], [0, 0, 1, 0], [0, 0, 0, 1]])
print("a = \n")
print(a)

b = np.array([[1, 2, 1, 2], [0, 1, 3, 4], [0, 0, 3, 5], [0, 0, 0, 4]])
print("b = \n")
print(b)

c = np.linalg.inv(a)
print("inv a = \n")
print(c)

temp = np.dot(a, b)
d = np.dot(temp, c)
print("ans = \n")
print(d)


