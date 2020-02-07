import numpy as np 
a = np.array([ [1/3, 1/2, 1], [1/4, 1/3, 1/2], [1/5, 1/4, 1/3]])
print(a)
a_inv = np.linalg.inv(a)
print(a_inv)