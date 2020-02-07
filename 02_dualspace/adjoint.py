import numpy as np 
m = np.array([ [10, 2, 2], [2, 30, 5], [2, 5, 10]])
a = np.linalg.cholesky(m)
print(a)