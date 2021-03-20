#!/usr/bin/env python3
import numpy as np
import time

# matrix A size
N = 4000
# matrix B columns
M = 100
# generate random number
A = np.random.rand(N, N)
B = np.random.rand(N, M)


np.savetxt('matA.txt', A)
np.savetxt('matB.txt', B)
# solve
start = time.time_ns()
X = np.linalg.solve(A, B)
end = time.time_ns()
elapsed = (end - start) / (10 ** 9)
print(f'Time: {elapsed} s')
np.savetxt('reference.txt', X, fmt='%12.7f')
