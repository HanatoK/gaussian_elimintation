#!/usr/bin/env python3
import numpy as np
import timeit

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
X = np.linalg.solve(A, B)
np.savetxt('reference.txt', X, fmt='%12.7f')
