#!/usr/bin/env python3
import numpy as np

np.set_printoptions(formatter={'float': '{:10.5f}'.format})

matA = np.array([[ 2.0,  0.5,  1.0, -2.0,  3.0],
                 [ 0.5,  1.0,  0.1,  4.0, -9.0],
                 [ 1.0,  0.1, -3.0, -2.0,  0.0],
                 [-2.0,  4.0, -2.0,  0.2, -1.0],
                 [ 3.0, -9.0,  0.0, -1.0, -0.3]])
R = np.copy(matA)
x = R.transpose()[0]
u = x - np.linalg.norm(x) * np.identity(5)[0]
print("u:")
print(u)
H = np.linalg.norm(u) * np.linalg.norm(u) * 0.5
P1 = np.identity(5) - np.outer(u, u) / H
R = np.matmul(P1, R)
print("P1:")
print(P1)
print("After applying P1:")
print(R)

x = x = R.transpose()[1][1:]
u = x - np.linalg.norm(x) * np.identity(4)[0]
H = np.linalg.norm(u) * np.linalg.norm(u) * 0.5
P2 = np.identity(4) - np.outer(u, u) / H
P2 = np.insert(P2, 0, 0, axis=0)
P2 = np.insert(P2, 0, 0, axis=1)
P2[0][0] = 1
R = np.matmul(P2, R)
print("P2:")
print(P2)
print("After applying P2:")
print(R)

x = x = R.transpose()[2][2:]
u = x - np.linalg.norm(x) * np.identity(3)[0]
H = np.linalg.norm(u) * np.linalg.norm(u) * 0.5
P3 = np.identity(3) - np.outer(u, u) / H
P3 = np.insert(P3, 0, 0, axis=0)
P3 = np.insert(P3, 0, 0, axis=1)
P3[0][0] = 1
P3 = np.insert(P3, 0, 0, axis=0)
P3 = np.insert(P3, 0, 0, axis=1)
P3[0][0] = 1
R = np.matmul(P3, R)
print("P3:")
print(P3)
print("After applying P3:")
print(R)

x = x = R.transpose()[3][3:]
u = x - np.linalg.norm(x) * np.identity(2)[0]
H = np.linalg.norm(u) * np.linalg.norm(u) * 0.5
P4 = np.identity(2) - np.outer(u, u) / H
P4 = np.insert(P4, 0, 0, axis=0)
P4 = np.insert(P4, 0, 0, axis=1)
P4[0][0] = 1
P4 = np.insert(P4, 0, 0, axis=0)
P4 = np.insert(P4, 0, 0, axis=1)
P4[0][0] = 1
P4 = np.insert(P4, 0, 0, axis=0)
P4 = np.insert(P4, 0, 0, axis=1)
P4[0][0] = 1
R = np.matmul(P4, R)
print("P4:")
print(P4)
print("After applying P4:")
print(R)
Q = P1 @ P2 @ P3 @ P4
print("Q:")
print(Q)

print("Matrix A:")
print(matA)
print("Q * R:")
print(Q @ R)
print("Is Q orthonormalized?")
print(Q @ Q.transpose())