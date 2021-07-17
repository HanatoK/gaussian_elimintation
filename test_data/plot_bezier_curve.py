#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

points = np.array([[1, 3], [2, 1], [3.0, 1.0], [3.6, 2]])
t = np.linspace(0, 1, 100)
X = points[0][0] * (1 - t) * (1 - t) * (1 - t) + 3.0 * points[1][0] * t * (1 - t) * (1 - t) + 3.0 * points[2][0] * t * t * (1 - t) + points[3][0] * t * t * t
Y = points[0][1] * (1 - t) * (1 - t) * (1 - t) + 3.0 * points[1][1] * t * (1 - t) * (1 - t) + 3.0 * points[2][1] * t * t * (1 - t) + points[3][1] * t * t * t
plt.scatter(np.transpose(points)[0], np.transpose(points)[1], color='coral')
plt.plot(np.transpose(points)[0], np.transpose(points)[1], color='tab:green')
plt.plot(X, Y, label='Natural', color='deepskyblue')
plt.savefig('test_bezier_curve.png', dpi=300)