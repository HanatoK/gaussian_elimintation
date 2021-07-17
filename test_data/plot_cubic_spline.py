#!/usr/bin/env python3 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

origin_X = np.arange(0, 10)
origin_Y = np.sin(origin_X / 2.0 * np.pi)

spline_data = pd.read_csv('spline_interp_natural.txt', header=None, delimiter='\s+', comment='#')
X = np.array(spline_data[0])
Y = np.array(spline_data[1])
Y2 = np.array(spline_data[2])
plt.plot(origin_X, origin_Y, label='Origin')
plt.plot(X, Y, label='Natural')
plt.plot(X, Y2, label='Not-a-knot')
plt.legend()
plt.savefig('test_spline.png', dpi=300)