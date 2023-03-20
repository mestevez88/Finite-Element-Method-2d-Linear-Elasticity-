#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:31:15 2023

@author: manuelestevez
"""

import numpy as np
import matplotlib.pyplot as plt

#import node dict
from FEM1 import nd as nd1
from FEM2 import nd as nd2
from FEM3 import nd as nd3
from FEM4 import nd as nd4

ndls = [nd1, nd2, nd3, nd4]

#import K matrices
from FEM1 import K as K1
from FEM2 import K as K2
from FEM3 import K as K3
from FEM4 import K as K4

Kls = [K1, K2, K3, K4]

C = np.array([[0,1], [1,1], [2,1], [0,.5], [1,.6], [2,.7], [0,0], [1,.25], [2,.5]]) #esto esta super chafa, cambiarlo
#nodes
nodes = 9
#supports
sup = [0, 1, 6, 7, 12, 13]

#K
K = np.zeros((2*nodes, 2*nodes))
for i in range(len(ndls)):
    for j in ndls[i]:
        for k in ndls[i]:
            K[2*ndls[i][j], 2*ndls[i][k]] = K[2*ndls[i][j], 2*ndls[i][k]] + Kls[i][2*j, 2*k]
            K[2*ndls[i][j], 2*ndls[i][k]+1] = K[2*ndls[i][j], 2*ndls[i][k]+1] + Kls[i][2*j, 2*k+1]
            K[2*ndls[i][j]+1, 2*ndls[i][k]+1] = K[2*ndls[i][j]+1, 2*ndls[i][k]+1] + Kls[i][2*j+1, 2*k+1]
            K[2*ndls[i][j]+1, 2*ndls[i][k]] = K[2*ndls[i][j]+1, 2*ndls[i][k]] + Kls[i][2*j+1, 2*k]
#slices for K
Kf = np.delete(np.delete(K, sup, 0), sup, 1)
Kfinv = np.linalg.inv(Kf)

#prescribed forces
load = {0: [0, -20], 1: [0, -20], 2: [0, -20]} # node:[fx, fy]
f_ = np.zeros((2*nodes, 1))
for i in load:
    f_[2*i, 0] = load[i][0]
    f_[2*i+1, 0] = load[i][1]
#slices for f
f = np.delete(f_, sup, 0)

#displacement vector
u = np.matmul(Kfinv, f)
d = np.zeros((2*nodes, 1))
c=0
for i in range(2*nodes):
    if i in sup:
        d[i,0] = 0
        c = c+1
    else:
        d[i,0] = u[i-c,0]

#plot
scale = 9.221*(10**3)
dplot = np.reshape(d, (9,2))
plot = scale*dplot+C
#ranges
x = range(2)
y = range(0, 1)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(*C.T, c='b', marker="s", label='original')
ax1.scatter(*plot.T, c='r', marker="x", label='deformed')
plt.legend(loc='lower right')
plt.show()

print('d:')
print(d)
print('K*d = f + r:')
print(np.matmul(K,d))