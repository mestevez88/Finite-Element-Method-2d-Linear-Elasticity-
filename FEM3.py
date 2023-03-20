#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 18:34:23 2023

@author: manuelestevez
"""
import numpy as np

#Node Dictionary, pos:label
nd = {0:4, 1:7, 2:8, 3:5}

#Location and weights of Gauss Points (pag 89)
GaussLoc={1:[0], 
          2:[-np.sqrt(1/3), np.sqrt(1/3)],
          3:[-0.7745966692, 0.7745966692, 0],
          4:[-0.8611363116, 0.8611363116, -0.3399810436, 0.3399810436],
          5:[-0.9061798459, 0.9061798459, -0.5384693101, 0.5384693101, 0],
          6:[-0.9324695142, 0.9324695142, -0.6612093865, 0.6612093865, -0.2386191861, 0.2386191861]
          }
GaussWeights={1:[2], 
          2:[1, 1],
          3:[0.5555555556, 0.5555555556, 0.8888888889],
          4:[0.3478548451, 0.3478548451, 0.6521451549, 0.6521451549],
          5:[0.2369268851, 0.2369268851, 0.4786286705, 0.4786286705, 0.5688888889],
          6:[0.1713244924, 0.1713244924, 0.3607615730, 0.3607615730, 0.4679139346, 0.4679139346]
          }

#number of Gauss Points for Gaussian Quadrature
Gpx = 2
Gpy = 2
xi = GaussLoc[Gpx]
Wi = GaussWeights[Gpx]
eta = GaussLoc[Gpy]
Wj = GaussWeights[Gpy]

#Example 9.2 Quadrilateral element (pag 233)
E = 3*(10**7)
nu = .3

D = (E/(1 - nu**2))*np.array([
    [1, nu, 0],
    [nu, 1, 0],
    [0, 0, (1 - nu)/2]
    ])

C = np.array([
     [1, 0.6],
     [1, 0.25],
     [2, 0.5],
     [2, 0.7]
     ])

N = []
J = []
detJ = []
Jinv = []
JinvN = []
B = []
BT = []
Ke = []
for i in range(len(xi)):
    for j in range(len(eta)):
        N_ij = np.array([
            [eta[j]-1, 1-eta[j], 1+eta[j], -eta[j]-1],
            [xi[i]-1, -xi[i]-1, 1+xi[i], 1-xi[i]]
            ])
        J_ij = (1/4)*np.matmul(N_ij, C)
        Jinv_ij = np.linalg.inv(J_ij)
        JinvN_ij = (1/4)*np.matmul(Jinv_ij, N_ij)
        B_ij = np.zeros((3,8))
        B_ij[0,0] = JinvN_ij[0,0]
        B_ij[0,2] = JinvN_ij[0,1]
        B_ij[0,4] = JinvN_ij[0,2]
        B_ij[0,6] = JinvN_ij[0,3]
        B_ij[1,1] = JinvN_ij[1,0]
        B_ij[1,3] = JinvN_ij[1,1]
        B_ij[1,5] = JinvN_ij[1,2]
        B_ij[1,7] = JinvN_ij[1,3]
        B_ij[2,0] = JinvN_ij[1,0]
        B_ij[2,1] = JinvN_ij[0,0]
        B_ij[2,2] = JinvN_ij[1,1]
        B_ij[2,3] = JinvN_ij[0,1]
        B_ij[2,4] = JinvN_ij[1,2]
        B_ij[2,5] = JinvN_ij[0,2]
        B_ij[2,6] = JinvN_ij[1,3]
        B_ij[2,7] = JinvN_ij[0,3]
        BT_ij = np.transpose(B_ij)
        detJ_ij = np.linalg.det(J_ij)
        K_ij = Wi[i]*Wj[j]*np.matmul(np.matmul(BT_ij,D), B_ij)*detJ_ij
        #appends
        N.append(N_ij)
        J.append(J_ij)
        Jinv.append(Jinv_ij)
        detJ.append(detJ_ij)
        JinvN.append(JinvN_ij)
        B.append(B_ij)
        BT.append(BT_ij)
        Ke.append(K_ij)

#Element stiffness matrix
K=sum(Ke)