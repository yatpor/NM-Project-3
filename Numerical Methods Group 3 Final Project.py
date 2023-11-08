### Numerical Methods Group 3 Final Project ###

## Python package import (these packages must be installed for the script to run) ##

import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import sparse
import scipy.sparse.linalg
np.set_printoptions(threshold=sys.maxsize)


## 2D Mesh Generation ##

x, y = np.meshgrid(np.linspace(0,1,26), np.linspace(0,1,26))    # Setting plate size and number of grid points
h = x[0][1] - x[0][0]       # Calculating grid spacing

## Internal Source ##

def f(x, y):

    if x> 0.45 and x<0.55 and y>0.45 and y<0.55: # If you are in small square at center of plate
        source = 100000     # Non-zero source specification

    else: 
        source = 0          # Source everywhere else is zero

    return source

## Initial Conditions ##

T = np.zeros(((len(x)-2)*(len(y)-2), 1))    # Creating initial T matrix
T = sparse.csc_matrix(T)                    # Converting T to a sparse matrix (boosts solution efficiency)

## A Matrix Specification ##

A = np.zeros(((len(x)-2)*(len(y)-2), (len(x)-2)*(len(y)-2)))    # Creating initial A matrix

for row in range((len(x)-2)*(len(y)-2)):            # The following conditional statements populate the A matrix with the necessary values
    for column in range((len(x)-2)*(len(y)-2)):
        if column == row:       # Set diagonals
            A[row][column] = 4
        if column == row + 1:       # Consider temperature of node to the right
            A[row][column] = -1
            for i in range((len(x)-2)*(len(y)-2)):
                if column == i*(len(x)-2):      # Unless there is no node to the right
                    A[row][column] = 0
        elif column >= 0 and column == row - 1:     # Consider temperature of node to the left
            A[row][column] = -1
            for i in range((len(x)-2)*(len(y)-2)):
                if column == i*(len(x)-2)-1:        # Unless there is no node to the left
                    A[row][column] = 0
        elif column == row + len(x)-2:      # Consider temperature of node below
            A[row][column] = -1
            if row == max(range((len(x)-2)*(len(y)-2))):        # Unless there is no node below
                A[row][column] = 0
        elif column >= 0 and column == row - (len(x)-2):        # Consider temperature of node above
            A[row][column] = -1
            if row == 0:        # Unless there is no node above
                A[row][column] = 0

A = sparse.csc_matrix(A)        # Converting A matrix to a sparse matrix

## A matrix component breakdown ##

U = -1*sparse.triu(A, k=1)      # Breaking A matrix down into upper triangle (U), lower triangle (L), and diagonal (D) components
L = -1*sparse.tril(A, k=-1)
D = A + U + L

## RHS Matrix + Boundary Conditions ##

b = np.zeros(((len(x)-2)*(len(y)-2), 1))        # Creating initial rhs matrix (boundary conditions + source terms)

Top = 0        # Dirichlet boundary conditions for each wall
Bottom = 500
Left = 400
Right = 200

for i in range(len(b)):     # Populating rhs matrix with boundary conditions
    if i < len(x)-2:
        b[i][0] = Top       # Top boundary temp
    elif i >= (len(x)-2)*(len(y)-2)-len(x)+2:
        b[i][0] = Bottom       # Bottom boundary temp
    for j in range(len(x)-2):
        b[j*(len(x)-2)][0] = Left    # Left boundary temp
    for j in range(len(x)-2):
        b[j*(len(x)-2)-1][0] = Right  # Right boundary temp

for i in range(len(b)):         # Boundary condition corner correction
    if i==0:
        b[i][0] = Top+Left
    elif i == int(max(range(len(b)))):
        b[i][0] = Bottom+Right
    elif i == 0 + int((len(x)-2) -1):
        b[i][0] = Top+Right
    elif i == int(max(range(len(b)))) - int((len(x)-2) -1 ):
        b[i][0] = Bottom+Left

x_2 = []
for i in range(len(x)-2):
    x_2.append(x[i+1][1:-1])

y_2 = []
for i in range(len(y)-2):
    y_2.append(y[i+1][1:-1])

for i in range(len(y_2)):       # Applying internal source (with grid spacing consideration)
    for j in range(len(x_2)):
        b[i*(len(y_2))+j][0] += ((h**2) * f(x_2[i][j], y_2[i][j]))


## Solving for final T matrix ##

def Jloop(D, U, L, b, T, iterations):       # Jacobian iteration function

    Q = scipy.sparse.linalg.inv(D)@(L+U)      # Create T (named Q to avoid confusion with temperature)
    c = scipy.sparse.linalg.inv(D)@b          # Create c
    x_init = T      # Initialize temperature
    it = 0          # Initialize iteration count

    while it < iterations:      # Loop performing Jacobian iteration until specified iteration count is reached
        xnew = Q@x_init + c
        x_init = xnew
        it += 1
    
    return xnew     # Returning final solution

iterations = 1000      # iteration count specification

T = Jloop(D, U, L, b, T, iterations)        # Temperature solution


## Plotting (Deactivate if using animation collection section) ##

T_plot = T.reshape((len(x)-2, len(y)-2))      # Reshaping T for contour plotting

fig, axs = plt.subplots()     # Creating figure

axs.imshow(T_plot, cmap = plt.cm.inferno, vmin = np.ndarray.min(T), vmax = np.ndarray.max(T), extent = [x[0][1], x[-1][-2], y[1][0], y[-2][-1]])      # Generating contour
norm = mpl.colors.Normalize(vmin = np.ndarray.min(T), vmax = np.ndarray.max(T))
sm = plt.cm.ScalarMappable(norm=norm, cmap = plt.cm.inferno)
sm.set_array([])
fig.colorbar(ax = axs, mappable = sm)

plt.savefig(str(len(x)-1) + 'x' + str(len(y)-1)+' ('+str(iterations)+' iterations).png')      # Saving image to current working directory


## Animation image collection ##

# for i in range(iterations):        # Jacobian solution loop for animation creation (generates multiple images of solutions using different iteration counts)
#     iterations = i+1
#     T = np.zeros(((len(x)-2)*(len(y)-2), 1))    # Initializing T matrix
#     T = sparse.csc_matrix(T)
#     T = Jloop(D, U, L, b, T, iterations)        # Solving for T for current iteration count
#     T_plot = T.reshape((len(x)-2, len(y)-2))

#     fig, axs = plt.subplots()

#     axs.imshow(T_plot, cmap = plt.cm.inferno, vmin = np.ndarray.min(T), vmax = np.ndarray.max(T), extent = [x[0][1], x[-1][-2], y[0][1], y[-1][-2]])
#     norm = mpl.colors.Normalize(vmin = np.ndarray.min(T), vmax = np.ndarray.max(T))
#     sm = plt.cm.ScalarMappable(norm=norm, cmap = plt.cm.inferno)
#     sm.set_array([])
#     fig.colorbar(ax = axs, mappable = sm)

#     plt.savefig(str(len(x)-1) + 'x' + str(len(y)-1)+' ('+str(iterations)+' iterations).png')
#     plt.close(fig)