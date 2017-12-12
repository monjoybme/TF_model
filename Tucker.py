import numpy as np

X = np.arange(24).reshape(4, 3, 2)
mode = 2
x, y, z = X.shape
print(X)
if mode == 0:
    A = np.zeros((y, x*z), dtype=float)
    print(A.shape)
    k=0
    for i in range (0, x):
        for j in range (0, z):
            k=k+1
            A[:, k-1] = X[i, :, j]
            # for k in range (0, x-1):
            print(i,j,k-1, X[i, :, j])

if mode == 1:
    A = np.zeros((z, x*y), dtype=float)
    print(A.shape)
    k=0
    for i in range (0, x):
        for j in range (0, y):
            k=k+1
            A[:, k-1] = X[i, j, :]
            # for k in range (0, x-1):
            print(i,j,k-1, X[i, j, :])

if mode == 2:
    A = np.zeros((x, y*z), dtype=float)
    print(A.shape)
    k=0
    for i in range (0, z):
        for j in range (0, y):
            k=k+1
            A[:, k-1] = X[:, j, i]
            # for k in range (0, x-1):
            print(i,j,k-1,X[:, j, i])
print(A)




# def unfolding(X, mode):
#
#     return A
#
#
# def tucker (rank, modes, iter):
#     X = np.arange(125).reshape(5, 5, 5)
#     print(X)
#
#
#     return G, B1, B2, B3
#
#
# tucker(rank=0, modes=0, iter=0)