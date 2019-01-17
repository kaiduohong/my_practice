import numpy as np
def rbf(X1, X2, gamma):
    x1_sq = np.sum(np.power(np.matrix(X1), 2), 1)
    n1 = np.shape(X1)[0]
    if type(X2) == type(None):
        t = x1_sq * np.ones([1, n1])
        D = t + t.transpose() - 2 * X1 * X1.transpose()
    else:
        x2_sq = np.sum(np.power(np.matrix(X2), 2), 1)
        n2 = np.shape(X2)[0]
        D = x1_sq * np.ones([1, n2]) + (x2_sq * np.ones([1, n1])).transpose() - 2 * X1 * X2.transpose()
    return np.exp(-gamma * D)


X1 = np.matrix([[1,2,3,4],[1,3,2,3],[3,4,2,5]])

X2 = np.array([[2,4,5,6],[3,3,2,5]])

print(rbf(X1,X2,gamma=1))