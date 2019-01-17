import numpy as np
from functools import partial

def zscore(X,ddof = 1):
    X = np.matrix(X)
    #ddof number of freedom variable
    #zscore scales X using the sample standard deviation, with n - ddof in the denominator of the standard deviation formula.
    #ddof ddof == 0 then zscore scales X using the sample standard deviation, with n in the denominator of the standard deviation formula.
    #ddof ddof == 1(default) then zscore scales X using the population standard deviation, with n - 1 in the denominator of standard deviation formula.
    n = np.shape(X)[0]

    return np.divide((X - np.tile(np.mean(X,0),(n,1)) ),np.tile(np.std(X,0,ddof=ddof),(n,1)))



def average_row(X):
    X = np.matrix(X)
    return np.multiply(np.tile(1 / np.sum(X, 1), (1, np.shape(X)[1])), X)

#X_ri = X_ri / norm(X_ri,2)
def normalize(X):
    X = np.matrix(X)
    return np.multiply(np.tile(1 / np.sqrt(np.sum(np.power(X,2),1)),(1,np.shape(X)[1])) , X)

def kernel(ker,X1,X2=None,**kwargs):

    def linear(X1,X2,**kwargs):
        if type(X2) == type(None):
            K = X1 * X1.transpose()

        else:
            K = X1 * X2.transpose()
        return K

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

    def sam(X1,X2,**kwargs):
        if type(X2) == type(None):
            D = X1 * X1.transpose()
        else:
            D = X1 * X2.transpose()

    switch = {
        "linear": partial(linear),
        "rbf":partial(rbf),
        "sam":partial(sam)
    }
    X1 = np.matrix(X1)
    X2 = np.matrix(X2) if type(X2) != type(None) else None

    return switch[ker](X1,X2,**kwargs)