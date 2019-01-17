#TCA
import numpy as np
import  sklearn
import scipy as sp
from scipy.io import loadmat
from operator import itemgetter
from MyLib.common_utils import *
import os
from sklearn.neighbors import KNeighborsClassifier
from scipy.linalg import eig
from scipy.sparse.linalg import  eigs
#import mkl
#import Slepc
#from sklearn import preprocessing


def JDA(X_src,Y_src,X_tar,Y_tar,**kwargs):#one sample per row

    step = kwargs["iter_step"]
    acc_iter = []
    Y_tar_pseudo = []

    X_src = np.matrix(X_src)
    X_tar = np.matrix(X_tar)
    n_src = np.shape(X_src)[0]
    n_tar = np.shape(X_tar)[0]


    for i in range(step):
        #Z = XA,the transformed result
        Z, A = JDA_core(X_src, Y_src, X_tar, Y_tar_pseudo, **kwargs)
        #normalization for better classification performance, normalize every row
        #Z = normalize(Z)
        Zs = Z[:n_src]
        Zt = Z[n_src:]

        knn_model = KNeighborsClassifier(n_neighbors=5).fit(Zs,Y_src)
        Y_tar_pseudo = knn_model.predict(Zt)

        acc = np.sum(Y_tar_pseudo == Y_tar) / n_tar
        print('JDA+NN=%0.4f\n', acc)
        acc_iter.append(acc)
    return A

def JDA_core(X_src, Y_src, X_tar, Y_tar_pseudo, **kwargs):
    _lambda = kwargs["lambda"]
    dim = kwargs["dim"]
    kernel_type = kwargs["kernel_type"]
    gamma = kwargs["gamma"]#kernel_typeisthekernelname, primal | linear | rbf

    X = normalize(np.vstack((X_src,X_tar)))

    n, m = np.shape(X) # n samples, m dim per sample

    ns,nt = np.shape(X_src)[0],np.shape(X_tar)[0];
    e = np.matrix(np.hstack([np.ones((1,ns)) / ns, - np.ones((1,nt)) / nt]))
    C = len(np.unique(Y_src.tolist())) #class number

    M = e.transpose() * e * C #multiply C for better  normalization
    ####### Mc
    N = 0
    if type(Y_tar_pseudo) != type(None) and len(Y_tar_pseudo) == nt:
        for c in np.unique(Y_src):

            e = np.matrix(np.zeros((1, n)))
            e[:,np.argwhere(Y_src == c)] = 1. /  np.count_nonzero(Y_src == c)
            e[:,ns + np.argwhere(Y_tar_pseudo == c)] = -1. / np.count_nonzero(Y_tar_pseudo == c)
            e[e == float("inf")] = 0.
            N = N + e.transpose() * e


    if type(Y_tar_pseudo) != type(None) and len(Y_tar_pseudo) != nt and len(Y_tar_pseudo)>0:
        raise Exception("len(Y_tar_pseudo)!= nt",len(Y_tar_pseudo),nt)

    M = M + N
    M = M / np.linalg.norm(M, 'fro')
    # CenteringmatrixH
    H = np.matrix(np.eye(n) - np.ones((n, n)) / n)
    #print(H[0:10,0:10])
    # Calculation

    if kernel_type == 'primal':
        eig_value,A = eig(X.transpose() * M * X + _lambda * np.eye(m) , X.transpose()*H*X) # estimate(X * M * X'+lambda*eye(m))A = X*H*X'A \Phi
        #A = np.real(A)
        top_k_index = np.argsort(eig_value)[:dim]
        A = A[:,top_k_index]
        Z = X * A
    else:
        K = kernel(kernel_type, X, None, gamma=gamma)

        eig_value,A = eig(K * M * K + _lambda * np.eye(n), K * H * K)
        A = np.real(A)
        top_k_index = np.argsort(eig_value)[0:dim]

        A = A[:, top_k_index]

        Z = K * A

    return Z,A




if __name__ == '__main__':
    Xs, Ys = itemgetter("fts", "labels")(loadmat(r'C:\Users\Administrator\Desktop\project\python\test\data\surf\Caltech10_SURF_L10.mat'))
    Xs = average_row(Xs)# average row
    Xs = zscore(Xs, 0)

    Xt, Yt = itemgetter("fts", "labels")(loadmat(r'C:\Users\Administrator\Desktop\project\python\test\data\surf\amazon_SURF_L10.mat'))# targetdomain
    Xt = average_row(Xt)
    Xt = zscore(Xt, 0)

    Ys,Yt = np.squeeze(np.array(Ys)),np.squeeze(np.array(Yt))

    kwargs={}
    kwargs["iter_step"] = 50   # iterations , default=10
    kwargs["gamma"] = 2 # the parameter
    kwargs["kernel_type"] = 'linear'
    kwargs["lambda"] = 1.0
    kwargs["dim"] = 25
    print("begin")
    #os.system("pause")
    Acc_iter, A = JDA(Xs, Ys, Xt, Yt, **kwargs)









