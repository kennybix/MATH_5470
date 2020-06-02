import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import halton_sequence as hs
import lhs
import pattern_search
import preprocessing
import nearestPD as NPD


def inv(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")

    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]

class Kriging:

    def __init__(self,x,y,kernels,theta,weights,eps=1.48e-08):
        self.x = x
        self.y = y
        self.kernels = kernels
        self.theta = theta
        self.weights = weights
        self.eps = eps
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]

    def compute_distances(self,X1,X2): #squared euclidean distance
        n = X2.shape[0] #if not for this bug, I wont need to do the unnecessary transpose for K_s
        k = X1.shape[0]  #fix when all works well
        X1 = X1.reshape(k,1) #must reflect fix
        X2 = X2.reshape(n,1) #must reflect fix
        dists = np.zeros((n,k))
        for i in range(n):
            dists[i,:] = np.sqrt(np.sum((X1 - (X2[i,:]))**2, axis=1)) #effort to compute D instead
        return dists

    def compute_K(self,X,Y,kernel,theta):
        #kernel = self.kernel
        d = self.compute_distances(X,Y)
        self.d = d
        if kernel == "exponential":
            K = (np.exp(-1.0 * np.abs(d)/theta))
        elif kernel == "matern3_2":
            K = (1 + (np.sqrt(3)*d)/theta)*np.exp(((-np.sqrt(3))*d)/theta) 
        elif kernel == "gaussian":
            K = (np.exp(-0.5 * ((d/theta) ** 2))) #on the suspicion that d is already squared
        elif kernel == "matern5_2":
            K = (1 + (np.sqrt(5)*d)/theta + (5/3*(d/theta)**2))*np.exp(((-np.sqrt(5))*d)/theta) 
        else:
            print("Unknown kernel")
        return K

    def compute_R(self,X,Y,theta,weights):
        kernels = self.kernels
        #weights = self.weights
        R = 1
        for i in  range(self.Nk):
            R_k = np.zeros((self.Ns,self.Ns))
            for j in range(len(weights)):
                R_k += weights[j] * (self.compute_K(X[:,i],Y[:,i],kernels[j],theta[i]) + np.diag(self.eps*np.eye(self.Ns)))
            R *= R_k                                                                                                                                                              
        return R

    def compute_Beta(self,R,y):
        self.F = np.ones(self.Ns)[:,np.newaxis]
        try:
            Ri = np.linalg.inv(R)
        except np.linalg.LinAlgError as err:
            R = inv(R)
            Ri = np.linalg.inv(R)
        
        FT = (self.F).T
        temp = (np.dot(FT,np.dot(Ri,self.F)))
        temp2 = (np.dot(FT,np.dot(Ri,y)))
        invtemp = np.linalg.inv(temp)
        Beta = np.dot(invtemp,temp2)
        return Beta 
 

    def K_components(self,Xt):
        K = 1 #initialization
        K_s = 1 #initialization
        theta = self.theta
        Xt = np.array(Xt)
        Xtrain = np.copy(self.x)
        Xtest = np.copy(Xt)
        k = self.Nk #dimension of the sample 
        for i in range(k): #gets the number of dimensions 
            K_k = np.zeros((self.Ns,self.Ns))
            K_s_k = np.zeros((Xt.shape[0],self.Ns)) #continue
            for j in range(len(self.weights)):
                K_k += self.weights[j] * (self.compute_K(Xtrain[:,i],Xtrain[:,i],self.kernels[j],theta[i]) + np.diag(self.eps*np.eye(self.Ns)))
                K_s_k += self.weights[j] * self.compute_K(Xtrain[:,i],Xtest[:,i],self.kernels[j],theta[i])
            K *= K_k
            K_s *= K_s_k
        return K,K_s.T  #had to use transpose for this to work, take note

    def use(self,testdata):   
        self.testdata = testdata
        self.Xtest = np.array(self.testdata)
        self.Nsx = self.Xtest.shape[0] #number of test data
        self.Nkx = self.Nk
        Kc = self.K_components(self.Xtest)
        K = Kc[0] # K(X,X)
        self.K_s = Kc[1] # K(X,X*)

        if(NPD.isPD(K)): #sane check
            self.K = K
        else:
            self.K = NPD.nearestPD(K)
        self.L = np.linalg.cholesky(self.K) # lower cholesky factor
        self.Lk = np.linalg.solve(self.L, self.K_s)
    
    def predict(self):
        Beta = self.compute_Beta(self.K,self.y)
        self.Beta = Beta
        Y = (self.y - np.dot(self.F,Beta))
        mu = np.dot(self.Lk.T, np.linalg.solve(self.L,Y)).reshape((self.Nsx,))
        yhat = np.copy(mu)
        # for i in range(self.Nsx):
        #     yhat[i] += Beta
        Fx = np.ones(self.Nsx)
        Fx = Fx.reshape(self.Nsx,1)
        Beta = Beta.reshape(1,1)
        yhat = yhat + (np.dot(Fx,Beta)).reshape(self.Nsx,)
        self.y_predict = yhat.reshape(-1,1)
        self.y_output = self.y_predict
        return self.y_output

    def computeRMSE(self,y_exact):
        m = len(self.y_output) 
        sum = 0.0
        for i in range(m):
            sum += np.power((y_exact[i] - self.y_output[i]),2)
        self.RMSE = np.sqrt(sum / m)
        return self.RMSE
    