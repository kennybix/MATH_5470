import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy.optimize import minimize
from interpolation_models.kriging_Multilayer_CKL import halton_sequence as hs
from interpolation_models.kriging_Multilayer_CKL import lhs
from interpolation_models.kriging_Multilayer_CKL import pattern_search
from interpolation_models.kriging_Multilayer_CKL import preprocessing
from interpolation_models.kriging_Multilayer_CKL import nearestPD as NPD



class Kriging:

    def __init__(self,x,y,kernels,theta0,weight0 = "",eps=1.48e-08):
        self.x = x
        self.y = y
        self.kernels = kernels
        self.theta0 = theta0
        self.eps = eps
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]

        if (weight0==""):
            weight0 = [1/len(self.kernels)] * len(self.kernels)
        weights = weight0
        for i in range(0,self.Nk-1):
            weights = np.vstack((weights,weight0))
        self.weights0 = weights

    def compute_distances(self,X1,X2): #squared euclidean distance
        n = X2.shape[0] #if not for this bug, I wont need to do the unnecessary transpose for K_s
        k = X1.shape[0]  #fix when all works well
        X1 = X1.reshape(k,1) #must reflect fix
        X2 = X2.reshape(n,1) #must reflect fix
        dists = np.zeros((n,k))
        for i in range(n):
            dists[i,:] = np.sqrt(np.sum((X1 - (X2[i,:]))**2, axis=1)) #effort to compute D instead
        # d = X1 - X2
        # dists = np.abs(d)
        return dists

    def compute_K(self,X,Y,kernel,theta):
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
        #weights input must be flattened
        weights = np.array_split(weights,len(weights))
        R = 1
        for i in  range(self.Nk):
            R_k = np.zeros((self.Ns,self.Ns))
            for j in range(len(kernels)):
                w = weights.pop(0)
                R_k += w * self.compute_K(X[:,i],Y[:,i],kernels[j],theta[i]) 
            R *= R_k                                                                                                                                                              
        return R

    def compute_Beta(self,R,y):
        self.F = np.ones(self.Ns)[:,np.newaxis]
        Ri = np.linalg.inv(R)
        FT = (self.F).T
        temp = (np.dot(FT,np.dot(Ri,self.F)))
        temp2 = (np.dot(FT,np.dot(Ri,y)))
        invtemp = np.linalg.inv(temp)
        Beta = np.dot(invtemp,temp2)
        return Beta

    def NLL(self, hyperparameter):
        theta = np.zeros(len(self.theta0))
        hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
        for i in range(len(self.theta0)):
            theta[i] = hyperparameter.pop(0)
        weights = np.concatenate(hyperparameter)

        y = self.y
        n = len(y)
        R = self.compute_R(self.x,self.x,theta,weights) + np.diag(self.eps*np.eye(self.Ns))
        try:
            Beta = self.compute_Beta(R,y)
            Y = (y - np.dot(self.F,Beta))

            c = np.linalg.inv(np.linalg.cholesky(R))      
            Ri = np.dot(c.T,c)
            self.sigma2 = 1.0/self.Ns * np.dot(Y.T,(np.dot(Ri,Y)))
            (sign, logdetR) = np.linalg.slogdet(R)  # returns the sign and natural logarithm of the determinant of an array
            sigma2 = self.sigma2
            self.detR = np.linalg.det(R)
            nll = 1.0/2.0 * ((self.Ns * np.log(self.sigma2)) + np.log(np.linalg.det(R)))
            if (nll == -np.inf or math.isnan(nll)):
                nll = np.inf
        except np.linalg.LinAlgError: 
            print("Error in Linear Algebraic operation")
            nll = np.inf
        return nll

    def constraint_func(self,hyperparameter):
        theta = np.zeros(len(self.theta0))
        hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
        for i in range(len(self.theta0)):
            theta[i] = hyperparameter.pop(0)
        weights = np.concatenate(hyperparameter)
        weights = weights.tolist()
        weights = np.array_split(weights,len(weights))
        w = []
        w_d = []
        for o in range(self.Nk):
            w.append([])
        for o in range(self.Nk):
            for j in range(len(self.kernels)):
                x = weights.pop(0)
                w[o].append(float(x))    
        w = np.array(w)   
        i = np.linspace(0,self.Nk-1,self.Nk)
        i = i.tolist()
        for t in range(len(i)):
            i[t] = int(i[t])
        w_d = w.sum(1)[i] - 1  
        # return (np.dot(w,w.T) - np.eye(self.Nk))
        return w_d

    '''
    Ignore putting constraints in place
    Use boundaries to froce the weights to be between 0 and 1
    Carry out the optimization
    When the result is obtained,
    Normalize the matrix columns and use to build model
    
    '''
    def something(self,t):
        u = np.linspace(0,self.Nk-1,self.Nk)
        return u

    def get_theta(self, hyperparameter):
        xk = hyperparameter

                
        theta = np.zeros(len(self.theta0))
        hyperparameter = np.array_split(hyperparameter,len(hyperparameter))
        for i in range(len(self.theta0)):
            theta[i] = hyperparameter.pop(0)
        weights = np.concatenate(hyperparameter)
        weights = weights.tolist()
        weights = np.array_split(weights,len(weights))
        w = []
        for o in range(self.Nk):
            w.append([])
        for o in range(self.Nk):
            for j in range(len(self.kernels)):
                x = weights.pop(0)
                w[o].append(float(x))    
        w = np.array(w)   
        # w_d = lambda i: w.sum(1)[i] - 1  
        # t = np.linspace(0,self.Nk-1,self.Nk)
        cons = ({'type':'eq','fun':self.constraint_func})

        bounds = []
        theta_bound = (0.00001,1000000000.0)
        weight_bound = (0.00001,1)

        for i in range(len(self.theta0)): 
            bounds.append(theta_bound)
        weight_f = (np.array(self.weights0)).flatten()
        for j in range(len(weight_f)):
            bounds.append(weight_bound)

        res = minimize(self.NLL,xk,method='SLSQP',bounds=bounds,constraints=cons,options={'ftol':1e-20,'disp':True,'maxiter': 500})
        # res = minimize(self.NLL,xk,method='SLSQP',bounds=bounds,options={'disp':True})
        optimal_hyperparameter = res.x
        optimal_hyperparameter = np.array_split(optimal_hyperparameter,len(optimal_hyperparameter))
        self.theta = self.theta0
        for i in range(len(self.theta0)):   
            self.theta[i] = optimal_hyperparameter.pop(0)

        weights_vector = optimal_hyperparameter
        weights_vector = np.array_split(weights_vector,len(weights_vector))
        weights = []
        for p in range(self.Nk):
            weights.append([])
        for l in range(self.Nk):
            for m in range(len(self.kernels)):
                w = weights_vector.pop(0)
                weights[l].append(float(w))
        weights = np.array(weights)
        self.weights = weights
        # self.weights = weights / np.sum(weights,axis=1,keepdims=True)
        print(self.theta)
        print(self.weights)

    def train(self):
        weights0 = (np.array(self.weights0)).flatten()
        weights0 = weights0.tolist()
        hyperparameter = self.theta0 + weights0
        self.get_theta(hyperparameter)


    def K_components(self,Xt):
        K = 1 #initialization
        K_s = 1 #initialization
        
        theta = self.theta
        weights = self.weights
        weights = weights.flatten()

        Xt = np.array(Xt)
        Xtrain = np.copy(self.x)
        Xtest = np.copy(Xt)

        weights = np.array_split(weights,len(weights))

        k = self.Nk #dimension of the sample 
        for i in range(k): #gets the number of dimensions 
            K_k = np.zeros((self.Ns,self.Ns))
            K_s_k = np.zeros((Xt.shape[0],self.Ns)) #continue
            for j in range(len(self.kernels)):
                w = weights.pop(0)
                kernel = self.kernels[j]
                K_k += w * (self.compute_K(Xtrain[:,i],Xtrain[:,i],kernel,theta[i]))
                K_s_k += w * self.compute_K(Xtrain[:,i],Xtest[:,i],kernel,theta[i])
            K *= K_k
            K_s *= K_s_k
        return K,K_s.T  #had to use transpose for this to work, take note

    
    def predict(self,testdata):
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
    def computeNRMSE(self,y_exact):
        m = len(self.y_output) 
        sum = 0.0
        for i in range(m):
            sum += np.power((y_exact[i] - self.y_output[i]),2)
        self.RMSE = np.sqrt(sum / m)
        self.RMSE /= (np.max(y_exact)-np.min(y_exact))
        return self.RMSE

