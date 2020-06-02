import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy.optimize import minimize
from interpolation_models.core import halton_sequence as hs
from interpolation_models.core import lhs
from interpolation_models.core import pattern_search
from interpolation_models.core import preprocessing
from interpolation_models.core import nearestPD as NPD
from interpolation_models.core import constrNMPy as cNM
import cma



class Kriging:

    def __init__(self,x,y,kernel,theta0,optimizer="CMA-ES",optimizer_noise=1.0,eps=1.48e-08,restarts=1):
        self.x = x
        self.y = y
        self.kernel = kernel
        self.theta0 = theta0
        self.optimizer = optimizer
        self.optimizer_noise = optimizer_noise
        self.eps = eps
        self.restarts = restarts
        self.Ns = self.x.shape[0]
        self.Nk = self.x.shape[1]

        #self.x_normal, self.y_normal, self.x_mean,self.y_mean,self.x_std, self.y_std = preprocessing.standardize(self.x,self.y)
        # self.x_normal, self.y_normal, self.x_min,self.y_min,self.x_max, self.y_max = preprocessing.norm(self.x,self.y)

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
        kernel = self.kernel
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

    def compute_R(self,X,Y,theta):
        kernel = self.kernel
        R = 1
        for i in  range(self.Nk):
            R *= self.compute_K(X[:,i],Y[:,i],kernel,theta[i])                                                                                                                                                               
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

        
    def _HJfun_obj(self, theta):
        y = self.y
        n = len(y)
        R = self.compute_R(self.x,self.x,theta)
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
        return float(nll)

    def _checkPositive(self, x):
        # check that all design variables are > 0
        Ndv = len(x)
        count = 0 
        for i in range(Ndv):
            x[i] = float(x[i])
            if x[i] > 0:
                # positive entry, increment count
                count = count + 1
        if count == Ndv:
            self.constraint = True
        else:
            self.constraint = False 

    def _explore(self, xk, sk, delta, minf, rhok, fxk): 
        n = len(xk)
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1
            ski = sk + delta*e
            xk = np.array(xk)
            xki = xk.flatten() + ski 
            fxki = self._HJfun_obj(xki) 

            if fxki < minf:
                rhok = fxk - fxki 
                minf = fxk
                i
                sk = ski
            else: 
                ski = sk - delta*e
                xk = np.array(xk)
                xki = xk.flatten() + ski
                self._checkPositive(xki) 
                if self.constraint:
                    fxki = self._HJfun_obj(xki)
                    if fxki < minf:
                        rhok = fxk - fxki 
                        minf = fxki
                        sk = ski 
        return rhok, sk, minf

    def _hj(self, xk, delta, rhok, sk, limDelta, shrink):
        k = 0
        while delta > limDelta: 
            fxk = self._HJfun_obj(xk)
            minf = fxk
            f0 = minf
            xk = np.array(xk)
            test = xk.flatten() + sk
            self._checkPositive(test)
            if rhok > 0 and self.constraint:
                trialF = self._HJfun_obj(xk + sk)
                if trialF != np.inf:
                    rhok = fxk - trialF
                    minf = trialF
                    rhok, sk, minf = self._explore(xk, sk, delta, minf, rhok, fxk)
                else:
                    rhok = 0
                    sk = 0
            else: 
                sk = 0
                rhok = 0
                minf = fxk 

                rhok, sk, minf = self._explore(xk, sk, delta, minf, rhok, fxk)

                if minf >= f0 and minf != np.inf:
                    delta = shrink*delta

            # Pattern move
            xk = np.array(xk)
            xk = xk.flatten() + sk
            k = k+1
            if minf == np.inf: 
                xk = xk + delta

        self.theta = xk
        self.likelihood = self._HJfun_obj(xk)

    def get_theta(self, theta0):
        xk  = theta0     
        bounds = []
        theta_bound = (0.0001,100000.0)
        for i in range(len(self.theta0)):
            bounds.append(theta_bound)  

        if (self.optimizer=="CMA-ES"):
            xopts, es = cma.fmin2(self._HJfun_obj,xk,self.optimizer_noise,{'bounds':[0.001,100000.0],'verbose':-9},restarts=self.restarts)
            theta = xopts
            # es = cma.CMAEvolutionStrategy(xk,self.optimizer_noise,{'bounds':[0.001,100000.0]},restarts=self.restarts)
            # #es.opts.set({'bounds':theta_bound})
            # es.optimize(self._HJfun_obj)
            # res1 = es.result

            # #hack
            # if res1.xbest is None:
            #     theta = res1.xfavorite
            # else:
            #     theta = res1.xbest
            self.theta = theta

        elif self.optimizer == "hooke-jeeves":
            delta = 1
            rhok = 0 
            sk = np.zeros(self.Nk)
            sk = np.array(sk)
            limDelta = 1e-4
            shrink = 0.5
            self._hj(xk, delta, rhok, sk, limDelta, shrink)
        
        elif self.optimizer == "nelder-mead-c":
            LB = [0.00000000000000001]*len(self.theta0)
            UB = [1000000000.0]*len(self.theta0)
            res = cNM.constrNM(self._HJfun_obj,xk,LB,UB,full_output=True)
            self.theta = res['xopt']

        elif self.optimizer == "nelder-mead" or "SLSQP" or "COBYLA" or "TNC":
                res1 = minimize(self._HJfun_obj,xk,method=self.optimizer,bounds=bounds,options={'ftol':1e-20,'disp':False})
                self.theta = res1.x
        self.likelihood = self._HJfun_obj(self.theta)

    def train(self):
        self.get_theta(self.theta0)


    def K_components(self,Xt):
        K = 1 #initialization
        K_s = 1 #initialization
        K_ss = 1 #initialization
        theta = self.theta
        Xt = np.array(Xt)
        Xtrain = np.copy(self.x)
        Xtest = np.copy(Xt)
        k = self.Nk #dimension of the sample 
        for i in range(k): #gets the number of dimensions    
            K *= (self.compute_K(Xtrain[:,i],Xtrain[:,i],self.kernel,theta[i]) + np.diag(self.eps*np.eye(self.Ns)))

            K_s *= self.compute_K(Xtrain[:,i],Xtest[:,i],self.kernel,theta[i])
            K_ss *= self.compute_K(Xtest[:,i],Xtest[:,i],self.kernel,theta[i])
        self.K_ss = K_ss
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
        self.variance = np.diag(self.K_ss) - np.sum(self.Lk**2,axis=0) # variance
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

        # hack 
        # if (self.y_output.shape[1]==1):
        #     self.y_output = self.y_output.reshape(m,)

        sum = 0.0
        for i in range(m):
            sum += np.power((y_exact[i] - self.y_output[i]),2)
        self.RMSE = np.sqrt(sum / m)
        self.RMSE /= (np.max(y_exact)-np.min(y_exact))
        return self.RMSE
